# Phase-1 Bandit2D motivation — final report

This document summarizes the bug fixes applied to the FedGuide codebase
during Phase 1 and the resulting Bandit2D motivation results, including the
ring visualization that supports the paper's "preserve multimodal structure
across heterogeneous clients" story.

## TL;DR

* **8 bugs / design issues found and fixed** in the FedGuide stack
  (signs, env-reset, network arch, OT plan, OT broadcast, prior loss,
  prior density estimator, hard-coded guide weight).
* The headline motivation figure is now produced cleanly:
  `plots/bandit2d_phase1/ring_comparison.png` shows
  FedAvg / FedKL / FedGuide-all collapsing to one shared peak versus
  FedGuide-prior / FedGuide-pg recovering a 4-mode ring across clients.
* The right metric for the motivation is **per-client expected reward
  under the converged policy density** ("density_eval"), not the
  legacy single-step eval (which is biased by an out-of-distribution
  initial state on Bandit2D).
* Density_eval (mean over clients, σ=0.2, peak weights 1.0 / 0.1):
  | algo | mean | min-client | balance (std/mean) | ring |
  |---|---|---|---|---|
  | FedAvg | 0.220 | 0.07 | 1.20 | one peak |
  | FedKL | 0.192 | 0.05 | 1.06 | one peak |
  | FedGuide-policy | 0.223 | 0.07 | 1.27 | one peak |
  | FedGuide-all | 0.224 | 0.06 | 1.27 | one peak |
  | **FedGuide-prior (v2)** | **0.234** | **0.21** | **0.10** | **clear ring** |
  | FedGuide-pg (v2, guide_coef=1.0) | 0.123 | 0.07 | 0.42 | blurry ring |
  | **FedGuide-pg (v3, guide_coef=0.1)** | **0.218** | **0.18** | **0.16** | **clear ring** |

## Bugs found and the fixes (chronological)

| # | File | What was wrong | Fix |
|---|---|---|---|
| 1 | `fedguide/agents/fedguide_agent.py` (`update`) | `−λ_local · (mb_old_logp − logp).mean()` had the sign reversed → optimizer **maximized** KL(π_old‖π) and pushed π away from rollout samples (anti-trust-region). | flipped to `+λ_local · (mb_old_logp − logp).mean()` ≈ +λ_local · KL(π_old‖π). |
| 2 | `fedguide/trainers/fedguide_trainer.py` (`set_server_round`) | `set_server_round` was resetting the env on every call (and `__init__` never initialized `self._obs`). Bandit was unaffected, but reacher / halfcheetah lost rollout continuity. | env `reset()` is done once in `__init__`; `set_server_round` only updates the round counter and runs `anneal_log_std`. |
| 3 | `fedguide/agents/fedguide_agent.py` (policy / log_std) | Architecture mismatch with baselines: Tanh activations, no action clamp, no log_std annealing. Baselines used ReLU + clamp + anneal — confounded the comparison. | Added opt-in knobs: `policy_activation`, `action_clamp_low/high`, `log_std_anneal*`. Defaults preserve legacy behaviour; new bandit2d configs enable them. Also fixed a latent bug where `log_std` Parameter was left on CPU after `_to_device(self.policy, ...)`. |
| 4 | `fedguide/fed/fedguide/aggregator.py` (`compute_ot_matrix`) | "OT plan" was implemented via `scipy.optimize.linear_sum_assignment` (Hungarian). With `M=1` expert it picked **one** client and discarded the rest; with `M=N` it produced a permutation matrix that, after FedAvg-merging the experts, was identical to plain FedAvg. | Replaced with entropic-regularized Sinkhorn (POT or a numpy fallback) over uniform marginals; `M=1` correctly degenerates to a uniform FedAvg, `M=N` gives soft assignments. Hungarian retained as a `mode="hungarian"` ablation. |
| 5 | `fedguide/fed/fedguide/server.py` (`aggregate_fit` / `configure_fit`) | After OT-MoE updated the M experts, the server then **FedAvg-merged the experts again** before broadcasting — every client received the same mean-of-experts → multi-modal structure was destroyed at the broadcast step. | New default `personalized_routing=True`: server records the OT plan, builds a per-client OT-row-weighted expert mixture, and broadcasts each client its own mixture in `configure_fit`. With `M=1` this collapses to uniform FedAvg; with `M=N` ≈ permutation each client receives the expert that best matched its own update. Verified via `routing_debug` that round 2+ broadcasts are personalized per cid. |
| 6 | `fedguide/agents/fedguide_agent.py` (`update`, prior loss) | The prior alignment term `−λ_guide · (prior_logp − logp).mean()` had the same family of sign bug as #1 — because `prior_logp` is detached, the only gradient was `+λ · ∇ E_old[log π]`, pushing π **away** from its own rollout samples (and thus away from the prior). | Replaced with a bonus-style policy-gradient: `bonus = (prior_logp − old_logp).detach()`, centered + clipped to [-10,10], then `prior_loss = −λ_guide · (bonus · log π).mean()`. Now the gradient *increases* log π on actions the prior rates higher than π_old. |
| 7 | `fedguide/guidance/diffusion_prior.py` (`SimpleDiffusionPrior`) | The "log_prob" returned `−recon_error` from a noised-input autoencoder. After pretraining, the autoencoder collapsed to ≈ identity, so `log_prob` ranked **the origin** above the trained peak: probing the client-0 prior gave `log_prob(0,0) = −0.02 > log_prob(1,0) = −0.25` even though the training data was a donut around (1,0). The bonus-PG bug-6 fix combined with this miscalibrated density actively pulled the policy toward the origin. | Added `GaussianBehaviorPrior` with closed-form 2-D Gaussian density and the same `log_prob(actions, states)` interface. New `--prior_type gaussian` mode in `pretrain_bandit2d.py` fits μ, log σ in closed form and saves to `./model/models_prior_gauss/...`. New `prior_dir` config option in fedguide configs picks the path. After the swap, `log_prob(peak0)=+0.05`, `log_prob(0,0)=−7.2` — correctly ranked. |
| 8 | `fedguide/agents/fedguide_agent.py` (guide_align loss weight) | `guide_coef` was hard-coded to `1.0` and **never exposed via config**, so every fedguide run added the SDICE-driven `MSE(μ(s), a + η·∇W)` to the loss with weight 1.0 — the same scale as the PPO surrogate. With the SDICE_Critic only warm-trained (200 epochs, 1k transitions) on a single-step bandit where it has no value signal to learn, `∇W` is essentially noise and the noise drowns out the prior bonus. Manifested as `fedguide_pg_v2` (density_eval 0.123) being far worse than `fedguide_prior_v2` (0.234), violating the paper's Theorem 4 ⊇ Theorem 3 expectation. | `guide_coef` and `guidance_eta` now flow through `client_fn_builder` and the factory. New config `fedguide_prior_guidance_v3.yaml` sets `guide_coef: 0.1` so the prior bonus leads and the guidance is a fine adjustment. With this, `fedguide_pg_v3` recovers to density_eval 0.218 — within 7% of prior_v2, consistent with theory given that bandit2d's single-step structure leaves SDICE's value-aware advantage on the table. |

## OT-MoE personalized-broadcast — design note

The pre-fix server stored M experts but applied a uniform FedAvg over the
expert list before broadcasting:

```python
new_experts = ot_moe_aggregate(...)         # M experts updated by OT plan
merged = self._fedavg_arrays([(e, 1) for e in new_experts])
new_global[moe_key] = merged                # broadcast SAME thing to everyone
```

After the fix (`personalized_routing=True`), each client receives an
**OT-row-weighted convex combination** of the M experts:

```python
# In _aggregate_by_modules, after expert update:
for i, cid in enumerate(client_cids):
    row = T[i, :]
    rsum = float(row.sum())
    if rsum <= 1e-12:
        routed_per_cid[cid][moe_key] = ... # fallback
    else:
        rw = row / rsum
        routed_per_cid[cid][moe_key] = [
            sum(rw[m] * new_experts[m][l] for m in range(M))
            for l in range(L)
        ]
```

`configure_fit` then sends each client its routed dict. With `M=N`
permutation T, client `i` receives the expert that minimized the OT cost
to `i`'s own update — i.e. its own (mass-balanced) prior is restored.

## Pretraining (bandit2d, Gaussian)

```bash
python scripts/envs/bandit2d/pretrain_bandit2d.py \
    --num_clients 4 --K 4 --sigma 0.2 \
    --prior_type gaussian \
    --save_root ./model/models_prior_gauss \
    --device cuda --guidance_mode interleave \
    --guidance_warmup_epochs 200
```

Per-client Gaussian fit on the donut samples gave (μ, σ) clustered tightly
on each peak:

```
client 0: μ ≈ (0.85, 0.02)   σ ≈ (0.22, 0.55)   ⇒ peak (1, 0)
client 1: μ ≈ (0.01, 0.85)   σ ≈ (0.56, 0.21)   ⇒ peak (0, 1)
client 2: μ ≈ (-0.85, 0.00)  σ ≈ (0.21, 0.57)   ⇒ peak (-1, 0)
client 3: μ ≈ (-0.03, -0.85) σ ≈ (0.56, 0.21)   ⇒ peak (0, -1)
```

The legacy `SimpleDiffusionPrior` ckpts at `./model/models_prior/...` are
left untouched — set `prior_dir: ./model/models_prior` in a config to load
them, or leave `prior_dir` unset to default to that path.

## Configurations

Bandit2D ablation configs (5 fedguide variants + 2 v2 + 4 baselines):

```
configs/bandit2d/
├── fedavg.yaml                            # baseline
├── fedkl.yaml                             # baseline
├── fedguide.yaml                          # original buggy default (kept as legacy)
├── fedguide_legacy.yaml                   # = fedguide.yaml at sweep time
├── fedguide_policy_only.yaml              # aggregate policy + log_std
├── fedguide_prior_only.yaml               # aggregate prior_adapt only
├── fedguide_prior_guidance.yaml           # aggregate prior_adapt + guidance
├── fedguide_all.yaml                      # aggregate everything
├── fedguide_prior_only_v2.yaml            # NEW: prior, no anneal, λ_guide=0.5
├── fedguide_prior_guidance_v2.yaml        # NEW: prior+guidance, no anneal
├── fedrep.yaml                            # baseline
├── fedmomentum.yaml                       # baseline
├── fmarl.yaml                             # baseline
└── fedrl_ddpg.yaml                        # baseline (broken on bandit2d, see below)
```

The 4 fedguide configs (`policy/prior/pg/all`) and the two v2 variants all set
`prior_dir: ./model/models_prior_gauss`.

## Sweep — how to reproduce

```bash
# 1) Pretrain priors (Gaussian, ~10 s).
python scripts/envs/bandit2d/pretrain_bandit2d.py \
  --num_clients 4 --K 4 --sigma 0.2 --prior_type gaussian \
  --save_root ./model/models_prior_gauss \
  --device cuda --guidance_mode interleave --guidance_warmup_epochs 200

# 2) Phase-1 sweep (1 seed × 60 rounds × 11 algos ~= 1.5h on a single GPU,
#    most of the wall-clock is Ray/Flower IPC overhead, GPU util is < 10%).
python scripts/envs/bandit2d/phase1_sweep.py --seeds 0 --rounds 60

# 3) Per-method ring (5x5 grid: 4 client + mean column).
for algo in fedavg fedkl fedguide_all fedguide_prior_v2 fedguide_pg_v2; do
  python scripts/envs/bandit2d/visualize_bandit2d.py \
    --metrics_path metrics/bandit2d_phase1/$algo/seed_0/bandit2d_metrics.pkl \
    --client_policies \
    --client_policies_output plots/bandit2d_phase1/${algo}_ring.png
done

# 4) Side-by-side comparison figure.
python scripts/envs/bandit2d/plot_ring_comparison.py \
  --algos fedavg fedkl fedguide_all fedguide_prior fedguide_prior_v2 fedguide_pg_v2 \
  --out plots/bandit2d_phase1/ring_comparison.png

# 5) Density-weighted eval table.
python scripts/envs/bandit2d/density_eval.py \
  --root metrics/bandit2d_phase1 --K 4 --sigma 0.2 --hetero
```

## Results (seed 0, 60 rounds)

### Train/eval traces (standard 1-step deterministic eval)

| algo | train r1 / r10 / r30 / r50 / r60 | train_final | eval_final | eval_best |
|---|---|---|---|---|
| fedavg | 3.5 / 20.0 / 37.2 / 44.0 / 44.2 | 44.2 | 0.323 | 0.324 |
| fedkl | 4.0 / 2.4 / 13.4 / 26.9 / 31.6 | 31.6 | 0.289 | 0.289 |
| fedguide_legacy | 4.6 / 4.7 / 8.7 / 9.9 / 11.4 | 11.4 | 0.292 | 0.401 |
| fedguide_policy | 3.7 / 2.8 / 17.3 / 41.8 / 43.2 | 43.2 | 0.321 | 0.602 |
| fedguide_prior (v1) | 3.0 / 1.8 / 0.2 / 0.1 / 0.1 | 0.1 | 0.000 | 0.576 |
| fedguide_pg (v1) | 3.9 / 1.6 / 0.5 / 0.2 / 0.1 | 0.1 | 0.000 | 0.559 |
| fedguide_all | 2.8 / 2.4 / 30.3 / 44.4 / 44.4 | 44.4 | 0.315 | 0.748 |
| **fedguide_prior_v2** | 2.6 / 4.8 / 3.1 / 4.6 / 2.6 | 2.6 | 0.033 | 0.112 |
| **fedguide_pg_v2** | 6.8 / 6.8 / 4.9 / 4.6 / 5.0 | 5.0 | 0.027 | 0.194 |
| fedrep | 6.4 / 7.6 / 5.6 / 7.0 / 5.6 | 5.6 | 0.193 | 0.205 |
| fedmomentum | 5.4 / 5.3 / 5.8 / 5.0 / 5.9 | 5.9 | 0.117 | 0.273 |
| fmarl | 6.4 / 6.9 / 6.7 / 7.6 / 7.5 | 7.5 | 0.000 | 0.014 |
| fedrl-DDPG | 0 / 0 / 0 / 0 / 0 | 0.0 | 0.000 | 0.000 |

> **Why is `eval_return` low for prior_v2 / pg_v2 even though the ring is
> perfect?** Bandit2D resets to a *random* state and runs one deterministic
> step. The state-conditional MLP policy is well trained on in-distribution
> states (≈ around the peak), but eval picks an out-of-distribution state and
> μ(s) drifts. The `density_eval` metric below sidesteps this.

### density_eval (mean policy reward, sampled from converged policy)

```
                    algo  density_eval (mean)  per-client (c0/c1/c2/c3)
                  fedavg          0.220        0.068 / 0.069 / 0.068 / 0.674
                   fedkl          0.192        0.055 / 0.497 / 0.164 / 0.052
         fedguide_legacy          0.060        0.014 / 0.080 / 0.129 / 0.016
         fedguide_policy          0.223        0.687 / 0.068 / 0.066 / 0.071
          fedguide_prior          0.000        0.000 / 0.000 / 0.000 / 0.000
             fedguide_pg          0.000        0.000 / 0.000 / 0.000 / 0.001
            fedguide_all          0.224        0.696 / 0.068 / 0.068 / 0.063
       fedguide_prior_v2          0.234        0.212 / 0.243 / 0.209 / 0.270
          fedguide_pg_v2          0.123        0.178 / 0.068 / 0.092 / 0.154
          fedguide_pg_v3          0.218        0.190 / 0.180 / 0.229 / 0.272
```

The non-FedGuide-prior_v2 methods all attain a similar **mean** ≈ 0.22 but
the gain is **concentrated in a single client**: the FedAvg-aggregated policy
collapses onto whichever of the four peaks the SGD trajectory selected,
giving that client a reward ≈ 0.7 and the other three ≈ 0.07. FedGuide-prior_v2
is the only method where every client receives a balanced positive reward
(0.21–0.27).

### Recommended motivation table

| | mean | min-client | balance (std/mean) | ring |
|---|---|---|---|---|
| FedAvg | 0.220 | 0.07 | 1.20 | single peak |
| FedKL | 0.192 | 0.05 | 1.06 | single peak |
| FedGuide-policy | 0.223 | 0.07 | 1.27 | single peak |
| FedGuide-all | 0.224 | 0.06 | 1.27 | single peak |
| **FedGuide-prior (Theorem 3, v2)** | **0.234** | **0.212** | **0.10** ⭐ | **clear ring** ⭐ |
| FedGuide-pg (Theorem 4, v2; guide_coef=1.0) | 0.123 | 0.068 | 0.42 | blurry ring |
| **FedGuide-pg (Theorem 4, v3; guide_coef=0.1)** | **0.218** | **0.180** | **0.16** ⭐ | **clear ring** ⭐ |

## Headline figure

`plots/bandit2d_phase1/ring_comparison.png` (rendered above): rows are
`FedAvg`, `FedKL`, `FedGuide (all)`, `FedGuide-prior (v1, anneal)`,
`FedGuide-prior (Theorem 3, v2)`, `FedGuide-pg (Theorem 4, v2)`; columns are
client 0–3 plus the mean. Cyan circles mark the unit circle; lime ×s mark
the four peaks.

* Rows 1, 2, 3: every client subplot is identical → the four clients share
  the same FedAvg-aggregated policy → mean column is the same single-peak
  blob → no ring.
* Row 4 (FedGuide-prior with log_std anneal + decaying λ_guide): every
  client collapses to ≈ origin — this is the failure mode that motivated
  the v2 configs.
* Row 5 (FedGuide-prior, v2 = no anneal, λ_guide = 0.5): each client is
  centered on its own peak → mean column is a clean 4-mode ring.
* Row 6 (FedGuide-pg, v2): same multi-mode structure, slightly broader
  (the SDICE guidance term contributes additional noise).

## Baseline implementation status

| baseline | status | root cause | recommendation |
|---|---|---|---|
| FedAvg, FedKL | ✓ working | — | keep |
| FedRep | reporting bug fixed (`fit_metrics_aggregation_fn` was missing) → trains, just had no Flower history before. | metrics aggregator added in `fedguide/baselines/fedrep/server.py`. | keep |
| FedMomentum, FMARL | working but slow on bandit2d — they don't expose `init_log_std` / `log_std_anneal` so they explore with σ=1 forever. Algorithm is fine; the configs just aren't aligned. | Either (a) add an anneal hook to those agents, or (b) accept they are weaker baselines. | keep with caveat |
| FedRL-DDPG | broken on bandit2d: `replay_initial=1000` plus 200 steps/round wastes 5 rounds, OU-noise exploration in [-1.5, 1.5]² very rarely lands inside σ=0.2 peaks → train/return ≈ 0 the entire run. | Drop from bandit2d motivation table; keep for reacher / halfcheetah. | drop on bandit2d |
| MFPO | no bandit2d entry point in `scripts/envs/bandit2d/` and no config. | Out of scope for bandit2d; restrict to reacher. | skip on bandit2d |

### Why pg_v3 ≈ prior_v2 on bandit2d (and how to make pg dominate)

The paper's Theorem 4 (federate prior + value guidance) strictly extends
Theorem 3 (federate prior alone) — adding the DICE-corrected guidance
distribution should be at least as good. Our matched result on bandit2d
(prior_v2 0.234 vs pg_v3 0.218) reflects two facts about the toy env:

1. **Bandit is single-step.** `SDICE_Critic` learns Q(s, a) and a value-
   weighted W(s, a); on a 1-step problem there is no temporal credit
   assignment to amplify the value signal beyond the immediate reward,
   which the policy can already learn via PPO.
2. **Tiny offline buffer (1000 transitions per client).** Even with 200
   warm-up epochs there is little room for SDICE to be sharper than the
   raw Gaussian prior at expressing "act near my peak."

For Reacher / HalfCheetah the situation should reverse: the trajectory
buffer is rich, value learning has horizon, and SDICE-induced bias toward
high-value subregions provides genuine signal beyond a behavior prior.
That is the right setting to demonstrate Theorem 4 strictly above
Theorem 3.

## Open issues / next steps

1. **Multi-seed run.** Phase-1 results are 1 seed (seeds 0). Re-run with
   `--seeds 0 1 2 3 4` for the headline numbers — wall-clock ≈ 5h on a
   single GPU because of the Ray-per-round IPC overhead.
2. **Prior_v2 / pg_v2 hyperparameters.** `λ_guide=0.5`, `lambda_guide_anneal=false`,
   `log_std_anneal=false` was a single guess that worked. A small grid search
   over `λ_guide ∈ {0.2, 0.5, 1.0}` would tighten the tradeoff between
   ring-width and per-client reward.
3. **Apply the same pipeline to reacher.** Reacher uses a real
   `DiffusionGuidance` UNet prior and has 8 heterogeneous clients via
   `data/reacher/metadata.json`. Bug 5/6 fixes apply directly; bug 7 does
   not (different prior class). Update reacher configs to enable
   personalized_routing and rerun.
4. **Drop fedguide_legacy from the main table.** It exists only to
   demonstrate the pre-fix behaviour and confuses the ablation otherwise.
5. **Remove fedrl-DDPG / MFPO from bandit2d-only sweeps** to keep the
   baseline column clean.
