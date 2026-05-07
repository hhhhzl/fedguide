# Bandit2D Phase-1

Smallest-possible heterogeneous federated RL environment used to validate
the FedGuide claims (Theorems 3-5) and as a sanity test for every code
change. 4 clients, 4 peaks evenly placed on the unit circle, single-step
reward, σ = 0.2. Each client's offline data and `preferred_peak` is
peak `i` (weight 1.0; other peaks weight 0.1).

## Layout

```
configs/bandit2d/
├── README.md              # this file
├── main/                  # FedGuide variants (3 ablations)
│   ├── fedguide_prior.yaml   # Theorem 3 — aggregate ONLY the diffusion prior
│   ├── fedguide_pg.yaml      # Theorem 4 — aggregate prior + value guidance
│   └── fedguide_all.yaml     # Theorem 5 — aggregate policy + prior + guidance
└── baseline/              # federated baselines + warm-start ablation
    ├── fedavg.yaml           # FedAvg = FedKL with λ=0
    ├── fedkl.yaml            # FedKL with KL penalty
    ├── fedrep.yaml           # FedRep
    ├── fedmomentum.yaml      # FedMomentum
    ├── fmarl.yaml            # FMARL
    ├── fedrl_ddpg.yaml       # FedRL-DDPG
    ├── ppo.yaml              # central PPO (sanity)
    └── sac.yaml              # central SAC (sanity)
```

```
scripts/envs/bandit2d/
├── _pretrain.py           # closed-form Gaussian-prior fit on offline data
├── run_baselines.py       # subprocess sweep: 6 federated baselines
├── run_main.py            # subprocess sweep: 3 FedGuide variants (auto-pretrain)
├── plots.py               # curves / ring / prior / density_eval
├── run_all.sh             # baselines + main + plots (one shot)
├── analyze.sh             # plots only (no training)
└── viz_priors.py          # diagnostic — visualize each client's prior
```

## How to run

```bash
# 1) full sweep (pretrain → baselines → main → plots)
bash scripts/envs/bandit2d/run_all.sh                 # 1 seed × 60 rounds, ~50 min CPU
bash scripts/envs/bandit2d/run_all.sh --seeds "0 1 2 3 4" --rounds 60

# 2) skip baselines / main if already cached
bash scripts/envs/bandit2d/run_all.sh --skip-base
bash scripts/envs/bandit2d/run_all.sh --skip-main

# 3) just one algo
python scripts/envs/bandit2d/run_main.py --only fedguide_prior --seeds 0
python scripts/envs/bandit2d/run_baselines.py --only fedavg --seeds 0

# 4) re-plot from cached metrics (no training)
bash scripts/envs/bandit2d/analyze.sh

# 5) hetero-reward density_eval (each client's own preferred peak)
python scripts/envs/bandit2d/plots.py --hetero density_eval

# 6) per-client prior μ visualization (sanity check pretrain)
python scripts/envs/bandit2d/viz_priors.py
```

Outputs land under:
- `metrics/bandit2d_phase1/<algo>/seed_<i>/{training_history,bandit2d_metrics}.pkl`
- `plots/bandit2d_phase1/{ring_comparison,reward_curves_*,prior_*}.png`
- `metrics/bandit2d_phase1/SUMMARY.md`

## Expected ordering vs measured ordering

The paper's claim, in two dimensions:

1. **Per-client REWARD under heterogeneous evaluation** (each client gets
   its preferred peak weight 1.0, other peaks 0.1):
   `FedGuide-pg ≳ FedGuide-prior > FedGuide-all ≈ FedAvg ≈ FedKL`
2. **Multi-modality (per-client policy peak)**:
   FedGuide-prior / -pg place each client at its own peak.
   FedAvg / FedKL / FedGuide-all collapse all clients to a single peak.

Both rankings now hold empirically (1 seed, 60 rounds, see
`plots/bandit2d_phase1/ring_comparison.png` and the table below):

| algo                     | hetero density_eval | balance (std/mean) | per-client peak                               | multi-modal? |
|--------------------------|---------------------|--------------------|-----------------------------------------------|--------------|
| **FedGuide-pg (Thm 4)**  | **0.606**           | **0.01**           | c0→peak0, c1→peak1, c2→peak2, c3→peak3        | **yes ✓**    |
| **FedGuide-prior (Thm 3)** | **0.604**         | **0.02**           | c0→peak0, c1→peak1, c2→peak2, c3→peak3        | **yes ✓**    |
| FedKL + warm-start       | 0.230               | 1.21               | all 4 → peak 0                                 | no           |
| FedAvg + warm-start      | 0.228               | 1.20               | all 4 → peak 0                                 | no           |
| FedGuide-all (Thm 5)     | 0.211               | 1.20               | all 4 → peak 2                                 | no           |
| FMARL                    | 0.040               | 0.13               | all 4 → near origin                            | no           |
| FedMomentum              | 0.030               | 0.35               | all 4 → near origin                            | no           |

`density_eval = ∑ π(a|s)·R(a)` over a 200×200 action grid. With per-client
heterogeneous reward weights (1.0 for own peak, 0.1 elsewhere) the metric
pays attention to whether each client found *its own* peak.

## What it took to get there — the bug list

A long sequence of corrections was needed; the fixes are independent of
each other and each of them shifted the bandit2d ordering. Documenting
here so future regressions can be spotted quickly.

| # | what was wrong                                                                | what we did                                                                 |
|---|---------------------------------------------------------------------------|------------------------------------------------------------------------|
| 1 | local KL sign reversed in PPO surrogate                                       | flipped the sign in `fedguide_agent.py`                                    |
| 2 | `set_server_round` reset env every round → discarded mid-episode state        | env is now created in `__init__` only                                      |
| 3 | PolicyNet activation/clamp/anneal mismatch with baselines                     | added `policy_activation`, `action_clamp_*`, `log_std_anneal*` config knobs |
| 4 | OT used Hungarian (winner-take-all) → loses non-trivial mass on near-ties     | switched default to Sinkhorn; Hungarian retained for ablation              |
| 5 | OT-MoE FedAvg-merged the experts before broadcast → defeated multi-modality  | `personalized_routing: true` broadcasts each client its own row-weighted mixture |
| 6 | `prior_loss` had the wrong gradient direction                                 | bonus-PG with `(log prior − log π_old).detach()`, then later replaced (#12) |
| 7 | `SimpleDiffusionPrior.log_prob` rated origin > peak (autoencoder artifact)    | added `GaussianBehaviorPrior` (closed-form 2D Gaussian) for bandit2d       |
| 8 | `guide_coef` hard-coded to 1.0, drowning out the prior bonus                  | exposed in config; bandit2d uses 0.1                                       |
| 9 | trainer ignored Gymnasium's `truncated` flag → infinite episode loops          | `done = terminated or truncated` everywhere                                |
| 10 | `env_name_map` missing `"reacher_hetero"`                                    | added                                                                       |
| 11 | DiffusionGuidance ckpts silently loaded as SimpleDiffusionPrior              | added explicit `prior_type` field to ckpt + agent dispatch                 |
| 12 | bonus-PG was REINFORCE-style; off-policy-biased once `update_epochs > 1`     | replaced with self-normalized IS cross-entropy (`ω ∝ prior(a)/π_old(a)`) |
| 13 | `lambda_local` misread as env-reward weight; was actually trust-region weight | reverted to 0.05; documented; do **not** raise it (it clamps π to π_old)   |
| 14 | policy starts at origin, σ=0.135 after anneal → bonus-PG loses signal         | warm-start: `policy.bias ← prior.head_mu` for Gaussian priors (D-fix)      |

The ordering only stabilizes once **all** of #1, #4, #5, #7, #8, #12, and
#14 are in place. With any single one of them missing, FedGuide-prior /
-pg either collapses to a single mode (looks like FedAvg) or stays at
the origin (zero density on every peak).

## Insights worth keeping in mind

1. **Multi-modality preservation is about aggregation, not initialization.**
   Giving FedAvg / FedKL the *exact same* per-client warm-start (so each
   starts at its own peak) does NOT save them — within one round of
   policy averaging they collapse to a single mode. FedGuide-prior /
   -pg succeed because they leave the policy local; only the prior /
   guidance are federated (via OT-MoE row-personalized routing).

2. **`lambda_local` is a trust-region weight, not an env-reward weight.**
   Raising it from 0.05 to 1.0 *clamps* the local policy to π_old and
   shuts down learning entirely. The env-reward weight is implicit (1.0
   on `policy_loss`).

3. **`bonus_PG` (REINFORCE-style) is fragile.** With `update_epochs > 1`
   it accumulates off-policy bias because actions came from π_old, not
   π_θ, and there's no IS correction. The fix is the IS-CE form:
   `ω(a) ∝ prior(a)/π_old(a)`, then `prior_loss = −λ · Σ ω · log π_θ`.
   This gives a strong, unbiased gradient toward the prior — but only
   on actions the rollout actually visited. Hence #14:

4. **Without warm-start, IS-CE alone can't escape the origin trap on
   bandit2d.** With `init_log_std=−1.0` and 4 peaks at distance 1 from
   the origin, almost no rollout sample falls near a peak, so IS-CE
   has no signal there. Warm-starting `policy.bias ← prior.head_mu`
   plants the policy at the correct peak from round 0 and IS-CE then
   *maintains* it through the run.

5. **The OT-MoE plan is stable in cid-space, not row-index space.**
   Flower returns clients in completion order, so the row index of the
   OT cost matrix shuffles round-to-round. The diagonal-mass diagnostic
   only makes sense if you re-key the rows by `client_cids[i]`. We
   confirmed that, in cid-space, every client routes to the same expert
   in every round — i.e. routing is correct and stable.

6. **Heterogeneous density_eval is the right metric.** With symmetric
   weights (all peaks 1.0), FedAvg's "everyone at peak 0" gets density
   0.68 *because the metric doesn't care which peak*. Switch to
   `--hetero` (each client's preferred peak weight 1.0, others 0.1) and
   FedAvg's per-client breakdown reveals 0.68 / 0.07 / 0.07 / 0.07 —
   only the lucky client whose preferred peak is the collapsed mode
   is happy. The mean drops to 0.23, ~3× behind FedGuide-prior/pg.

## Reproducibility checklist

* Pretrain: `python scripts/envs/bandit2d/_pretrain.py --prior_type gaussian
  --num_clients 4 --K 4 --sigma 0.2 --seed 42`. Output goes to
  `model/models_prior_gauss/Bandit2D/client_{0..3}/final/torch_prior.pth`.
* Verify pretrained priors land on their target peaks:
  `python scripts/envs/bandit2d/viz_priors.py` should report
  dist→target ≈ 0.15 for every client.
* Run sweeps: `bash scripts/envs/bandit2d/run_all.sh`.
* Inspect: `bash scripts/envs/bandit2d/analyze.sh` and view
  `plots/bandit2d_phase1/ring_comparison.png` — FedGuide-prior / -pg
  should show 4 distinct peaks, FedAvg / -KL / -all should show a
  single collapsed peak.
