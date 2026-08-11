import os
from typing import Dict, Optional, Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_device(module_or_tensor, device):
    if hasattr(module_or_tensor, "to"):
        return module_or_tensor.to(device)
    return module_or_tensor


class FedguideAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        prior: Optional[nn.Module] = None,
        guidance: Optional[nn.Module] = None,
        prior_ctor: Optional[Any] = None,
        guidance_ctor: Optional[Any] = None,
        prior_ctor_kwargs: Optional[Dict] = None,
        guidance_ctor_kwargs: Optional[Dict] = None,

        # pretrain path
        prior_ckpt: Optional[str] = None,
        guidance_ckpt: Optional[str] = None,
        actor_ckpt: Optional[str] = None,

        # PPO
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        prior_coef: float = 1.0,
        guide_coef: float = 1.0,
        max_grad_norm: float = 0.5,

        online_guidance: bool = False,
        online_guidance_steps: int = 1,
        online_prior: bool = False,
        prior_lr: float = 1e-4,
        prior_reg_coef: float = 1e-3,
        prior_adapt_fallback_all: bool = False,
        device: Optional[str] = None,

        use_sampling_guidance: bool = False,
        guidance_eta: float = 0.1,
        guide_align_coef: float = 0.0,
        entropy_coef: float = 0.0,
        init_log_std: float = 0.0,
        # Opt-in policy architecture knobs (defaults preserve legacy behaviour).
        policy_activation: str = "tanh",
        action_clamp_low: Optional[float] = None,
        action_clamp_high: Optional[float] = None,
        log_std_anneal: bool = False,
        log_std_anneal_target: float = -2.0,
        log_std_anneal_rounds: int = 40,
        # Route 3: DICE reshapes the diffusion prior instead of pushing
        # policy μ via Q-gradient. When enabled and a guidance critic with
        # a Q head is loaded, the prior log-prob used by the IW prior loss
        # becomes log p̃(a|s) = log p(a|s) + reshape_beta · Q(s,a). The
        # PPO guide_align MSE term is forced to zero so DICE never touches
        # the policy gradient directly.
        prior_reshape: bool = False,
        reshape_beta: float = 0.1,
        bc_blend_alpha: float = 1.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.init_log_std = float(init_log_std)
        self.policy_activation = str(policy_activation).lower()
        self.action_clamp_low = action_clamp_low
        self.action_clamp_high = action_clamp_high
        self.log_std_anneal = bool(log_std_anneal)
        self.log_std_anneal_target = float(log_std_anneal_target)
        self.log_std_anneal_rounds = int(log_std_anneal_rounds)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.prior_coef = prior_coef
        self.guide_coef = guide_coef
        self.max_grad_norm = max_grad_norm

        self.prior_reshape = bool(prior_reshape)
        self.reshape_beta = float(reshape_beta)

        self.online_guidance = online_guidance
        self.online_guidance_steps = online_guidance_steps
        self.online_prior = online_prior
        self.prior_lr = prior_lr
        self.prior_reg_coef = prior_reg_coef
        self.prior_adapt_fallback_all = bool(prior_adapt_fallback_all)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # -------- Policy / Value ----------
        if self.policy_activation == "relu":
            _act = nn.ReLU
        else:
            _act = nn.Tanh  # default, backward compatible
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256), _act(),
            nn.Linear(256, 256), _act(),
            nn.Linear(256, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        self.value_fn = nn.Sequential(
            nn.Linear(state_dim, 256), _act(),
            nn.Linear(256, 256), _act(),
            nn.Linear(256, 1),
        )

        # -------- Prior ----------
        self.prior = prior
        # The local prior remains the object uploaded to the server.  For
        # Bandit2D, the server may additionally broadcast a density-space
        # mixture used by the policy loss without overwriting that local prior.
        self.routing_prior = None
        if self.prior is None and prior_ctor is not None:
            self.prior = prior_ctor(**(prior_ctor_kwargs or {}))
        if self.prior is not None:
            _to_device(self.prior, self.device)
            self._init_prior_adapt_params()
            self._maybe_load_prior(prior_ckpt)
            # Warm-start the policy mean from the prior's mode where possible.
            # For Gaussian priors (bandit2d) this puts policy μ ≈ prior μ at
            # round 0, so the very first rollout already samples near the
            # client's target mode and IS-CE has signal to maintain it.
            self._warm_start_policy_from_prior()
        else:
            self.prior_adapt_params = []
            self._prior_shadow = None

        # -------- Guidance ----------
        self.guidance = guidance
        if self.guidance is None and guidance_ctor is not None:
            self.guidance = guidance_ctor(**(guidance_ctor_kwargs or {}))
        if self.guidance is not None:
            _to_device(self.guidance, self.device)
            self._maybe_load_guidance(guidance_ckpt)

        # -------- Actor ----------
        if actor_ckpt is not None and os.path.isfile(actor_ckpt):
            sd = torch.load(actor_ckpt, map_location="cpu")
            a = max(0.0, min(1.0, float(bc_blend_alpha)))

            def _blend_state_dict(module, src_sd):
                # Blend module params: w ← a * w_BC + (1-a) * w_init.
                # Only blend keys present in both; preserve dtype/device.
                dst_sd = module.state_dict()
                for k, src in src_sd.items():
                    if k not in dst_sd:
                        continue
                    dst = dst_sd[k]
                    if dst.shape != src.shape:
                        continue
                    src_t = src.to(dst.device, dtype=dst.dtype)
                    dst.mul_(1.0 - a).add_(src_t, alpha=a)
                module.load_state_dict(dst_sd, strict=False)

            if "policy" in sd:
                _blend_state_dict(self.policy, sd["policy"])
            elif isinstance(sd, dict):
                try:
                    _blend_state_dict(self.policy, sd)
                except Exception:
                    pass

            if "value" in sd:
                try:
                    _blend_state_dict(self.value_fn, sd["value"])
                except Exception:
                    pass
            if "log_std" in sd and sd["log_std"] is not None:
                try:
                    ls = sd["log_std"]
                    if isinstance(ls, torch.Tensor):
                        ls_t = ls.to(self.log_std.device, dtype=self.log_std.dtype)
                        self.log_std.data.mul_(1.0 - a).add_(ls_t, alpha=a)
                except Exception:
                    pass
            print(f"[FedguideAgent] BC warm-start ← {actor_ckpt} (blend α={a:.2f})")

        # -------- Optimizers ----------
        self.lr = lr
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + [self.log_std] + list(self.value_fn.parameters()),
            lr=self.lr
        )

        # prior
        self.prior_opt = None
        if self.prior is not None and len(self.prior_adapt_params) > 0:
            self.prior_opt = torch.optim.Adam(self.prior_adapt_params, lr=prior_lr)
            self._prior_shadow = [p.detach().clone().to(self.device) for p in self.prior_adapt_params]

        _to_device(self.policy, self.device)
        _to_device(self.value_fn, self.device)
        # Parameter.to() returns a new tensor — move log_std in-place to keep
        # it on the same device as the policy/value nets.
        self.log_std.data = self.log_std.data.to(self.device)
        self.log_std.data.clamp_(-5.0, 2.0)

        # switch
        self.use_sampling_guidance = bool(use_sampling_guidance)
        self.guidance_eta = float(guidance_eta)
        self.guide_align_coef = float(guide_align_coef)
        self.entropy_coef = float(entropy_coef)

    # ========= Warm-start =========
    def _warm_start_policy_from_prior(self) -> None:
        """If the prior exposes a `head_mu` (e.g. GaussianBehaviorPrior on
        bandit2d), zero the policy's final Linear weight and set its bias to
        `head_mu`. The policy then maps every state to ≈ head_mu at round 0.

        For state-conditional priors (DiffusionGuidance UNet on reacher) we
        cannot extract a single µ, so this is a no-op — equivalent warm-start
        for those envs would be a separate BC pretrain pass.
        """
        try:
            head_mu = getattr(self.prior, "head_mu", None)
            if head_mu is None or not isinstance(head_mu, torch.nn.Parameter):
                return
            if head_mu.shape[-1] != self.action_dim:
                return
            last_lin = None
            for m in reversed(list(self.policy.modules())):
                if isinstance(m, nn.Linear) and m.out_features == self.action_dim:
                    last_lin = m
                    break
            if last_lin is None:
                return
            mu = head_mu.detach().to(last_lin.bias.device, dtype=last_lin.bias.dtype)
            with torch.no_grad():
                last_lin.weight.zero_()
                last_lin.bias.copy_(mu)
            print(f"[FedguideAgent] warm-started policy μ ← prior.head_mu = "
                  f"{mu.cpu().tolist()}")
        except Exception as e:
            print(f"[FedguideAgent] warm-start skipped: {e}")

    # ========= Prior & Guidance pretrain load =========
    def _maybe_load_prior(self, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        if not os.path.isfile(ckpt_path):
            print(f"[FedguideAgent] prior ckpt not found: {ckpt_path}")
            return
        sd = torch.load(ckpt_path, map_location="cpu")
        try:
            # Decide which prior class produced this checkpoint.
            # Priorities: explicit `prior_type` field → key-shape heuristics.
            is_simple_prior = False
            is_diffusion_unet = False
            if isinstance(sd, dict):
                pt = sd.get("prior_type")
                if pt in ("diffusion", "diffusion_unet", "diffusion_guidance"):
                    is_diffusion_unet = True
                elif pt == "gaussian":
                    is_simple_prior = False  # caller already constructed Gaussian
                else:
                    inner = sd.get("prior") if "prior" in sd and isinstance(sd["prior"], dict) else sd
                    if isinstance(inner, dict) and any(k.startswith("model.") for k in inner.keys()):
                        is_diffusion_unet = True
                    elif "prior" in sd or ("state_dim" in sd and "unet" not in sd):
                        is_simple_prior = True
                    elif not ("unet" in sd or "scheduler_config" in sd):
                        if any("encoder" in k or "decoder" in k for k in sd.keys()):
                            is_simple_prior = True
            
            # Handle different checkpoint formats
            if isinstance(sd, dict):
                if is_diffusion_unet:
                    # DiffusionGuidance saves UNet under prior["model.*"].
                    inner = sd["prior"] if "prior" in sd and isinstance(sd["prior"], dict) else sd
                    self.prior.load_state_dict(inner, strict=False)
                    print(f"[FedguideAgent] loaded pretrained DiffusionGuidance(UNet) from: {ckpt_path}")
                elif is_simple_prior:
                    # SimpleDiffusionPrior format
                    if "prior" in sd:
                        self.prior.load_state_dict(sd["prior"], strict=False)
                        print(f"[FedguideAgent] loaded pretrained SimpleDiffusionPrior from: {ckpt_path}")
                    elif "state_dim" in sd:
                        # Try loading as full state dict (may have extra metadata)
                        try:
                            self.prior.load_state_dict(sd, strict=False)
                        except Exception:
                            # If that fails, try extracting just the prior state dict
                            # by filtering out metadata keys
                            prior_sd = {k: v for k, v in sd.items() 
                                      if k not in ["state_dim", "action_dim", "hidden_dim", "timesteps"]}
                            self.prior.load_state_dict(prior_sd, strict=False)
                        print(f"[FedguideAgent] loaded pretrained SimpleDiffusionPrior from: {ckpt_path}")
                    else:
                        # Direct state dict
                        self.prior.load_state_dict(sd, strict=False)
                        print(f"[FedguideAgent] loaded pretrained SimpleDiffusionPrior from: {ckpt_path}")
                else:
                    # DiffusionGuidance format
                    # Check if it's a pretrain checkpoint with "unet" key
                    if "unet" in sd:
                        # Load UNet weights into prior.model
                        self.prior.model.load_state_dict(sd["unet"], strict=False)
                        print(f"[FedguideAgent] loaded pretrained UNet from: {ckpt_path}")
                    # Check if it's a nested format with "prior" key
                    elif "prior" in sd and isinstance(sd["prior"], dict):
                        self.prior.load_state_dict(sd["prior"], strict=False)
                        print(f"[FedguideAgent] loaded pretrained prior from: {ckpt_path}")
                    # Otherwise try to load directly as prior state_dict
                    else:
                        self.prior.load_state_dict(sd, strict=False)
                        print(f"[FedguideAgent] loaded pretrained prior from: {ckpt_path}")
            else:
                # Direct state dict - try to determine type by checking if prior has model attribute
                if hasattr(self.prior, "model"):
                    # DiffusionGuidance
                    self.prior.model.load_state_dict(sd, strict=False)
                    print(f"[FedguideAgent] loaded pretrained UNet from: {ckpt_path}")
                else:
                    # SimpleDiffusionPrior
                    self.prior.load_state_dict(sd, strict=False)
                    print(f"[FedguideAgent] loaded pretrained SimpleDiffusionPrior from: {ckpt_path}")
        except Exception as e:
            print(f"[FedguideAgent] load prior failed ({ckpt_path}): {e}")
            import traceback
            traceback.print_exc()

    def _maybe_load_guidance(self, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        if not os.path.isfile(ckpt_path):
            print(f"[FedguideAgent] guidance ckpt not found: {ckpt_path}")
            return
        sd = torch.load(ckpt_path, map_location="cpu")
        try:
            if isinstance(sd, dict) and "guidance" in sd and isinstance(sd["guidance"], dict):
                self.guidance.load_state_dict(sd["guidance"], strict=False)
            else:
                self.guidance.load_state_dict(sd, strict=False)
            print(f"[FedguideAgent] loaded pretrained guidance from: {ckpt_path}")
        except Exception as e:
            print(f"[FedguideAgent] load guidance failed ({ckpt_path}): {e}")

    def _init_prior_adapt_params(self):
        named = list(self.prior.named_parameters())
        adapt, adapt_names, frozen = [], [], []
        for n, p in named:
            if any(tag in n.lower() for tag in ("lora", "adapter", "head")):
                adapt.append(p)
                adapt_names.append(n)
            else:
                frozen.append(p)
        # Experimental fallback for priors without adapter/lora/head naming.
        # Keeps default behavior unless explicitly enabled by config.
        if len(adapt) == 0 and self.prior_adapt_fallback_all:
            adapt = [p for _, p in named]
            adapt_names = [n for n, _ in named]
            frozen = []
        for p in frozen:
            p.requires_grad = False
        for p in adapt:
            p.requires_grad = True
        self.prior_adapt_params = adapt
        self.prior_adapt_names = set(adapt_names)

    # ========= Distribution / Evaluate =========
    def _dist(self, state: torch.Tensor):
        mu = self.policy(state)
        if self.action_clamp_low is not None and self.action_clamp_high is not None:
            mu = torch.clamp(mu, float(self.action_clamp_low), float(self.action_clamp_high))
        std = self.log_std.exp().clamp(min=1e-6)
        return torch.distributions.Normal(mu, std), mu

    def anneal_log_std(self, server_round: int, target: float = -2.0, decay_rounds: int = 40):
        """Linearly decay log_std from init_log_std to target over decay_rounds."""
        if decay_rounds <= 0:
            return
        progress = min(1.0, server_round / max(1, decay_rounds))
        new_val = self.init_log_std + (target - self.init_log_std) * progress
        new_val = max(-5.0, min(2.0, new_val))
        with torch.no_grad():
            self.log_std.data.fill_(new_val)

    def evaluate(self, state, action):
        state = state.to(self.device).float()
        action = action.to(self.device).float()
        dist, mu = self._dist(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_fn(state).squeeze(-1)
        return log_prob, entropy, value, mu

    # ========= aggregate hook for fed =========
    def get_parameters(self) -> Dict[str, Any]:
        out = {
            "policy": {k: v.detach().cpu() for k, v in self.policy.state_dict().items()},
            "log_std": self.log_std.detach().cpu(),
            "value": {k: v.detach().cpu() for k, v in self.value_fn.state_dict().items()},
        }
        if self.prior is not None and len(self.prior_adapt_params) > 0:
            out["prior_adapt"] = {
                k: v.detach().cpu()
                for k, v in self.prior.state_dict().items()
                if k in self.prior_adapt_names
            }
        if self.guidance is not None:
            out["guidance"] = {k: v.detach().cpu() for k, v in self.guidance.state_dict().items()}
        return out

    def set_parameters(self, parameters: Dict[str, Any]):
        if "policy" in parameters:
            self.policy.load_state_dict({k: v.to(self.device) for k, v in parameters["policy"].items()}, strict=False)
        if "log_std" in parameters:
            self.log_std.data = parameters["log_std"].to(self.device).clone()
        if "value" in parameters:
            self.value_fn.load_state_dict({k: v.to(self.device) for k, v in parameters["value"].items()}, strict=False)
        if self.prior is not None and "prior_adapt" in parameters:
            self.routing_prior = None
            sd = self.prior.state_dict()
            for k, v in parameters["prior_adapt"].items():
                if k in sd:
                    sd[k] = v.to(self.device)
            self.prior.load_state_dict(sd, strict=False)
        if "prior_mixture" in parameters:
            mixture = parameters["prior_mixture"]
            if isinstance(mixture, (list, tuple)) and len(mixture) == 3:
                from fedguide.guidance.diffusion_prior import GaussianMixtureBehaviorPrior

                self.routing_prior = GaussianMixtureBehaviorPrior(
                    mixture[0], mixture[1], mixture[2]
                ).to(self.device)
        if self.guidance is not None and "guidance" in parameters:
            self.guidance.load_state_dict({k: v.to(self.device) for k, v in parameters["guidance"].items()}, strict=False)

    def parameters_iter(self) -> Iterable[torch.nn.Parameter]:
        return list(self.policy.parameters()) + [self.log_std] + list(self.value_fn.parameters())

    @torch.no_grad()
    def select_action(self, state, deterministic=False):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        dist, mu = self._dist(state)
        action = mu if deterministic else dist.sample()
        if self.action_clamp_low is not None and self.action_clamp_high is not None:
            action = torch.clamp(
                action, float(self.action_clamp_low), float(self.action_clamp_high)
            )
        logp = dist.log_prob(action).sum(dim=-1)
        value = self.value_fn(state).squeeze(-1)

        # a -> a + ∇_a W_t
        if self.use_sampling_guidance and (self.guidance is not None) and hasattr(self.guidance, "calculate_guidance"):
            t = torch.rand(action.shape[0], device=self.device)
            g = self.guidance.calculate_guidance(action.clone(), t,
                                                 condition=state)  # ∇_a W_t(s,a)
            action = action + self.guidance_eta * g
            logp = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    # ========= update =========
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        epochs: int = 1,
        minibatch_size: Optional[int] = None,
        lambda_local: float = 0.0,
        lambda_guide: float = 0.0,
    ) -> Dict[str, float]:
        s = batch["s"].to(self.device).float()
        a = batch["a"].to(self.device).float()
        old_logp = batch["old_logp"].to(self.device).float()
        ret = batch["ret"].to(self.device).float()
        adv = batch["adv"].to(self.device).float()

        # normalize advantages (PPO standard)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        N = s.size(0)
        if not minibatch_size or minibatch_size <= 0:
            minibatch_size = N

        last_policy_loss = last_value_loss = last_prior_loss = last_guide_loss = 0.0
        last_entropy = 0.0
        clip_frac = 0.0
        approx_kl = 0.0

        for _ in range(epochs):
            perm = torch.randperm(N, device=self.device)
            for st in range(0, N, minibatch_size):
                idx = perm[st: st + minibatch_size]
                mb_s, mb_a, mb_old_logp, mb_ret, mb_adv = s[idx], a[idx], old_logp[idx], ret[idx], adv[idx]

                # policy/value/entropy under current policy
                logp, entropy, value, mu = self.evaluate(mb_s, mb_a)

                # PPO clip surrogate
                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, mb_ret)
                entropy_loss = -entropy.mean()

                # —— Prior ——
                # Self-normalized IS estimate of cross-entropy E_{prior}[-log π_θ(a|s)]:
                # take samples a ~ π_old, weight each by ω(a) ∝ prior(a|s)/π_old(a|s),
                # and minimize −Σ ω · log π_θ. SNIS is biased but consistent and
                # stable; importantly it gives a strong gradient signal at actions
                # the prior favors even when π_old has not yet covered them, which
                # the previous bonus-PG could not (its REINFORCE form was off-policy
                # biased once update_epochs > 1, and the centered bonus collapsed
                # toward zero as the policy approached the prior).
                prior_loss = torch.tensor(0.0, device=self.device)
                prior_report = torch.tensor(0.0, device=self.device)  # for logging
                active_prior = self.routing_prior if self.routing_prior is not None else self.prior
                if (active_prior is not None) and hasattr(active_prior, "log_prob"):
                    try:
                        prior_logp = active_prior.log_prob(mb_a, mb_s)  # [B]
                        # Route 3: reshape the prior with DICE Q so the IW
                        # weights reflect log p̃(a|s) = log p(a|s) + β·Q(s,a).
                        # Q is standardized within the minibatch to keep the
                        # reshape additive on the same scale as log-prob diffs.
                        if (
                            self.prior_reshape
                            and self.guidance is not None
                            and hasattr(self.guidance, "q0")
                        ):
                            with torch.no_grad():
                                q_raw = self.guidance.q0(mb_a, mb_s).detach().squeeze(-1)
                                q_norm = (q_raw - q_raw.mean()) / (q_raw.std() + 1e-6)
                            prior_logp = prior_logp + self.reshape_beta * q_norm
                        if abs(lambda_guide) > 0.0:
                            with torch.no_grad():
                                log_iw = (prior_logp - mb_old_logp).detach()
                                # Stabilize: max-shift, exp, then self-normalize.
                                log_iw = log_iw - log_iw.max()
                                iw = log_iw.exp()
                                iw = iw / (iw.sum() + 1e-12)
                            # Pull π_θ toward prior by maximizing the IW-weighted
                            # log-prob of rollout samples: gradient is
                            # −λ · Σ ω · ∇ log π_θ(a|s), which on samples favored
                            # by the prior pushes log π up.
                            prior_loss = - lambda_guide * (iw * logp).sum()
                            prior_report = (prior_logp - logp).mean().detach()
                        else:
                            # legacy: maximize log prior(a|s)  => minimize -log prior
                            prior_mle = -prior_logp.mean()
                            prior_loss = self.prior_coef * prior_mle
                            prior_report = prior_mle.detach()
                    except Exception:
                        pass

                # —— Guidance ——
                # -------- guidance align (train-time distillation; no change to sampling flow) --------
                # Route 3: when prior_reshape is on, DICE has already entered
                # the loss through prior_logp; never touch the policy gradient
                # directly via Q-gradient MSE.
                guide_align = torch.tensor(0.0, device=self.device)
                if (
                    not self.prior_reshape
                    and self.guidance is not None
                    and hasattr(self.guidance, "calculate_guidance")
                ):
                    with torch.no_grad():
                        t = torch.rand(mb_a.shape[0], device=self.device)
                        gvec = self.guidance.calculate_guidance(mb_a.clone(), t, condition=mb_s)
                        target_a = mb_a + self.guidance_eta * gvec  # a = a + η·∇_a W_t
                    guide_align = F.mse_loss(mu, target_a)

                # prefer new guide_align_coef if set (>0), else fall back to old self.guide_coef.
                guide_weight = self.guide_align_coef if (getattr(self, "guide_align_coef", 0.0) > 0.0) else self.guide_coef

                # Local trust region: penalize KL(π_old || π).
                # E_{a~π_old}[log π_old - log π] ≈ KL(π_old || π) ≥ 0, so we ADD λ_local * (old - new).
                local_kl = (mb_old_logp - logp).mean()
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                    + prior_loss
                    + self.guide_coef * guide_align
                    + lambda_local * local_kl
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters_iter(), self.max_grad_norm)
                self.optimizer.step()

                last_policy_loss = float(policy_loss.detach().cpu())
                last_value_loss = float(value_loss.detach().cpu())
                last_prior_loss = float(prior_report.cpu())  # unweighted report (KL or MLE), like before
                last_guide_loss = float(guide_align.detach().cpu())
                last_entropy = float(entropy.mean().detach().cpu())
                approx_kl = float((mb_old_logp - logp).mean().abs().detach().cpu())
                clip_frac = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().detach().cpu())

        return {
            "loss/total": last_policy_loss
                          + self.vf_coef * last_value_loss
                          + (self.prior_coef if abs(lambda_guide) == 0.0 else 0.0) * last_prior_loss
                          + guide_weight * last_guide_loss,
            "loss/policy": last_policy_loss,
            "loss/value": last_value_loss,
            "loss/prior": last_prior_loss,  # reports KL(π||prior) or MLE prior (unweighted)
            "loss/guide": last_guide_loss,  # unweighted MSE align
            "entropy": last_entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }

    @torch.no_grad()
    def _select_high_adv_mask(self, adv: torch.Tensor, top_p: float = 0.5):
        q = torch.quantile(adv, 1 - top_p)
        return adv >= q

    def online_guidance_step(self, batch: Dict[str, torch.Tensor]):
        if not self.online_guidance or self.guidance is None:
            return
        s = batch["s"].to(self.device).float()
        a = batch["a"].to(self.device).float()
        adv = batch["adv"].to(self.device).float()

        mask = self._select_high_adv_mask(torch.relu(adv), top_p=0.5)
        if mask.sum() == 0:
            return
        s, a = s[mask], a[mask]

        # Check if guidance is SimpleDiffusionPrior (has update method with states, actions, lr signature)
        from fedguide.guidance.diffusion_prior import SimpleDiffusionPrior
        if isinstance(self.guidance, SimpleDiffusionPrior):
            # SimpleDiffusionPrior uses update(states, actions, lr) method
            try:
                # Use prior_lr for online training (same as online_prior_step)
                guidance_lr = getattr(self, "prior_lr", 1e-4)
                self.guidance.update(s.detach(), a.detach(), lr=guidance_lr)
            except Exception as e:
                print(f"[FedguideAgent] SimpleDiffusionPrior.update() failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            # SDICE_Critic or other guidance types with update_wt/update_v0 methods
            if hasattr(self.guidance, "update_wt"):
                # gd = {"states": s.detach(), "actions": a.detach(), "weights": torch.relu(adv[mask]).detach()}
                gd = {"s": s.detach(), "a": a.detach(), "weights": torch.relu(adv[mask]).detach()}
                try:
                    self.guidance.update_wt(gd)
                except TypeError:
                    pass

            if "r" in batch and "s_next" in batch and "done" in batch and hasattr(self.guidance, "update_v0"):
                gd2 = {
                    "states": s.detach(),
                    "actions": a.detach(),
                    "rewards": batch["r"][mask].to(self.device).detach(),
                    "next_states": batch["s_next"][mask].to(self.device).detach(),
                    "dones": batch["done"][mask].float().to(self.device).detach(),
                }
                try:
                    self.guidance.update_v0(gd2)
                except TypeError:
                    pass

    def online_prior_step(self, batch: Dict[str, torch.Tensor]):
        if not self.online_prior or self.prior is None or self.prior_opt is None:
            return
        s = batch["s"].to(self.device).float().detach()
        a = batch["a"].to(self.device).float().detach()
        adv = batch["adv"].to(self.device).float().detach()

        mask = self._select_high_adv_mask(torch.relu(adv), top_p=0.5)
        if mask.sum() == 0:
            return
        s, a, w = s[mask], a[mask], torch.relu(adv[mask])

        self.prior_opt.zero_grad(set_to_none=True)
        logp = self.prior.log_prob(a, s)  # [B]
        loss_fit = (-(w * logp)).mean()

        loss_reg = torch.tensor(0.0, device=self.device)
        if self._prior_shadow is not None and len(self._prior_shadow) == len(self.prior_adapt_params):
            for p, p0 in zip(self.prior_adapt_params, self._prior_shadow):
                loss_reg = loss_reg + (p - p0).pow(2).mean()

        loss = loss_fit + self.prior_reg_coef * loss_reg
        loss.backward()
        nn.utils.clip_grad_norm_(self.prior_adapt_params, 1.0)
        self.prior_opt.step()
