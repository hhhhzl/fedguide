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
        device: Optional[str] = None,

        use_sampling_guidance: bool = False,
        guidance_eta: float = 0.1,
        guide_align_coef: float = 0.0,
        entropy_coef: float = 0.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.prior_coef = prior_coef
        self.guide_coef = guide_coef
        self.max_grad_norm = max_grad_norm

        self.online_guidance = online_guidance
        self.online_guidance_steps = online_guidance_steps
        self.online_prior = online_prior
        self.prior_reg_coef = prior_reg_coef

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # -------- Policy / Value ----------
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_fn = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1),
        )

        # -------- Prior ----------
        self.prior = prior
        if self.prior is None and prior_ctor is not None:
            self.prior = prior_ctor(**(prior_ctor_kwargs or {}))
        if self.prior is not None:
            _to_device(self.prior, self.device)
            self._init_prior_adapt_params()
            self._maybe_load_prior(prior_ckpt)
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
            if "policy" in sd:
                self.policy.load_state_dict(sd["policy"], strict=False)
            elif isinstance(sd, dict):
                try:
                    self.policy.load_state_dict(sd, strict=False)
                except Exception:
                    pass

            if "value" in sd:
                try:
                    self.value_fn.load_state_dict(sd["value"], strict=False)
                except Exception:
                    pass

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
        self.log_std.data.clamp_(-5.0, 2.0)

        # switch
        self.use_sampling_guidance = bool(use_sampling_guidance)
        self.guidance_eta = float(guidance_eta)
        self.guide_align_coef = float(guide_align_coef)
        self.entropy_coef = float(entropy_coef)

    # ========= Prior & Guidance pretrain load =========
    def _maybe_load_prior(self, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        if not os.path.isfile(ckpt_path):
            print(f"[FedguideAgent] prior ckpt not found: {ckpt_path}")
            return
        sd = torch.load(ckpt_path, map_location="cpu")
        try:
            if isinstance(sd, dict) and "prior" in sd and isinstance(sd["prior"], dict):
                self.prior.load_state_dict(sd["prior"], strict=False)
            else:
                self.prior.load_state_dict(sd, strict=False)
            print(f"[FedguideAgent] loaded pretrained prior from: {ckpt_path}")
        except Exception as e:
            print(f"[FedguideAgent] load prior failed ({ckpt_path}): {e}")

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
        adapt, frozen = [], []
        for n, p in named:
            if any(tag in n.lower() for tag in ("lora", "adapter", "head")):
                adapt.append(p)
            else:
                frozen.append(p)
        for p in frozen:
            p.requires_grad = False
        for p in adapt:
            p.requires_grad = True
        self.prior_adapt_params = adapt

    # ========= Distribution / Evaluate =========
    def _dist(self, state: torch.Tensor):
        mu = self.policy(state)
        std = self.log_std.exp().clamp(min=1e-6)
        return torch.distributions.Normal(mu, std), mu

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, deterministic: bool = False):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        dist, mu = self._dist(state)
        action = mu if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        value = self.value_fn(state).squeeze(-1)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

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
                if any(tag in k.lower() for tag in ("lora", "adapter", "head"))
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
            sd = self.prior.state_dict()
            for k, v in parameters["prior_adapt"].items():
                if k in sd:
                    sd[k] = v.to(self.device)
            self.prior.load_state_dict(sd, strict=False)
        if self.guidance is not None and "guidance" in parameters:
            self.guidance.load_state_dict({k: v.to(self.device) for k, v in parameters["guidance"].items()}, strict=False)

    def parameters_iter(self) -> Iterable[torch.nn.Parameter]:
        return list(self.policy.parameters()) + [self.log_std] + list(self.value_fn.parameters())

    @torch.no_grad()
    def select_action(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1: state = state.unsqueeze(0)
        dist, mu = self.dist(state)
        action = mu if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.value_fn(state).squeeze(-1)

        # a -> a + ∇_a W_t
        if self.use_sampling_guidance and (self.guidance is not None) and hasattr(self.guidance, "calculate_guidance"):
            t = torch.rand(action.shape[0], device=self.device)
            g = self.guidance.calculate_guidance(action.clone(), t,
                                                 condition=state)  # ∇_a W_t(s,a)  :contentReference[oaicite:5]{index=5}
            action = action + self.guidance_eta * g
            logp = dist.log_prob(action).sum(-1)
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
                # If lambda_guide != 0, use KL(π || prior) with weight lambda_guide (old-style λ_guide).
                # Else fall back to MLE-style prior loss with self.prior_coef for backward compatibility.
                prior_loss = torch.tensor(0.0, device=self.device)
                prior_report = torch.tensor(0.0, device=self.device)  # for logging
                if (self.prior is not None) and hasattr(self.prior, "log_prob"):
                    try:
                        prior_logp = self.prior.log_prob(mb_a, mb_s)  # [B]
                        if abs(lambda_guide) > 0.0:
                            # KL(π || prior) ≈ E[log prior - log π]
                            prior_kl = (prior_logp - logp).mean()
                            prior_loss = lambda_guide * prior_kl
                            prior_report = prior_kl.detach()
                        else:
                            # legacy: maximize log prior(a|s)  => minimize -log prior
                            prior_mle = -prior_logp.mean()
                            prior_loss = self.prior_coef * prior_mle
                            prior_report = prior_mle.detach()
                    except Exception:
                        pass

                # —— Guidance ——
                # -------- guidance align (train-time distillation; no change to sampling flow) --------
                guide_align = torch.tensor(0.0, device=self.device)
                if (self.guidance is not None) and hasattr(self.guidance, "calculate_guidance"):
                    with torch.no_grad():
                        t = torch.rand(mb_a.shape[0], device=self.device)
                        gvec = self.guidance.calculate_guidance(mb_a.clone(), t, condition=mb_s)
                        target_a = mb_a + self.guidance_eta * gvec  # a = a + η·∇_a W_t
                    guide_align = F.mse_loss(mu, target_a)

                # prefer new guide_align_coef if set (>0), else fall back to old self.guide_coef.
                guide_weight = self.guide_align_coef if (getattr(self, "guide_align_coef", 0.0) > 0.0) else self.guide_coef

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                    + prior_loss
                    + self.guide_coef * guide_align
                    + lambda_local * (mb_old_logp - logp).mean()
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

        if hasattr(self.guidance, "update_wt"):
            gd = {"states": s.detach(), "actions": a.detach(), "weights": torch.relu(adv[mask]).detach()}
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