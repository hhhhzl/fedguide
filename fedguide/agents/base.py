import torch
import torch.nn as nn


class BaseAgent:
    def __init__(
            self, policy: nn.Module,
            lr: float = 1e-3
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def rebuild_optimizer(self):
        """Recreate optimizer to rebind new parameters."""
        return NotImplementedError

    def act(self, state, deterministic: bool = False):
        dist = self.policy(torch.as_tensor(state, dtype=torch.float32))
        if hasattr(dist, "mean") and deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        logp = dist.log_prob(action)
        if logp.ndim > 0:
            logp = logp.sum(-1)
        a_np = action.detach().cpu().numpy()
        try:
            a_np = a_np.item()
        except Exception:
            pass
        return a_np, logp

    def get_parameters(self):
        """Return parameters as a list of numpy arrays (policy + value_fn)."""
        # sd = {}
        # sd.update({f"policy.{k}": v.detach().cpu() for k, v in self.policy.state_dict().items()})
        # return [t.numpy() for t in sd.values()]
        return NotImplementedError

    def set_parameters(self, parameters):
        # keys = [f"policy.{k}" for k in self.policy.state_dict().keys()]
        # assert len(parameters) == len(keys), f"param length mismatch: {len(parameters)} vs {len(keys)}"
        # pi_sd = self.policy.state_dict()
        # cursor = 0
        # for k in list(pi_sd.keys()):
        #     arr = parameters[cursor]
        #     pi_sd[k] = torch.tensor(arr, dtype=pi_sd[k].dtype)
        #     cursor += 1
        # self.policy.load_state_dict(pi_sd, strict=True)
        return NotImplementedError

    def to(self, device):
        if hasattr(self.policy, "to"):
            self.policy.to(device)
        return self

    def parameters(self):
        for p in self.policy.parameters():
            yield p

    def update(self, loss_or_batch):
        if not torch.is_tensor(loss_or_batch):
            raise TypeError("This BaseAgent.update expects a loss tensor; your trainer should call compute_loss or handle batch.")
        self.optimizer.zero_grad()
        loss_or_batch.backward()
        self.optimizer.step()
        return loss_or_batch.detach().cpu().item()

    def evaluate(self, state, action):
        return NotImplementedError
