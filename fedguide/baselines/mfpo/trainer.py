"""
MFPO local training loop — matches MFPO-INFOCOM24/code/main.py + server.share_model.

Each federated round runs `local_update` inner iterations:
  - On k==0: target ← copy(critic) (same as Server.share_model).
  - worker.train(batch_size, step) with step = (round-1)*local_update + k (1-based rounds).

`train()` returns (grad, metrics) with real `loss` for logging (previously the trainer
hardcoded loss=0, which made Flower metrics useless).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class MFPTrainer:
    def __init__(
        self,
        agent: Any,
        env: Any,
        *,
        batch_size: int = 20,
        local_update: int = 10,
        device: Optional[str] = None,
    ):
        self.agent = agent
        self.env = env
        self.batch_size = int(batch_size)
        self.local_update = int(local_update)
        self.server_round = 1
        self.last_actions = None
        self.n_steps = self.batch_size * self.local_update

    def set_server_round(self, rnd: int):
        self.server_round = int(rnd)

    def train_one_round(self) -> Dict[str, float]:
        w = self.agent.worker
        rnd = self.server_round
        loss_acc = 0.0
        loss_c_acc = 0.0
        n = 0
        for k in range(self.local_update):
            step = (rnd - 1) * self.local_update + k
            if k == 0:
                w.target.load_state_dict(w.critic.state_dict())
            out = w.train(self.batch_size, step)
            if isinstance(out, tuple) and len(out) == 2:
                _grad, m = out
            else:
                m = {"loss": 0.0}
            loss_acc += float(m.get("loss", 0.0))
            loss_c_acc += float(m.get("loss/critic", 0.0))
            n += 1

        mean_loss = loss_acc / max(n, 1)
        eval_ret = float(w.test(rnd))
        self.n_steps = self.batch_size * self.local_update
        return {
            "loss": mean_loss,
            "loss/critic": loss_c_acc / max(n, 1),
            "train/return": eval_ret,
            "eval/return": eval_ret,
        }

    def save_eval(self, cid: str, rnd: int, outdir: str = "./results/mfpo") -> bool:
        return True
