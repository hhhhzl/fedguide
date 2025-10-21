import flwr as fl
import torch
import numpy as np


class FedGuideClient(fl.client.NumPyClient):
    """Flower client wrapper for FedGuide"""

    def __init__(self, agent, env, trainer, aggregate_mode: str = "policy"):
        self.agent = agent
        self.env = env
        self.trainer = trainer
        self.cid = getattr(self, "cid", "unknown")

        assert aggregate_mode in {"policy", "prior"}, "aggregate_mode must be 'policy' or 'prior'"
        self.aggregate_mode = aggregate_mode

        # if self.aggregate_mode == "policy":
        #     self._tx_params = list(self.agent.policy.parameters())
        #     if len(self._tx_params) == 0:
        #         raise RuntimeError("Policy has no parameters to aggregate.")
        # else:  # prior
        #     if not hasattr(self.trainer, "prior") or not hasattr(self.trainer.prior, "parameters"):
        #         raise RuntimeError("Trainer.prior is missing or not an nn.Module.")
        #     self._tx_params = [p for p in self.trainer.prior.parameters() if p.requires_grad]
        #     if len(self._tx_params) == 0:
        #         raise RuntimeError("Prior has no trainable parameters to aggregate.")

    # ----------------------------
    # Federated parameter
    # ----------------------------
    # def get_parameters(self, config):
    #     return [p.detach().cpu().numpy() for p in self._tx_params]
    #
    # def set_parameters(self, parameters):
    #     with torch.no_grad():
    #         for p, np_p in zip(self._tx_params, parameters):
    #             p.copy_(torch.as_tensor(np_p, dtype=torch.float32, device=p.device))
    #     if hasattr(self.agent, "rebuild_optimizer"):
    #         self.agent.rebuild_optimizer()

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.agent.policy.parameters()]

    def set_parameters(self, parameters):
        """Load parameters without breaking gradient graph."""
        with torch.no_grad():
            for p, np_p in zip(self.agent.policy.parameters(), parameters):
                tensor = torch.tensor(np_p, dtype=torch.float32, device=p.device)
                p.copy_(tensor)
        self.agent.rebuild_optimizer()

    # ----------------------------------------------------------
    # Federated operations
    # ----------------------------------------------------------
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.trainer.train_one_round()
        rnd = int(config.get("server_round", 0))

        cid = getattr(self, "cid", "unknown")
        success = self.trainer.save_eval(cid, rnd)

        samples = self.trainer.n_steps
        new_params = self.get_parameters(config)
        return new_params, samples, {"loss": float(loss), "success": int(success)}

    def evaluate(self, parameters, config):
        return 0.0, len(parameters), {"eval_acc": 0.0}
