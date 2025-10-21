import flwr as fl
import torch
import numpy as np


class FedGuideClient(fl.client.NumPyClient):
    """Flower client wrapper for FedGuide"""

    def __init__(self, agent, env, trainer):
        self.agent = agent
        self.env = env
        self.trainer = trainer

    # ----------------------------------------------------------
    # Parameter synchronization
    # ----------------------------------------------------------
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

        cid = getattr(self, "cid", "unknown")
        rnd = int(config.get("server_round", 0))
        success = self.trainer.save_eval(cid, rnd)

        samples = self.trainer.n_steps
        new_params = self.get_parameters(config)
        return new_params, samples, {"loss": loss, "success": int(success)}

    def evaluate(self, parameters, config):
        return 0.0, len(parameters), {"eval_acc": 0.0}
