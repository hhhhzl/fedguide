from abc import ABC, abstractmethod


class BasePrior(ABC):
    """Abstract class for guidance used in FedGuide."""

    def __init__(self):
        self.params = None

    @abstractmethod
    def log_prob(self, actions, states):
        pass

    def state_dict(self):
        return self.params

    def load_state_dict(self, state):
        self.params = state
