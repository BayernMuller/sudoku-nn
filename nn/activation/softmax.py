import numpy as np

from base_activation import BaseActivation

class Softmax(BaseActivation):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.activate(x) * (1 - self.activate(x))
