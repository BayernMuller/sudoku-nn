import numpy as np

from base_activation import BaseActivation

class ReLU(BaseActivation):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)