from abc import ABC, abstractmethod
import numpy as np

class BaseActivation(ABC):
    @abstractmethod
    def activate(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass
