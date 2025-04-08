from abc import ABC, abstractmethod
import numpy as np

from nn.activation.base_activation import BaseActivation

class BaseLayer(ABC):
    def __init__(self, input_size: int, output_size: int, activation: BaseActivation):
        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._weights, self._biases = self._initialize_weights(input_size, output_size)
    
    @abstractmethod
    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward_propagation(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        pass

    @property
    def input_size(self) -> int:
        return self._input_size
    
    @property
    def output_size(self) -> int:
        return self._output_size
    
    @property
    def weights(self) -> np.ndarray:
        return self._weights
    
    @property
    def biases(self) -> np.ndarray:
        return self._biases
    
    def _initialize_weights(self, input_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
        w = np.random.randn(input_size, output_size) * 0.01
        b = np.zeros(output_size)
        return w, b
