from typing import Sequence, Callable

import jax.nn as jnn
import jax.random as jrand
from jax import jit
import equinox as eqx
import equinox.nn as nn


@jit
def identity(x):
    return x


class FNN(eqx.Module):
    data_classes: int
    layer_sizes: Sequence[int]
    is_relu: int
    layers: nn.Sequential
    use_bias: bool

    ridge_param: float
    lasso_param: float
    lasso_param_ratio: float
    group_lasso_param: float
    init_learn_rate: float
    adam_learn_rate: float
    adam_epsilon: float
    activation: Callable
    final_activation: Callable

    def __init__(self,
                 layer_sizes: Sequence[int] | int,
                 data_classes: int = 2,
                 is_relu: int = 1,
                 layer_nums: int | None = 0,
                 use_bias: bool = True,
                 lasso_param_ratio=0.1,
                 group_lasso_param=0.1,
                 ridge_param=0.1,
                 init_learn_rate=0.01,
                 adam_learn_rate=0.001,
                 adam_epsilon=1e-08,
                 *,
                 key: jrand.PRNGKey):
        super().__init__()

        print("initialize the neural network")

        self.data_classes = int(data_classes)
        self.use_bias = use_bias
        self.ridge_param = ridge_param
        self.lasso_param_ratio = lasso_param_ratio
        self.group_lasso_param = group_lasso_param
        self.init_learn_rate = init_learn_rate
        self.adam_learn_rate = adam_learn_rate
        self.adam_epsilon = adam_epsilon
        self.lasso_param = self.lasso_param_ratio * self.group_lasso_param

        if isinstance(layer_sizes, int):
            if layer_nums is None:
                ValueError(
                    "Layer Nums mustbe specified when layer_sizes is int")
            self.layer_sizes = [layer_sizes] * layer_nums
        else:
            self.layer_sizes = layer_sizes

        self.is_relu = is_relu
        if layer_sizes is not None:
            self._init_nn(key)

    def _init_nn(self, key):
        # categorical or continuous
        if self.data_classes < 2:
            num_out = 1
        else:
            num_out = self.data_classes

        # activation function
        if self.is_relu == 1:
            activation = jnn.relu
        else:
            activation = jnn.tanh

        layers = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            k1, k2 = jrand.split(key)
            hidden_layer = nn.Linear(
                fan_in, fan_out, key=k1, use_bias=self.use_bias)
            layers.append(hidden_layer)
            key = k2
        _, k2 = jrand.split(key)
        final_layer = nn.Linear(
            self.layer_sizes[-1], num_out, key=k2, use_bias=self.use_bias)
        layers.append(final_layer)
        if self.data_classes < 2:
            final_activation = identity
        else:
            final_activation = jnn.softmax
        self.layers = layers
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x
