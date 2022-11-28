import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

from jax import jit
import equinox as eqx
import equinox.nn as nn
import optax


class Flatten(eqx.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: jnp.ndarray):
        return x.reshape(x.size, -1)



@eqx.filter_value_and_grad
def ridge_reg(model):
    layers = jtu.tree_flatten(model.layers[1:])
    return jnp.sum(jnp.asarray(jtu.tree_map(l2_loss, layers[0])))


@eqx.filter_jit
def l2_loss(x):
    if x.ndim == 2:
        x = x.reshape(-1)
        return 0.5 * jnp.sum(x ** 2)
    else:
        return 0


@eqx.filter_jit
def l1_loss(x):
    x = x.reshape(-1)
    return jnp.sum(jnp.abs(x))


def get_codes(layers: nn.Sequential, x: jnp.ndarray):
    codes = []
    # ignore last layer
    for m in range(len(layers) - 1):
        if isinstance(layers[m], nn.Linear | nn.Conv2d | nn.Conv1d | nn.Conv3d):
            x = layers[m](x)
            codes.append(x.copy())
    return x, codes


def update_codes(codes, layers, targets, criterion, mu, lambda_c, n_iter, lr):
    pass