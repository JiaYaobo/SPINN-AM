from typing import Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrand
import equinox as eqx


from model import FNN
from altmin import get_codes, ridge_reg

if __name__ == '__main__':
    key = jrand.PRNGKey(0)
    m = FNN(layer_sizes=2, layer_nums=10, key=key)
    x = jnp.ones((2, ))
    # tree, treedef = jtu.tree_flatten(eqx.filter(m, filter_spec=eqx.is_inexact_array))
    trainable = eqx.filter(m, filter_spec=eqx.is_inexact_array)
    value, grads = ridge_reg(trainable)
    print(jtu.tree_flatten(trainable)[0])
    print(jtu.tree_flatten(grads)[0])


