from typing import Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrand
import equinox as eqx


from model import FNN
from altmin import get_codes, compute_codes_loss
from spinn import all_pen_loss, l2_reg, l1_reg

if __name__ == '__main__':
    key = jrand.PRNGKey(0)
    m = FNN(layer_sizes=2, layer_nums=2, key=key)
    x = jnp.ones((2, ))
    x, codes = get_codes(m.layers, x)


