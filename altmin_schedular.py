from typing import Sequence

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import vmap
import equinox as eqx

from model import FNN


def allocate_model(models: Sequence[FNN], x, y):

    seq = list(range(len(models)))

    @eqx.filter_jit
    def _allocate_model(models: Sequence[FNN], xi, yi):
        abs_err = jnp.array(jtu.tree_map(
            lambda i: jnp.abs(models[i](xi) - yi), seq))
        return jnp.argmin(abs_err)

    return vmap(_allocate_model, in_axes=(None, 0, 0))(models, x, y)


def collect_data_groups(which_group, x, y, group, z):
    return x[z == which_group, ], y[z == which_group, ], group[z == which_group, ]
