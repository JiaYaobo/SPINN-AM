import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrand
import optax
import equinox as eqx
import equinox.nn as nn

from jax import vmap, jit, lax
from jax import tree_util as jtu

from typing import Sequence, Any, Callable



@eqx.filter_jit
def mean_square_loss(model, x, y_true):
    y_pred = vmap(model, in_axes=0)(x)
    return jnp.mean((y_pred - y_true) ** 2)


@eqx.filter_jit
def cross_entropy_loss(model, x, label):
    pred = vmap(model, in_axes=0)(x)
    bxe = label * jnp.log(pred) + (1 - label) * jnp.log(1-pred)
    bxe = -jnp.mean(bxe)
    return bxe




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


@eqx.filter_jit
def ridge_reg(model):
    layers = jtu.tree_flatten(model.layers[1:])
    return jnp.sum(jnp.asarray(jtu.tree_map(l2_loss, layers[0])))


@eqx.filter_value_and_grad
def smooth_pen_loss(model):
    loss = ridge_reg(model)
    return 0.5 * model.ridge_param * loss 




@eqx.filter_jit
def l1_reg(model):
    return l1_loss(model.layers[0].weight)


@eqx.filter_jit
def l2_reg(model):
    w = model.layers[0].weight.reshape(-1)
    return jnp.sum(jnp.sqrt(jnp.sum(w ** 2, axis=1)))


# @eqx.filter_jit
def all_pen_loss(model, loss_func, data):
    x, y = data.x, data.y
    unpen_loss, unpen_grads = eqx.filter_value_and_grad(loss_func)(model, x, y)
    smooth_loss, pen_grads = smooth_pen_loss(model)
    grads = eqx.apply_updates(unpen_grads, pen_grads)
    return smooth_loss + model.lasso_param * l1_reg(model) + model.group_lasso_param * l2_reg(model), smooth_loss + unpen_loss, unpen_loss, grads

