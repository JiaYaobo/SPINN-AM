import jax
import jax.numpy as jnp
import equinox as eqx

from jax import vmap
from jax import tree_util as jtu


@eqx.filter_jit
def mean_square_loss(model, x: jnp.ndarray, y_true: jnp.ndarray):
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
    x = x.reshape(-1)
    return 0.5 * jnp.sum(x ** 2)


@eqx.filter_jit
def l1_loss(x):
    x = x.reshape(-1)
    return jnp.sum(jnp.abs(x))


@eqx.filter_jit
def ridge_reg(model):
    layers, _ = jtu.tree_flatten(model.layers[1:])
    return jnp.sum(jnp.asarray(jtu.tree_map(l2_loss, layers)))


@eqx.filter_value_and_grad
def ridge_loss(model):
    loss = ridge_reg(model)
    return 0.5 * model.ridge_param * loss


@eqx.filter_jit
def l1_reg(model):
    w = model.layers[0].weight
    return l1_loss(w)


@eqx.filter_jit
def l2_reg(model):
    w = model.layers[0].weight
    return jnp.sum(jnp.sqrt(jnp.sum(w ** 2, axis=1)))


@eqx.filter_value_and_grad
def grad_loss(model, x, y):
    y_pred = jax.vmap(model, in_axes=(0))(x)
    return jnp.mean((y - y_pred) ** 2)


@eqx.filter_jit
def all_pen_loss(model, x, y):
    unpen_loss, unpen_grads = grad_loss(model ,x, y)
    rdg_loss, pen_grads = ridge_loss(model)
    grads = eqx.apply_updates(unpen_grads, pen_grads)
    smooth_loss = rdg_loss + unpen_loss
    all_loss = smooth_loss + (1 - model.group_lasso_param) * l1_reg(model) + model.group_lasso_param * l2_reg(model)
    return all_loss, smooth_loss, unpen_loss, grads
