import argparse

import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
import equinox as eqx
import optax
from optax import GradientTransformation

from model import FNN
from spinn import all_pen_loss, grad_loss


def make_step(model: FNN, optim, opt_state, x, y):
    loss, grads = grad_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return None, None, loss, model, opt_state


def make_step_adam_prox(model: FNN, optim, opt_state, x, y):
    all_loss, smooth_loss, unpen_loss, grads = all_pen_loss(
        model, x, y)
    updates, opt_state = optim.update(grads, opt_state)

   # do proximal gradient step
    values, treedef = jtu.tree_flatten(updates)
    adam_weights = values[0]

    # Do proximal gradient for lasso: soft threshold
    weights = model.layers[0].weight
    weigths_updated = jnp.multiply(jnp.sign(weights), jnp.maximum(
        jnp.abs(weights) - model.lasso_param * adam_weights, 0))

    # do proximal gradient step for group lasso
    group_norms = 1e-10 + jnp.linalg.norm(weights, axis=1).reshape(-1, 1)
    group_lasso_scale_factor = jnp.maximum(
        1 - model.group_lasso_param * adam_weights / group_norms, 0)
    weights_updated = jnp.multiply(group_lasso_scale_factor, weigths_updated)

    # update model
    values[0] = weights_updated - weights
    updates = jtu.tree_unflatten(treedef, values)
    model = eqx.apply_updates(model, updates)

    return all_loss, smooth_loss, unpen_loss, model, opt_state


def make_step_prox(model: FNN, learn_rate, x, y):
    all_loss, smooth_loss, unpen_loss, grads = all_pen_loss(
        model, x, y)
    updates = jtu.tree_map(lambda x: model.lasso_param * learn_rate * x, grads)
    model = eqx.apply_updates(model, updates)
    # do proximal gradient step
    values, treedef = jtu.tree_flatten(updates)

    # Do proximal gradient for lasso: soft threshold
    weights = model.layers[0].weight
    weigths_updated = jnp.multiply(jnp.sign(weights), jnp.maximum(
        jnp.abs(weights) - model.lasso_param * learn_rate, 0))

    # do proximal gradient step for group lasso
    group_norms = 1e-10 + jnp.linalg.norm(weights, axis=1).reshape(-1, 1)
    group_lasso_scale_factor = jnp.maximum(
        1 - model.group_lasso_param * learn_rate / group_norms, 0).reshape(-1, 1)
    weights_updated = jnp.multiply(group_lasso_scale_factor, weigths_updated)

    for i in range(len(values)):
        values[i] = jnp.zeros_like(values[i])

    values[0] = weights_updated - weights
    updates = jtu.tree_unflatten(treedef, values)
    model = eqx.apply_updates(model, updates)
    return all_loss, smooth_loss, unpen_loss, model
