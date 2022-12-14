from typing import Sequence

import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrand
from jax import vmap

from model import FNN
from spinn import all_pen_loss
from data_gen.data_generator import DataGenerator
from data_gen.data_gen_funs import six_variable_linear_func1, last_six_variable_linear_func2
from data_gen.dataloader import dataloader
from altmin_schedular import allocate_model, collect_data_groups

if __name__ == '__main__':
    key1 = jrand.PRNGKey(0)
    key2 = jrand.PRNGKey(1)
    key3 = jrand.PRNGKey(2)
    m1 = FNN(layer_sizes=[100, 10], group_lasso_param=0.99, data_classes=1, key=key1)
    m2 = FNN(layer_sizes=[100, 10], group_lasso_param=0.99, data_classes=1, key=key2)
    dt1 = DataGenerator(100, six_variable_linear_func1,
                        False, group=0, snr=2).create_data(2000)
    dt2 = DataGenerator(100, last_six_variable_linear_func2,
                        False, group=1, snr=2).create_data(2000)
    x = np.vstack([dt1.x, dt2.x])
    y = np.vstack([dt1.y, dt2.y])
    group0 = np.zeros((dt1.x.shape[0], 1))
    group1 = np.ones((dt2.x.shape[0], 1))
    group = np.vstack([group0, group1])
    x_, y_, group_ = next(dataloader([x, y, group], 32, key=key3))
    z = allocate_model([m1, m2], x_, y_)
    print(z)
    x0 = collect_data_groups(0, x_, y_, group_, z)
    print(x0.shape)