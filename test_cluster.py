from typing import Sequence

import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrand
from jax import vmap

from sklearn.cluster import KMeans

from model import FNN
from spinn import all_pen_loss
from data_gen.data_generator import DataGenerator
from data_gen.data_gen_funs import six_variable_linear_func1, six_variable_additive_func, last_six_variable_linear_func2
from data_gen.dataloader import dataloader
from altmin_schedular import allocate_model, collect_data_groups

if __name__ == '__main__':
    key1 = jrand.PRNGKey(0)
    key2 = jrand.PRNGKey(1)
    key3 = jrand.PRNGKey(2)
    x = jrand.normal(key1, (50, 100))
    print(jnp.linalg.norm(x, axis=1).shape)
    # m1 = FNN(layer_sizes=[100, 10], group_lasso_param=0.99, data_classes=1, key=key1)
    # m2 = FNN(layer_sizes=[100, 10], group_lasso_param=0.99, data_classes=1, key=key2)
    # dt1 = DataGenerator(100, last_six_variable_linear_func2,
    #                     False, group=0, snr=2).create_data(2000)
    # dt2 = DataGenerator(100, six_variable_additive_func,
    #                     False, group=1, snr=2).create_data(2000)
    # x = np.vstack([dt1.x, dt2.x])
    # y = np.vstack([dt1.y, dt2.y])
    # group0 = np.zeros((dt1.x.shape[0], 1))
    # group1 = np.ones((dt2.x.shape[0], 1))
    # group = np.vstack([group0, group1])
    # data = np.hstack([x, y])
    # data_, group_ = next(dataloader([data,  group], 512, key=key3))
    # group_ = group_.reshape(-1).astype(int)
    # yy_ = data_[:,-1].reshape(-1, 1)
    # kms = KMeans(n_clusters=2)
    # kms.fit(yy_)
    # print(1- np.mean(np.abs(kms.labels_ - group_)))
