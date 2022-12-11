import argparse
from typing import Sequence

import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
import equinox as eqx
import optax

from model import FNN
from train_step import make_step_adam_prox, make_step_prox
from data_gen.data_gen_funs import six_variable_linear_func1, last_six_variable_linear_func2
from data_gen.data_generator import DataGenerator
from data_gen.dataloader import dataloader
from altmin_schedular import allocate_model, collect_data_groups, batch_warmup



def get_dataset(num_p, num_groups, n_obs, func_list):
    assert (num_groups == len(func_list))
    if isinstance(n_obs, int):
        n_obs = [n_obs] * num_groups
    x = np.array([]).reshape(0, num_p)
    y = np.array([]).reshape(0, 1)
    group = np.array([]).reshape(0, 1)
    for i in range(num_groups):
        dt = DataGenerator(num_p, func_list[i],
                           False, group=i, snr=2).create_data(n_obs[i])
        x = np.vstack([dt.x, x])
        y = np.vstack([dt.y, y])
        group_ = np.ones((dt.x.shape[0], 1)) * i
        group = np.vstack([group_, group])

    return x, y, group


def train(args):
    # ensure correct input
    assert (args.layer_sizes[0] == args.num_p)
    key = jrand.PRNGKey(args.seed)
    loader_key, *model_keys = jrand.split(key, args.num_groups + 1)

    models: Sequence[FNN] = []
    opt_states = []
    optims = []
    for i in range(args.num_groups):
        model = FNN(
            layer_sizes=args.layer_sizes,
            data_classes=args.data_classes,
            is_relu=args.is_relu,
            layer_nums=args.layer_nums,
            use_bias=args.use_bias,
            lasso_param_ratio=args.lasso_param_ratio,
            group_lasso_param=args.group_lasso_param,
            ridge_param=args.ridge_param,
            init_learn_rate=args.init_learn_rate,
            adam_learn_rate=args.adam_learn_rate,
            adam_epsilon=args.adam_epsilon,
            key=model_keys[i]
        )
        optim = optax.adam(args.adam_learn_rate, eps=args.adam_epsilon)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        models.append(model)
        opt_states.append(opt_state)
        optims.append(optim)

    x, y, group = get_dataset(args.num_p, args.num_groups, [2000, 2000], func_list=[
                             six_variable_linear_func1, last_six_variable_linear_func2])
    
    def train_mixed_models_with_single_sample_adam(xi, yi, groupi, z):
        all_loss, smooth_loss, unpen_loss, models[z], opt_states[z] = make_step_adam_prox(
            models[z], optims[z], opt_states[z], xi, yi)
        if step % args.print_every == 0 or step == args.max_iters - 1:
                print(
                f"Model: {z}, batch_size: {1} Step: {step}, All Loss: {all_loss}, Smooth Loss: {smooth_loss}, Unpen Loss: {unpen_loss}")

    def train_mixed_models_with_batch(xi, yi, groupi, z):
        lr = args.adam_learn_rate
        flag = -1
        for i in range(args.num_groups):
            if flag == 1:
                break
            if models[i].support() == 0:
                flag += 1
                continue
            xi_, yi_, groupi_ = collect_data_groups(i, xi, yi, groupi, z)
            all_loss, smooth_loss, unpen_loss, models[i], opt_states[i], lr = make_step_adam_prox(
                models[i], optims[i], opt_states[i], xi_, yi_, lr)
            if step % args.print_every == 0 or step == args.max_iters - 1:
                 print(
                    f"Model: {i}, batch_size: {groupi_.size} Step: {step}, All Loss: {all_loss}, Smooth Loss: {smooth_loss}, Unpen Loss: {unpen_loss}, support: {models[i].support()}")
            

    # lr = args.init_learn_rate
    for step, (xi, yi, groupi) in zip(range(args.max_iters), dataloader(
            [x, y, group], args.batch_size, key=loader_key)
    ):
        # if step == 0:
        #     z = batch_warmup(args.num_groups, xi, yi)
        # else:
        #     z = allocate_model(models, xi, yi)
        # for i in range(args.num_groups):
        #     xi_, yi_, groupi_ = collect_data_groups(i, xi, yi, groupi, z)
        #     all_loss, smooth_loss, unpen_loss, models[i], = make_step_prox(
        #         models[i], lr, xi_, yi_)
        #     if step % args.print_every == 0 or step == args.max_iters - 1:
        #          print(
        #             f"Model: {i}, batch_size: {groupi_.size} Step: {step}, All Loss: {all_loss}, Smooth Loss: {smooth_loss}, Unpen Loss: {unpen_loss}, Support: {models[i].support()}")
        # lr = lr * 0.98
        # train_mixed_models_with_batch(xi, yi, groupi, z)
        # z = allocate_model(models, xi, yi)[0]
        # train_mixed_models_with_single_sample(xi, yi, groupi, z)
        z = allocate_model(models, xi, yi)
        train_mixed_models_with_batch(xi, yi, groupi, z)
    print("Stop training")
    for i in range(args.num_groups):
        print(f"Model: {i}, support: {models[i].support()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer-sizes', '--list',
                        nargs='+',  type=int, required=True)
    parser.add_argument('--data-classes', type=int, default=1)
    parser.add_argument('--layer-nums', type=int)
    parser.add_argument('--init-learn-rate', type=float, default=1e-3)
    parser.add_argument('--adam-learn-rate', type=float, default=1e-3)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--is-relu', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use-bias', action='store_true')
    parser.add_argument('--ridge-param', type=float, default=0.1)
    parser.add_argument('--lasso-param-ratio', type=float, default=0.1)
    parser.add_argument('--group-lasso-param', type=float, default=0.95)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-iters', type=int, default=1)
    parser.add_argument('--seed', type=int, default=520)
    parser.add_argument('--num-p', type=int, default=100)
    parser.add_argument('--num-groups', type=int, default=1)
    parser.add_argument('--n-obs', type=int, default=2000)
    parser.add_argument('--print-every', type=int, default=200)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
