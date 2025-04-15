import gymnasium as gym
import argparse
import pickle
from datetime import datetime
from pathlib import Path

import mgmmz
from dcscg.core.env_interaction import collect_data
from dcscg.core.gym_dataset import GymDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env_id', type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n_steps', type=int, default=None)
    group.add_argument('-n_episodes', type=int, default=None)
    parser.add_argument('-out_file', type=str, default=None)
    args = parser.parse_args()

    if args.out_file is None:
        args.out_file = f'data/dataset_{args.env_id}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pkl'

    env = gym.make(args.env_id)
    policy = lambda obs: (env.action_space.sample(), 0.0)
    trajectories = collect_data(env, policy, args.n_steps, args.n_episodes)

    # we might want to process the trajectories in some way

    dataset = GymDataset(args.env_id, datetime.now() ,trajectories)
    dataset.store(args.out_file)

    #for trajectory in trajectories:
    #    for t, (o, a, a_logits, r, term, trunc, valid) in enumerate(trajectory):
    #        print(f'[{o}|{term}]', end=' ')
    #    print('', flush=True)
