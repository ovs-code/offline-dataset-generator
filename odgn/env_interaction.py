from typing import Callable, Tuple
import sys

import gymnasium as gym
import tqdm

from odgn.trajectory import Trajectory
from odgn.constants import Numeric

def collect_data(
        env: gym.Env,
        policy: Callable[[Numeric], Tuple[Numeric, Numeric]],
        n_steps: int | None = None,
        n_episodes: int | None = None,
):
    if n_steps is None:
        progress_bar = tqdm.tqdm(total=n_episodes)
        update_pbar_step = lambda: None
        update_pbar_ep = lambda: progress_bar.update(1)
        n_steps = -1
    else:
        progress_bar = tqdm.tqdm(total=n_steps)
        update_pbar_step = lambda: progress_bar.update(1)
        update_pbar_ep = lambda: None
        n_episodes = -1

    step_counter, episode_counter = 0, 0
    trajectories = []

    trajectory = Trajectory()
    last_o, _ = env.reset()
    while progress_bar:
        a, a_logits = policy(last_o)
        o, r, term, trunc, _ = env.step(a)

        trajectory.add_step(last_o, a, a_logits, r, term, trunc)

        if term or trunc:
            trajectory.add_final_obs(o)
            trajectories.append(trajectory)
            trajectory = Trajectory()
            episode_counter += 1
            update_pbar_ep()
            o, _ = env.reset()

            if episode_counter == n_episodes or step_counter == n_steps:
                break

        step_counter += 1

        if step_counter == n_steps:
            trajectory.add_final_obs(o)
            trajectories.append(trajectory)
            break

        last_o = o
        update_pbar_step()

    progress_bar.close()

    return trajectories

