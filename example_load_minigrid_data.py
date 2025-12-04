import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from odgn.gym_dataset import GymDataset
from odgn.trajectory import Trajectory
from minigrid.core.constants import IDX_TO_OBJECT

MY_COLORS = {
    "unseen": np.array([0, 50, 50]),
    "empty": np.array([0, 0, 0]),
    "wall": np.array([100, 100, 100]),
    "red": np.array([30, 30, 30]),
    "door": np.array([0, 255, 0]),
    "key": np.array([0, 0, 255]),
    "ball": np.array([112, 39, 195]),
    "box": np.array([255, 255, 0]),
    "goal": np.array([255, 255, 255]),
    "lava": np.array([255, 128, 0]),
    "agent": np.array([255, 0, 255])
}

def transform_to_color(vec: np.ndarray) -> np.ndarray:
    object_name = IDX_TO_OBJECT[vec[0]]
    color = MY_COLORS[object_name]
    return color

def transform_obs_to_image(obs):
    o_flat = obs['image'].reshape(-1, 3)
    color_o = np.apply_along_axis(transform_to_color, 1, o_flat)
    color_o = color_o.reshape(*obs['image'].shape)
    return color_o.transpose((1, 0, 2))

def animate_trajectory(traj: Trajectory) -> None:
    # Extract data from the trajectory.
    observations = traj.o  # Images (assumed to be numpy arrays)
    observations = [transform_obs_to_image(x) for x in observations]
    rewards = np.array(traj.r)  # Rewards (numeric values)
    terminals = np.array(traj.term)  # Terminal flags (assumed bool or 0/1)
    truncateds = np.array(traj.trunc)  # Truncated flags (assumed bool or 0/1)

    frames = len(rewards)

    # Create a figure with two subplots: one for the image, one for the reward plot.
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(10, 5))

    # --- Left panel: Observation ---
    img_disp = ax_img.imshow(observations[0])
    ax_img.set_title("Observation")

    # --- Right panel: Reward plot ---
    reward_line, = ax_plot.plot([], [], lw=2, label='Reward')
    ax_plot.set_xlim(0, frames - 1)
    ax_plot.set_ylim(np.min(rewards) - 1, np.max(rewards) + 1)
    ax_plot.set_title("Reward Plot")
    ax_plot.set_xlabel("Step")
    ax_plot.set_ylabel("Reward")

    # Pre-initialize scatter plots for terminal and truncated flags.
    term_scatter = ax_plot.scatter([], [], marker='x', color='red', label='Terminal')
    trunc_scatter = ax_plot.scatter([], [], marker='o', color='blue', label='Truncated')
    ax_plot.legend()

    def init():
        reward_line.set_data([], [])
        term_scatter.set_offsets(np.empty((0, 2)))
        trunc_scatter.set_offsets(np.empty((0, 2)))
        return img_disp, reward_line, term_scatter, trunc_scatter

    def animate(i):
        # Update observation image (if available).
        if i < len(observations):
            img_disp.set_array(observations[i])

        # Update reward line up to the current frame.
        x = np.arange(i + 1)
        reward_line.set_data(x, rewards[:i + 1])

        # Update scatter points for terminal and truncated flags.
        term_idx = np.where(terminals[:i + 1])[0]
        trunc_idx = np.where(truncateds[:i + 1])[0]

        if term_idx.size > 0:
            term_points = np.column_stack((term_idx, rewards[term_idx]))
        else:
            term_points = np.empty((0, 2))
        if trunc_idx.size > 0:
            trunc_points = np.column_stack((trunc_idx, rewards[trunc_idx]))
        else:
            trunc_points = np.empty((0, 2))

        term_scatter.set_offsets(term_points)
        trunc_scatter.set_offsets(trunc_points)

        return img_disp, reward_line, term_scatter, trunc_scatter

    # Create the animation: adjust the interval as needed.
    ani = animation.FuncAnimation(
        fig, animate, frames=frames, init_func=init, blit=False, interval=200
    )

    plt.tight_layout()
    plt.show()

def main():
    dset = GymDataset.load('dataset')
    print(type(dset))
    for traj in dset[:10]:
        print(traj)

    animate_trajectory(dset[0])


if __name__ == '__main__':
    main()
