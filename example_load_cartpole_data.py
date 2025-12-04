import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from odgn.gym_dataset import GymDataset
from odgn.trajectory import Trajectory


def transform_obs_to_image(o, size=512):
    """
    Transform CartPole environment observations to images.
    According to https://gymnasium.farama.org/environments/classic_control/cart_pole/#observation-space,
    the observation consists of 
    [
        cart position, 
        cart velocity, 
        pole angle, 
        pole angular velocity
    ]
    We unpack the obsevation to draw the cart and the pole according to the
    gathered information.
    """
    x_limits = [-4.8, 4.8]  # from docs

    cart_w_perc = 0.16
    cart_h_perc = 0.1
    cart_col = (0.5, 0.5, 0.5)

    pole_l_perc = 0.2
    pole_s_perc = 0.02
    pole_col = (0.0, 0.0, 0.0)

    floor_h_perc = 0.1
    floor_col = (0.1, 0.35, 0.2)

    x, xdot, alpha, alphadot = o

    cart_rel = (x - x_limits[0]) / (x_limits[1] - x_limits[0])
    cart_center = cart_rel * size
    cart_l = (cart_rel - cart_w_perc / 2) * size 
    cart_r = (cart_rel + cart_w_perc / 2) * size
    cart_t = (floor_h_perc + cart_h_perc) * size
    cart_b = floor_h_perc * size
    tip_x = np.sin(alpha) * pole_l_perc + cart_rel
    tip_y = np.cos(alpha) * pole_l_perc + floor_h_perc
    tip_l = np.clip((tip_x - pole_s_perc / 2) * size, 0, size)
    tip_r = np.clip((tip_x + pole_s_perc / 2) * size, 0, size)
    tip_t = np.clip((tip_y + pole_s_perc / 2) * size, 0, size)
    tip_b = np.clip((tip_y - pole_s_perc / 2) * size, 0, size)
    floor_t = floor_h_perc * size
    
    cart_center = round(cart_center)
    cart_l = round(cart_l)
    cart_r = round(cart_r)
    cart_t = round(cart_t)
    cart_b = round(cart_b)
    tip_l = round(tip_l)
    tip_r = round(tip_r)
    tip_t = round(tip_t)
    tip_b = round(tip_b)
    floor_t = round(floor_t)

    img_obs = np.ones(shape=(size, size, 3), dtype=float)
    img_obs[cart_l:cart_r, cart_b:cart_t] = cart_col        # cart
    img_obs[:, :floor_t] = floor_col                        # floor
    img_obs[:, floor_t] = 0                                 # floor outline
    img_obs[cart_center, cart_b:cart_t] = 0
    img_obs[tip_l:tip_r, tip_b:tip_t] = pole_col

    return np.flip(img_obs.swapaxes(0, 1), 0)


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

    traj_lens = [len(traj) for traj in dset]
    longest = np.argmax(traj_lens)
    animate_trajectory(dset[longest])


if __name__ == '__main__':
    main()
