"""This module stores the functions for trajectory set generation."""

from typing import List, Union
import pickle
import numpy as np
from moviepy.editor import ImageSequenceClip
import warnings
import os

from scipy.spatial.transform import Rotation
from aprel.basics import Environment, Trajectory, TrajectorySet


def generate_trajectories_randomly(
    env: Environment,
    num_trajectories: int,
    max_episode_length: int = None,
    file_name: str = None,
    restore: bool = False,
    headless: bool = False,
    seed: int = None,
) -> TrajectorySet:
    """
    Generates :py:attr:`num_trajectories` random trajectories, or loads (some
    of) them from the given file.

    Args:
        env (Environment): An :class:`.Environment` instance containing the
        OpenAI Gym environment to be simulated.
        num_trajectories (int): the number of trajectories to generate.
        max_episode_length (int): the maximum number of time steps for the new t
        rajectories. No limit is assumed if None (or not given).
        file_name (str): the file name to save the generated trajectory set
          and/or restore the trajectory set from.
            :Note: If :py:attr:`restore` is true and so a set is being
            restored, then the restored file will be overwritten with the new
            set.
        restore (bool): If true, it will first try to load the trajectories
        from :py:attr:`file_name`. If the file has fewer trajectories
            than needed, then more trajectories will be generated to
            compensate the difference.
        headless (bool): If true, the trajectory set will be saved and
        returned with no visualization. This makes trajectory generation
            faster, but it might be difficult for real humans to compare
            trajectories only based on the features without any visualization.
        seed (int): Seed for the randomness of action selection.
            :Note: Environment should be separately seeded. This seed is only
            for the action selection.

    Returns:
        TrajectorySet: A set of :py:attr:`num_trajectories` randomly generated
        trajectories.

    Raises:
        AssertionError: if :py:attr:`restore` is true, but no :py:attr:`
        file_name` is given.
    """
    assert not (
        file_name is None and restore
    ), "Trajectory set cannot be restored, because no file_name is given."
    max_episode_length = 20
    if restore:
        try:
            with open("aprel_trajectories/" + file_name + ".pkl", "rb") as f:
                trajectories = pickle.load(f)
        except:
            warnings.warn(
                "Ignoring restore=True, because 'aprel_trajectories/"
                + file_name
                + ".pkl' is not found."
            )
            trajectories = TrajectorySet([])
        # if not headless:
        #     for traj_no in range(trajectories.size):
        #         if trajectories[traj_no].clip_path is None or not os.path.isfile(
        #             trajectories[traj_no].clip_path
        #         ):
        #             warnings.warn(
        #                 "Ignoring restore=True, because headless=False and some trajectory clips are missing."
        #             )
        #             trajectories = TrajectorySet([])
        #             break
    else:
        trajectories = TrajectorySet([])

    if not os.path.exists("aprel_trajectories"):
        os.makedirs("aprel_trajectories")

    if trajectories.size >= num_trajectories:
        trajectories = TrajectorySet(trajectories[:num_trajectories])
    else:
        # env_has_rgb_render = env.render_exists and not headless
        env_has_rgb_render = False
        if env_has_rgb_render and not os.path.exists("aprel_trajectories/clips"):
            os.makedirs("aprel_trajectories/clips")
        env.action_space.seed(seed)
        # get the image size from the environment
        # camera_width = env.env.env.env.env.camera_widths[0]
        # camera_height = env.env.env.env.env.camera_heights[0]

        # Randomly sample a goal configuration
        # goal_q = env.env.env.env.env.env._sample_valid_pos()

        goal_q = np.array([np.pi / 4, 0, -np.pi / 2, 0, -np.pi / 2, 0])

        # calculate the forward kinematics of the goal configuration
        goal_pos, goal_rot = env.robots[0].robot_model.get_eef_transformation(goal_q)

        goal_pose = np.concatenate(
            (
                goal_pos.copy() - env.base_position,
                Rotation.from_matrix(goal_rot).as_quat(),
            )
        )

        env.env.env.env.env.env.desired_q_goal = goal_q

        for traj_no in range(trajectories.size, num_trajectories):
            traj = []
            eef_traj = []
            joint_states = []
            # obs, _ = env.reset()
            obs = env.reset()

            # obs consists of a flattened numpy array consisting of two observations:
            # 1. The cartesian distance between the robot end-effector and the goal
            # 2. The camera image

            # goal_dist = obs[:3]
            eef_pos = obs[:3] - env.base_position
            # extract the pixel values of the rendered image from the flattened observation
            # and reshape it to 3 channels
            # image = obs[3:].reshape(camera_height, camera_width, 3)

            # if env_has_rgb_render:
            #     try:
            #         # extract the pixel values of the rendered image from the observation
            #         frames = [image]
            #     except Exception as e:
            #         print("Exception: ", e)
            #         env_has_rgb_render = False
            done = False
            t = 0
            random_action_step_duration = 10
            while not done and t < max_episode_length:
                # generate a new action every random_action_step_duration steps
                if t % random_action_step_duration == 0:
                    act = env.action_space.sample()
                # cut action to control dim
                ws_action = np.zeros(env.control_dim)
                ws_action[: env.control_dim] = act[: env.control_dim]

                # traj.append((obs, act))

                # obs, _, terminated, truncated, _ = env.step(act)
                obs, _, done, _ = env.step(act)
                #goal_dist = obs[:3]
                eef_pos = obs[:3] - env.base_position
                #traj.append((goal_dist, ws_action))
                traj.append((eef_pos, ws_action))
                # image = obs[3:].reshape(camera_height, camera_width, 3)
                joint_state = obs[3: 3 + 7]
                joint_states.append(joint_state)
                # done = terminated or truncated
                t += 1

                # get the end-effector position
                eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
                eef_traj.append(eef_pos.copy() - env.base_position)

                # if env_has_rgb_render:
                #     frames.append(image)
                # markers = env.sim._render_context_offscreen._markers
                # clear the markers from the previous step
                # env.sim._render_context_offscreen._markers.clear()

            # print('traj: ', traj)
            # traj.append((obs, None))
            # traj.append((goal_dist, None))
            traj.append((eef_pos, None))
            # if env_has_rgb_render:
            #     clip = ImageSequenceClip(frames, fps=30)
            #     clip_path = (
            #         "aprel_trajectories/clips/"
            #         + file_name
            #         + "_"
            #         + str(traj_no)
            #         + ".mp4"
            #     )
            #     print("writing clip to: ", clip_path)
            #     clip.write_videofile(clip_path, audio=False)
            # else:
            clip_path = None
            trajectories.append(
                Trajectory(
                    env,
                    traj,
                    clip_path,
                    goal=goal_pose,
                    joint_states=joint_states,
                    eef_traj=eef_traj,
                )
            )
            # env.sim._render_context_offscreen._markers.clear()
            # if env.close_exists:
            #     env.close()

    with open("aprel_trajectories/" + file_name + ".pkl", "wb") as f:
        pickle.dump(trajectories, f)

    if not headless and trajectories[-1].clip_path is None:
        warnings.warn(
            (
                "headless=False was set, but either the environment is missing "
                "a render function or the render function does not accept "
                "mode='rgb_array'. Automatically switching to headless mode."
            )
        )
    if headless:
        for traj_no in range(trajectories.size):
            trajectories[traj_no].clip_path = None
    return trajectories
