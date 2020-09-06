#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# LICENSE file in the root directory of this source tree.

# this script is a prototype implementation of transferring Habitat trained agent to the real world/physics enabled environment.
# in summary, this script does the following:
# 1. import habitat trained model
# 2. subscribe to topic that contain sensor readings related to habitat trained model
# 3. obtain an action_id from trained model and command agent to perform that action. Then stop subscribing to the topic
# 4. Wait until agent finishes that action command or some time has passed
# 5. Repeat steps 2-4 until agent reaches the goal


# To run:
# 1. start roscore
# 2. source your ros related setup.bash files
# 3. run this script with python (not rosrun!)
# 4. If testing in physics enabled habitat environment, run hab_ros_interface.py. Important! depth camera need resolution 256x256
# 4.1 If testing in a gazebo environment, first roslaunch habitat_interface trained_agent_in_gazebo.launch, then rosrun habitat_interface get_pointnav_depth_obs.py
# 5. run roslaunch habitat_interface default.launch to visualize what the agent is doing

import sys
import os

PKG = "numpy_tutorial"
import roslib

roslib.load_manifest(PKG)

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from geometry_msgs.msg import Twist
import numpy as np

initial_sys_path = sys.path
sys.path = [b for b in sys.path if "2.7" not in b]
sys.path.insert(0, os.getcwd())


import argparse
import random

import numpy as np
import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.common.utils import batch_obs

import time


sys.path = initial_sys_path


pub_vel = rospy.Publisher("cmd_vel", Twist, queue_size=1)
rospy.init_node("controller_nn", anonymous=True)

# define some useful global variables
flag = 2
observation = {}
t_prev_update = time.time()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        # following are modified based on ppo_trainer.py
        actor_critic, batch, device, not_done_masks, test_recurrent_hidden_states = trainer.eval_bruce()

    def transform_callback(data):
        nonlocal actor_critic
        nonlocal batch
        nonlocal not_done_masks
        nonlocal test_recurrent_hidden_states
        global flag
        global t_prev_update
        global observation

        if flag == 2:
            observation["depth"] = np.reshape(data.data[0:-2], (256, 256, 1))
            observation["pointgoal_with_gps_compass"] = data.data[-2:]
            flag = 1
            return

        pointgoal_received = data.data[-2:]
        translate_amount = 0.25  # meters
        rotate_amount = 0.174533  # radians

        isrotated = (
            rotate_amount * 0.95
            <= abs(pointgoal_received[1] - observation["pointgoal_with_gps_compass"][1])
            <= rotate_amount * 1.05
        )
        istimeup = (time.time() - t_prev_update) >= 4

        # print('istranslated is '+ str(istranslated))
        # print('isrotated is '+ str(isrotated))
        # print('istimeup is '+ str(istimeup))

        if isrotated or istimeup:
            vel_msg = Twist()
            vel_msg.linear.x = 0
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 0
            pub_vel.publish(vel_msg)
            time.sleep(0.2)
            print("entered update step")

            # cv2.imshow("Depth", observation['depth'])
            # cv2.waitKey(100)

            observation["depth"] = np.reshape(data.data[0:-2], (256, 256, 1))
            observation["pointgoal_with_gps_compass"] = data.data[-2:]

            batch = batch_obs([observation])
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)
            if flag == 1:
                not_done_masks = torch.tensor([0.0], dtype=torch.float, device=device)
                flag = 0
            else:
                not_done_masks = torch.tensor([1.0], dtype=torch.float, device=device)

            _, actions, _, test_recurrent_hidden_states = actor_critic.act(
                batch, test_recurrent_hidden_states, not_done_masks, deterministic=True
            )

            action_id = actions.item()
            print(
                "observation received to produce action_id is "
                + str(observation["pointgoal_with_gps_compass"])
            )
            print("action_id from net is " + str(actions.item()))

            t_prev_update = time.time()
            vel_msg = Twist()
            vel_msg.linear.x = 0
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 0
            if action_id == 0:
                vel_msg.linear.x = 0.25 / 4
                pub_vel.publish(vel_msg)
            elif action_id == 1:
                vel_msg.angular.z = 10 / 180 * 3.1415926
                pub_vel.publish(vel_msg)
            elif action_id == 2:
                vel_msg.angular.z = -10 / 180 * 3.1415926
                pub_vel.publish(vel_msg)
            else:
                pub_vel.publish(vel_msg)
                sub.unregister()
                print("NN finished navigation task")

    sub = rospy.Subscriber(
        "depth_and_pointgoal", numpy_msg(Floats), transform_callback, queue_size=1
    )
    rospy.spin()


if __name__ == "__main__":
    main()
