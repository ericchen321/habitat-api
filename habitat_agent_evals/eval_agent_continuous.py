#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random

import numpy as np
import torch
from gym.spaces import Box, Dict, Discrete

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat_baselines.common.utils import batch_obs
from habitat_baselines.rl.ppo import PointNavBaselinePolicy
from habitat_baselines.agents.ppo_agents import PPOAgent


def get_default_config():
    c = Config()
    c.INPUT_TYPE = "blind"
    c.MODEL_PATH = "data/checkpoints/blind.pth"
    c.RESOLUTION = 256
    c.HIDDEN_SIZE = 512
    c.RANDOM_SEED = 7
    c.PTH_GPU_ID = 0
    c.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/pointnav.yaml"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50
    )
    # control period defines the amount of time an agent should take to complete
    # an action. Measured in seconds
    parser.add_argument(
        "--control-period", type=float, default=1.0
    )

    args = parser.parse_args()

    config = get_config(args.task_config)

    agent_config = get_default_config()
    agent_config.INPUT_TYPE = args.input_type
    agent_config.MODEL_PATH = args.model_path
    num_episodes = args.num_episodes
    control_period = args.control_period

    agent = PPOAgent(agent_config)
    print("Establishing benchmark:")
    benchmark = habitat.Benchmark(config_paths=args.task_config, enable_physics=True)
    print("Evaluating:")
    metrics = benchmark.evaluate(agent, num_episodes=num_episodes, control_period=control_period) # eval 50 episodes for now

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
