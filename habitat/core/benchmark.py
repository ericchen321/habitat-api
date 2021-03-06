#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

import os
from collections import defaultdict
from typing import Dict, Optional

from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env, RLEnv

# use TensorBoard to visualize
from habitat_agent_evals.tensorboard_utils_new import TensorboardWriter, generate_video
from habitat.utils.visualizations.utils import observations_to_image
import numpy as np

# created based on SimpleRLEnv from shortest_path_follower_example.py
class BenchmarkingRLEnv(RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

class Benchmark:
    r"""Benchmark for evaluating agents in environments.
    """

    def __init__(
        self, config_paths: Optional[str] = None, eval_remote=False, enable_physics=False
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        config_env = get_config(config_paths)
        # embed top-down map and heading sensor in config
        config_env.defrost()
        config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config_env.TASK.SENSORS.append("HEADING_SENSOR")
        config_env.freeze()
        self._eval_remote = eval_remote

        self._enable_physics = enable_physics

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = BenchmarkingRLEnv(config=config_env)


    def remote_evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None
    ):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-api repository.
        import evaluation_pb2
        import evaluation_pb2_grpc
        import evalai_environment_habitat
        import grpc
        import pickle
        import time

        time.sleep(60)

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            return res_env["episode_over"]

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(env_address_port)
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        while count_episodes < num_episodes:
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )

            while not remote_ep_over(stub):
                obs = res_env["observations"]
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(
                            SerializedEntity=pack_for_grpc(action)
                        )
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                ).SerializedEntity
            )

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    def local_evaluate(self, agent: Agent, num_episodes: Optional[int] = None, control_period: Optional[float] = 1.0, frame_rate: Optional[int] = 1):
        if num_episodes is None:
            num_episodes = len(self._env._env.episodes)
        else:
            assert num_episodes <= len(self._env._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        writer = TensorboardWriter('tb_benchmark/', flush_secs=30) # flush_specs from base_trainer.py

        count_episodes = 0
        print("number of episodes: " + str(num_episodes))
        while count_episodes < num_episodes:
            print("working on episode " + str(count_episodes))
            observations_per_episode = []
            agent.reset()
            observations_per_action = self._env._env.reset()
            # initialize physic-enabled sim env. Do this for every
            # episode, since sometimes assets get deallocated
            if self._enable_physics:
                self._env._env.disable_physics()
                self._env._env.enable_physics()
            
            frame_counter = 0
            # act until one episode is over
            while not self._env._env.episode_over:
                action = agent.act(observations_per_action)
                observations_per_action = reward_per_action = done_per_action = info_per_action = None
                if (self._enable_physics is False):
                    (observations_per_action, 
                    reward_per_action, 
                    done_per_action, 
                    info_per_action)  = self._env.step(action)
                else:
                    # step with physics. For now we use hard-coded time step of 1/60 secs
                    # (used in the rigid object tutorial in Habitat Sim)
                    (observations_per_action, 
                    reward_per_action, 
                    done_per_action, 
                    info_per_action) = self._env.step_physics(
                        action, time_step=1.0/60.0, control_period=control_period)
                # generate an output image for the action. The image includes observations
                # and a top-down map showing the agent's state in the environment
                # we use frame_rate (num. of frames per action) to reduce computational overhead
                if frame_counter % frame_rate == 0:
                    out_im_per_action = observations_to_image(observations_per_action, info_per_action)
                    observations_per_episode.append(out_im_per_action)
                frame_counter = frame_counter + 1
            
            # episode ended
            # get per-episode metrics. for now we only extract
            # distance-to-goal, success, spl
            metrics = self._env._env.get_metrics()
            per_ep_metrics = {k: metrics[k] for k in ['distance_to_goal', 'success', 'spl']}
            # print distance_to_goal, success and spl
            for k, v in per_ep_metrics.items():
                print(f'{k},{v}')
            # calculate aggregated distance_to_goal, success and spl
            for m, v in per_ep_metrics.items():
                agg_metrics[m] += v
            count_episodes += 1
            # generate video
            generate_video(
                video_option=["disk", "tensorboard"],
                video_dir='video_benchmark_dir',
                images=observations_per_episode,
                episode_id=count_episodes-1,
                checkpoint_idx=0,
                metrics=per_ep_metrics,
                tb_writer=writer,
            )
            
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics

    def evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None, frame_rate: Optional[int] = 1, control_period: Optional[float] = 1.0
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes, control_period, frame_rate)
