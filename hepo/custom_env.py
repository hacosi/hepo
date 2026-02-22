from typing import Callable, Optional, Dict, Any
import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


class VecEnvRewardSplitWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, predicate: Callable = None):
        super().__init__(venv)

        def default_predicate(obs, reward, done, info):
            return True

        self.predicate = predicate if predicate is not None else default_predicate

    # task predicate should be a function that takes in the state, reward, done, info and returns the (task_reward, heuristic_reward)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        """
        Puts task and heuristic rewards into info dict
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        for i in range(self.num_envs):
            _obs, reward, done, info = obs[i], rewards[i], dones[i], infos[i]
            task_reward, heuristic_reward = self.predicate(
                _obs, reward, done, info)
            info["task_reward"] = float(task_reward)
            info["heuristic_reward"] = float(heuristic_reward)

        return obs, rewards, dones, infos
