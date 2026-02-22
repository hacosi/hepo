from typing import Callable


def default_predicate():
    def task_predicate(obs, reward, done, info):
        return False

    return task_predicate


def LunarLander_predicate():
    def task_predicate(obs, reward, done, info):
        terminated = bool(info.get("terminal_observation") is not None)
        truncated = bool(info.get("TimeLimit.truncated"))
        if terminated and not truncated:
            if reward not in (-100.0, 100.0):
                print("WRONG")
                print("Info is ", info)
                print("Reward is ", reward)
        return terminated and not truncated

    return task_predicate


def BipedalWalker_predicate():
    def task_predicate(obs, reward, done, info):
        terminated = bool(info.get("terminal_observation") is not None)
        truncated = bool(info.get("TimeLimit.truncated"))
        if terminated and not truncated:
            if reward not in (-100.0, 300.0):
                print("WRONG")
                print("Info is ", info)
                print("Reward is ", reward)
        return terminated and not truncated and reward in (-100.0, 300.0)

    return task_predicate


def predicate_h1hand_sit_simple_v0():
    def predicate(obs, reward, done, info):
        heuristic_reward = (
            info["upright"]
            * info["sitting_posture"]
            * info["small_control"]
            * info["dont_move"]
        )
        task_reward = reward
        return task_reward, heuristic_reward

    return predicate
