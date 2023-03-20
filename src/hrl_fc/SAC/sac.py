import matplotlib.pyplot as plt
import numpy as np
from sac_torch import Agent

from envs.lti_citation.lti_env import LTIEnv


if __name__ == "__main__":
    env = LTIEnv()  # gym.make('MountainCarContinuous-v0')
    agent = Agent(
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
    )
    n_games = 250

    filename = "pendulum.png"

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")

    for i in range(3):
        observation = env.reset()
        done = False
        score = 0
        counter = 0
        observations = []
        while not done and counter < 200:
            counter += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, int(done))

            if not load_checkpoint:
                a = agent.learn()
            observation = observation_
            observations.append(observation[0])

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print(
            f"episode {i} score {score} avg score {avg_score} best score {best_score}"
        )
        plt.plot(observations)
        plt.show()
