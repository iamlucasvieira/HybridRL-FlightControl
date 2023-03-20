import gym
from stable_baselines3 import SAC


env = gym.make("Pendulum-v1")
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20_000)

obs = env.reset()

for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()

    if done:
        break
