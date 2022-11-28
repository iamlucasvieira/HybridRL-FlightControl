from models.aircraft_environment import AircraftEnv
from stable_baselines3 import SAC
# from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

TRAIN = True
env = AircraftEnv('citation.yaml', dt=0.01, T=100)

if TRAIN:
    # check_env(env)
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(10000), log_interval=100)
    model.save("citation_sac")
else:
    model = SAC.load("citation_sac")

obs = env.reset()
print("Done")

for _ in range(10):
    states = []
    for i in range(20000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        states.append(obs[1])
        # print(reward)
        if done:
            print(f"finished at {i}")
            obs = env.reset()
            break
    plt.plot(states)
    plt.show()
