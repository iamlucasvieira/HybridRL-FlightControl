from models.aircraft_environment import AircraftEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

TRAIN = False
env = AircraftEnv('citation.yaml', dt=0.01, T=100)

if TRAIN:
    # check_env(env)
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=150_000, log_interval=100)
    model.save("sac_pendulum")
else:
    model = SAC.load("sac_pendulum")

obs = env.reset()
print("Done")
import matplotlib.pyplot as plt
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
