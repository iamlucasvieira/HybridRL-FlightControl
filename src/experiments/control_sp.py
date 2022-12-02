from models.aircraft_model import Aircraft
import matplotlib.pyplot as plt

acft = Aircraft(configuration='sp')

aoa_list = []
q_list = []
for _ in range(1000):
    de = -0.005
    states = acft.response(de)
    aoa_list.append(states[0, 0])
    q_list.append(states[1, 0])

plt.plot(aoa_list)
plt.show()
plt.plot(q_list)
plt.show()