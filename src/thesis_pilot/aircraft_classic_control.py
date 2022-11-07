import matplotlib.pyplot as plt
import control as ct
import numpy as np

from models.aircraft_model import Aircraft

acft = Aircraft('citation.yaml')
lst = []
for i in np.arange(0, 150, 0.01):
    if i == 0:
        u = -0.005
    y = acft.response(u)
    lst.append(y[2, 0])

plt.plot(lst)
plt.show()
print(acft.ss)
t, y = ct.forced_response(acft.ss, X0=[0,0,0,0], T=np.arange(0, 150, 0.01), U=-0.005)
# plt.plot(t, y[0, :].flatten())
# plt.plot(t, y[1, :].flatten())
plt.plot(t, y[2, :].flatten())
# plt.plot(t, y[3, :].flatten())

plt.show()
