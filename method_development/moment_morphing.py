# moment morphing
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, sigma=2, center=0):
    g = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return g

sigma_min = 1
sigma_max = 2
sigma_interp = 1.5

x = np.linspace(-6,6,100)

g1 = gaussian(x, sigma=sigma_min)
g2 = gaussian(x, sigma=sigma_max)

wt2 = (sigma_interp-sigma_min)/(sigma_max-sigma_min)
wt1 = 1-wt2

lin_interp = wt1*g1 + wt2*g2

actual = gaussian(x, sigma_interp)

plt.figure()
plt.plot(x, g1)
plt.plot(x, g2)
plt.plot(x, actual)
plt.plot(x, lin_interp,'--')
plt.plot(x, wt1*g1)
plt.plot(x, wt2*g2)

mu_dash = [wt1*0, wt2*0]
sigma_dash = [wt1*sigma_min, wt2*sigma_max]

x_dash = (x - mu_dash)

plt.show()