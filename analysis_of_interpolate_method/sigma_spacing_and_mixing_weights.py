import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def gaussian(x, sigma, center=0):
    gauss = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return gauss

def sigma_space(spacing):
    sigma_min = 1
    sigma_max = spacing

    def gaussian_mix(x, w1, w2):
        #Return a linear combination of two Gaussians with weights
        return (w1 * gaussian(x, sigma=sigma_min)
                + w2 * gaussian(x, sigma=sigma_max))

    sigma_values = np.linspace(sigma_min, sigma_max, 101)
    x = np.linspace(-10, 50, 101)

    rms_save = np.zeros(len(sigma_values))
    lower_mix = np.zeros(len(sigma_values))
    upper_mix = np.zeros(len(sigma_values))

    for i, sigma in enumerate(sigma_values):
        actual_gaussian = gaussian(x, sigma)
        (mixl, mixu), _ = curve_fit(gaussian_mix, x, ydata=actual_gaussian, p0=[0.5, 0.5])

        est_gaussian = gaussian_mix(x, mixl, mixu)
        rms_save[i] = np.sqrt(np.mean((actual_gaussian-est_gaussian)**2))

        lower_mix[i] = mixl
        upper_mix[i] = mixu

    rms_mean = np.mean(rms_save)

    # plt.figure()
    # plt.plot(sigma_values, lower_mix)
    # plt.plot(sigma_values, upper_mix)
    # plt.title('Sigma Value vs Upper and Lower Mixing Weights')

    return rms_mean, lower_mix, upper_mix

rms_mean, lower_mix, upper_mix = sigma_space(np.sqrt(2))

"""
Investigate which degree polynomial provides best fit to mixing weights
"""

X = np.linspace(1, np.sqrt(2), 101).reshape(101, 1)
y = lower_mix

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rmses = []
degrees = np.arange(1, 10)
min_rmse, min_deg = 1e10, 0

for deg in degrees:

    # Train features
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly_train = poly_features.fit_transform(x_train)

    # Linear regression
    poly_reg = LinearRegression()
    poly_reg.fit(x_poly_train, y_train)

    # Compare with test data
    x_poly_test = poly_features.fit_transform(x_test)
    poly_predict = poly_reg.predict(x_poly_test)
    poly_mse = mean_squared_error(y_test, poly_predict)
    poly_rmse = np.sqrt(poly_mse)
    rmses.append(poly_rmse)
    
    # Cross-validation of degree
    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        min_deg = deg
        
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(degrees, rmses)
ax.set_yscale('log')
ax.set_xlabel('Degree')
ax.set_ylabel('RMSE')

"""
Over a range of sigma_spacing values, determine relationship between
spacing and rmse & spacing and number of kernels required
"""
 
spaces = np.arange(1.1,1.5,0.01)
rms_space = []
n_kernels = []

for spacing in spaces:
    rms_mean, _, _= sigma_space(spacing)
    rms_space.append(rms_mean)

    # number of kernels required for set sigma range
    n_k = int(np.ceil(np.log(40/1)/np.log(spacing)))

    n_kernels.append(n_k)

rms_space = np.array(rms_space)
n_kernels = np.array(n_kernels)

plt.figure()
plt.plot(spaces, rms_space)
plt.xlabel('Sigma Spacing')
plt.ylabel('RMS Error')

plt.figure()
plt.loglog(spaces, n_kernels)
plt.xlabel('Sigma Spacing')
plt.ylabel('Number of Kernels Required')

plt.figure()
plt.title('RMS Error vs Number of Kernels required')
plt.plot(rms_space, n_kernels)
plt.xlabel('RMS Error')
plt.ylabel('Number of Kernels Required')

plt.show()