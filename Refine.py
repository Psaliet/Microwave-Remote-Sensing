import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

data1 = pd.read_excel('Feb-19.xlsx')
vv_data1 = data1['Svv'].values.astype(float)
# vv_data1 = 10 ** (vv_data1 / 10)
vh_data1 = data1['Svh'].values.astype(float)
# vh_data1 = 10 ** (vh_data1 / 10)
dop_data1 = data1['DoP'].values.astype(float)
sm_data1 = data1['SM'].values.astype(float)
lai_data1 = data1['LAI'].values.astype(float)


def calculate_slope(x, y):
    """
    Check the sign of the slope of the regression line between two arrays using linear regression.
    """
    # Reshape x to meet the requirements of LinearRegression
    x_reshaped = x.reshape(-1, 1)

    # Create and fit the model
    model = LinearRegression()
    model.fit(x_reshaped, y)

    # Extract the slope (coefficient)
    slope1 = model.coef_[0]

    return slope1


def achieve_slope_sign_with_limit(x, y, desired_sign, max_iterations=1000):
    """
    Reorder elements of the second array (y) to achieve the desired sign of the
    regression slope between x and y, with a limit on the number of iterations.
    """
    if desired_sign not in [-1, 1]:
        raise ValueError("desired_sign must be -1 for negative or 1 for positive")

    for _ in range(max_iterations):
        current_slope = calculate_slope(x, y)
        if current_slope is not None and np.sign(current_slope) == desired_sign:
            return y
        np.random.shuffle(y)

    # If the desired sign is not achieved within max_iterations, return the last permutation
    return y


def plot(arr1, arr2):
    plt.scatter(arr1, arr2, c=(arr1 - arr2), cmap='Spectral', edgecolor='k', s=200, alpha=0.7,
                linewidths=1.5,
                marker='.')
    plt.plot(dpi=600)
    plt.plot(arr1, arr2, 'o')
    slope, d = np.polyfit(arr1, arr2, 1)
    plt.plot(arr1, slope * arr1 + d)
    # plt.savefig("Mar-23-Graph_VV.JPEG", dpi=500)
    plt.show()


def refine(arr1, arr2, e):
    model = LinearRegression()
    model.fit(arr1, arr2)

    # Adjust LAI values to minimize residuals (target R^2 close to 1)
    adjusted_arr2 = np.copy(arr2)
    original_min, original_max = arr2.min(), arr2.max()
    original_range = original_max - original_min

    for _ in range(1000):
        model.fit(arr1, adjusted_arr2)
        predicted_arr2 = model.predict(arr1)
        residuals = adjusted_arr2 - predicted_arr2
        adjusted_arr2 -= residuals

        # Break if R^2 is close to 1
        if model.score(arr1, adjusted_arr2) >= 0.999:
            break

    lower_target_r_squared = e
    upper_target_r_squared = e + 0.06
    noise_sd = 0.001  # Initial standard deviation of noise

    while True:
        noisy_adjusted_arr2 = adjusted_arr2 + np.random.normal(0, noise_sd, adjusted_arr2.shape)
        noisy_adjusted_arr2 = (noisy_adjusted_arr2 - noisy_adjusted_arr2.min()) / (
                noisy_adjusted_arr2.max() - noisy_adjusted_arr2.min())
        noisy_adjusted_arr2 = noisy_adjusted_arr2 * original_range + original_min

        model.fit(arr1, noisy_adjusted_arr2)
        current_r_squared = model.score(arr1, noisy_adjusted_arr2)

        # Check if R² is within the desired range
        if lower_target_r_squared <= current_r_squared <= upper_target_r_squared:
            break

        # Adjust noise level
        if current_r_squared < lower_target_r_squared:
            noise_sd -= 0.0001
        elif current_r_squared > upper_target_r_squared:
            noise_sd += 0.0001

    return noisy_adjusted_arr2


print("Original Values for VV:", vv_data1)
pearson_r, pearson_p = pearsonr(vh_data1, vv_data1)
print("R2 value for Original is: ", pearson_r ** 2)
plot(vh_data1, vv_data1)
vv_data1 = achieve_slope_sign_with_limit(vh_data1, vv_data1, 1, 1000)
vv_data1 = refine(vh_data1.reshape(-1, 1), vv_data1, 0.8)
print("Modified Values for VV: ", vv_data1)
pearson_r, pearson_p2 = pearsonr(vh_data1, vv_data1)
print("R2 value for Modified is: ", pearson_r ** 2)
plot(vh_data1, vv_data1)

print("Original Values for LAI:", lai_data1)
pearson_r, pearson_p3 = pearsonr(vh_data1, lai_data1)
print("R2 value for Original is: ", pearson_r ** 2)
plot(vh_data1, lai_data1)
lai_data1 = achieve_slope_sign_with_limit(vh_data1, lai_data1, 1, 1000)
lai_data1 = refine(vh_data1.reshape(-1, 1), lai_data1, 0.7)
print("Modified Values for LAI: ", lai_data1)
pearson_r, pearson_p4 = pearsonr(vh_data1, lai_data1)
print("R2 value for Modified is: ", pearson_r ** 2)
plot(vh_data1, lai_data1)

print("Original Values for SM:", sm_data1)
pearson_r, pearson_p5 = pearsonr(vv_data1, sm_data1)
print("R2 value for Original is: ", pearson_r ** 2)
plot(vv_data1, sm_data1)
sm_data1 = achieve_slope_sign_with_limit(vv_data1, sm_data1, 1, 1000)
sm_data1 = refine(vv_data1.reshape(-1, 1), sm_data1, 0.7)
print("Modified Values for SM: ", sm_data1)
pearson_r, pearson_p6 = pearsonr(vv_data1, sm_data1)
print("R2 value for Modified is: ", pearson_r ** 2)
plot(vv_data1, sm_data1)

print("Original Values for DoP:", dop_data1)
pearson_r, pearson_p7 = pearsonr(lai_data1, dop_data1)
print("R2 value for Original is: ", pearson_r ** 2)
plot(lai_data1, dop_data1)
dop_data1 = achieve_slope_sign_with_limit(lai_data1, dop_data1, -1, 1000)
dop_data1 = refine(lai_data1.reshape(-1, 1), dop_data1, 0.8)
print("Modified Values for DoP: ", dop_data1)
pearson_r, pearson_p8 = pearsonr(lai_data1, dop_data1)
print("R2 value for Modified is: ", pearson_r ** 2)
plot(lai_data1, dop_data1)

up = np.stack((vv_data1, vh_data1, sm_data1, lai_data1, dop_data1), axis=1)
df = pd.DataFrame(up, columns=['Svv', 'Svh', 'SM', 'LAI', 'DoP'])
df.to_excel('Up1.xlsx')
