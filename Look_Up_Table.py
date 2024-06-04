import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

df2 = pd.read_excel('Up.xlsx')
data2 = df2.values

svv_i = data2[0:, 1].astype(float)
svh_i = data2[0:, 2].astype(float)
dprvi_i = data2[0:, 3].astype(float)
lai_i = data2[0:, 4].astype(float)
m_i = data2[0:, 5].astype(float)
sm_i = data2[0:, 6].astype(float)


def predict_backscatter(variables, lai, m, sm):
    a = variables[0]
    b = variables[1]
    e = variables[2]
    g = variables[3]

    c = 3 * 10 ** 8
    f = 4.75 * 10 ** 9
    k = (2 * np.pi * f) / c
    s = 0.0155
    tau_sq = np.exp(b * lai)
    e_s = 114 * (sm ** 2) + 13.3 * sm + 2.5
    theta = 2 * np.pi / 9

    surf_vh = (1 - m) * 0.1 * (sm ** 0.7) * (np.cos(theta) ** 2.2) * (1 - np.exp(-0.32 * (k * s) ** 1.8))
    q = 0.1 * (0.13 + np.sin(1.3 * theta)) ** 1.2 * (1 - np.exp(-0.9 * (k * s) ** 0.8))
    surf_vv = surf_vh / q

    inter_1 = np.sin(theta) ** 2 - e_s * (1 + (np.sin(theta) ** 2))
    inter_2 = np.sqrt(e_s - (np.sin(theta) ** 2))
    inter = np.absolute(((e_s - 1) * inter_1) / ((np.cos(theta) + inter_2) ** 2)) ** 2
    s_inter = g * lai * tau_sq * inter

    s_veg = m * a * (lai ** (e + 1))
    model = s_veg + tau_sq * surf_vv + s_inter
    model = 10 * np.log10(model)
    return model


def residual(variables, lai, m, sm, svv):
    y = predict_backscatter(variables, lai, m, sm)
    return svv - y


variables = np.array([0, 0, 0, 0])

svv = svv_i[:40]
svh = svh_i[:40]
dprvi = dprvi_i[:40]
lai = lai_i[:40]
m = m_i[:40]
sm = sm_i[:40]

svv_t = svv_i[40:]
svh_t = svh_i[40:]
dprvi_t = dprvi_i[40:]
lai_t = lai_i[40:]
m_t = m_i[40:]
sm_t = sm_i[40:]

out = least_squares(residual, x0=variables, args=(lai, m, sm, svv), method='trf')
test = residual(out.x, lai_t, m_t, sm_t, svv_t)
model = svv_t - test
print("Optimised Parameters are: {}".format(out.x))

print("model is {}".format(model))
print("svv is {}".format(svv_t))

rms = 0
for i in range(np.size(test)):
    rms = rms + (test[i] ** 2)
rms = rms / (np.size(test))
rmse = np.sqrt(rms)
print("RMSE is: {}".format(rmse))

r2 = r2_score(svv_t, model)
print("The coefficient of determination is: ", r2)


lai_iter = np.linspace(np.min(lai_i), np.max(lai_i), 100)
print("Length of iterative lai is: ", len(lai_iter))
sm_iter = np.linspace(np.min(sm_i), np.max(sm_i), 100)
print("Length of iterative sm is: ", len(sm_iter))
m_iter = np.linspace(np.min(m_i), np.max(m_i), 100)
print("Length of iterative m: ", len(m_iter))
svv_iter = np.zeros((len(lai_iter), len(m_iter), len(sm_iter)))

for i, lai_value in enumerate(lai_iter):
    for j, m_value in enumerate(m_iter):
        for k, sm_value in enumerate(sm_iter):
            svv_iter[i, j, k] = predict_backscatter(out.x, lai_value, m_value, sm_value)

correspond_lai = np.zeros(len(lai_t))
correspond_svv = np.zeros(len(svv_t))
correspond_m = np.zeros(len(m_t))
correspond_sm = np.zeros(len(sm_t))

for i in range(len(svv_t)):
    chosen_value = svv_t[i]
    flattened_backscatter = svv_iter.flatten()
    closest_index = np.abs(flattened_backscatter - chosen_value).argmin()
    index = np.unravel_index(closest_index, svv_iter.shape)
    correspond_lai[i] = lai_iter[index[0]]
    correspond_sm[i] = sm_iter[index[2]]
    correspond_svv[i] = flattened_backscatter[closest_index]

print("The value of Svv is: ", svv_t)
print("Predicted value of Svv is: ", correspond_svv)

print("The value of lai is: ", lai_t)
print("Predicted value of lai is: ", correspond_lai)

print("The value of sm is: ", sm_t)
print("The predicted value of sm is:", correspond_sm)

r2_lai = r2_score(lai_t, correspond_lai)
print("Coefficient of Determination for LAI is: ", r2_lai)
r2_sm = r2_score(sm_t, correspond_sm)
print("Coefficient of Determination for SM is: ", r2_sm)