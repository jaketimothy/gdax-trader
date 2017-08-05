import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression, LinearRegression, BayesianRidge, RidgeCV, PassiveAggressiveRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

print(sklearn.__version__)

train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')
print(len(train), len(train.columns))
print(len(test))

X = train.drop('next day norm close', axis=1)
y = train['next day norm close']

# fit scaler
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# svr
# svr = SVR().fit(X, y)
# y_svr = svr.predict(X)

# gpr
# gpr = GaussianProcessRegressor().fit(X, y)
# y_gpr = gpr.predict(X)

# ridge
# ridge = RidgeCV().fit(X, y)
# y_ridge = ridge.predict(X)

# br
br = BayesianRidge().fit(X, y)
y_br = br.predict(X)

# ard
# ard = ARDRegression().fit(X, y)
# y_ard = ard.predict(X)

# lr
# lr = LinearRegression().fit(X, y)
# y_lr = lr.predict(X)

# par
# par = PassiveAggressiveRegressor(random_state=123).fit(X, y)
# y_par = par.predict(X)

# knr
# knr = KNeighborsRegressor().fit(X, y)
# y_knr = knr.predict(X)

# dtr
# dtr = DecisionTreeRegressor(random_state=123).fit(X, y)
# y_dtr = dtr.predict(X)

# etr
etr = ExtraTreesRegressor(random_state=123).fit(X, y)
y_etr = etr.predict(X)

# gbr
gbr = GradientBoostingRegressor(random_state=123).fit(X, y)
y_gbr = gbr.predict(X)

x_plt = range(len(X))
plt.scatter(x_plt, y, label='data', s=4, color='k')
# plt.plot(x_plt, y_svr, label='SVR model')
# plt.plot(x_plt, y_gpr, label='GPR model')
# plt.plot(x_plt, y_ridge, label='Ridge model')
plt.plot(x_plt, y_br, label='BR model')
# plt.plot(x_plt, y_ard, label='ARD model')
# plt.plot(x_plt, y_lr, label='LR model')
# plt.plot(x_plt, y_par, label='PAR model')
# plt.plot(x_plt, y_knr, label='KNR model')
# plt.plot(x_plt, y_dtr, label='DTR model')
plt.plot(x_plt, y_etr, label='ETR model')
plt.plot(x_plt, y_gbr, label='GBR model')
plt.legend()
plt.show()

# predict on test
X = scaler.transform(test.drop('next day norm close', axis=1))
y = test['next day norm close']
# y_svr = svr.predict(X)
# y_gpr = gpr.predict(X)
# y_ridge = ridge.predict(X)
y_br = br.predict(X)
# y_ard = ard.predict(X)
# y_lr = lr.predict(X)
# y_par = par.predict(X)
# y_knr = knr.predict(X)
# y_dtr = dtr.predict(X)
y_etr = etr.predict(X)
y_gbr = gbr.predict(X)

x_plt = range(len(X))
plt.scatter(x_plt, y, label='data', s=4, color='k')
# plt.plot(x_plt, y_svr, label='SVR %f' % r2_score(y, y_svr))
# plt.plot(x_plt, y_gpr, label='GPR %f' % r2_score(y, y_gpr))
# plt.plot(x_plt, y_ridge, label='Ridge %f' % r2_score(y, y_ridge))
plt.plot(x_plt, y_br, label='BR %f' % r2_score(y, y_br))
# plt.plot(x_plt, y_ard, label='ARD %f' % r2_score(y, y_ard))
# plt.plot(x_plt, y_lr, label='LR %f' % r2_score(y, y_lr))
# plt.plot(x_plt, y_par, label='PAR %f' % r2_score(y, y_par))
# plt.plot(x_plt, y_knr, label='KNR %f' % r2_score(y, y_knr))
# plt.plot(x_plt, y_dtr, label='DTR %f' % r2_score(y, y_dtr))
plt.plot(x_plt, y_etr, label='ETR %f' % r2_score(y, y_etr))
plt.plot(x_plt, y_gbr, label='GBR %f' % r2_score(y, y_gbr))
plt.legend()
plt.show()
