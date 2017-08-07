import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression, LinearRegression, BayesianRidge, RidgeCV, PassiveAggressiveRegressor, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

def run(column_set='raw', train_delta=False, scale=True):
    print(sklearn.__version__)

    train = pd.read_pickle('train_rolling.pkl')
    test = pd.read_pickle('test_rolling.pkl')

    # X, y train
    def Xandy(df):
        if column_set == 'raw':
            X = df.drop(['next point norm close', 'norm close integral'], axis=1)
        elif column_set == 'pid':
            # last price, integral price, last relative volume
            X = df[['norm close 143', 'norm close integral', 'norm volume 143']]
            # first derivative
            X['norm close 143 d'] = df['norm close 143'] - df['norm close 142']
            # second derivative
            X['norm close 143 dd'] = ((df['norm close 143'] - df['norm close 142']) -
                (df['norm close 142'] - df['norm close 141']))
            # first derivative relative volume
            X['norm volume 143 d'] = df['norm volume 143'] - df['norm volume 142']
            # second derivative relative volume
            X['norm volume 143 dd'] = ((df['norm volume 143'] - df['norm volume 142']) -
                (df['norm volume 142'] - df['norm volume 141']))
        else:
            raise ValueError()

        return (X, df['next point norm close'])
    X, y = Xandy(train)
    print(len(X), len(X.columns))
    if train_delta:
        y = y - train['norm close 143']

    # fit scaler
    if scale:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    # svr - memory error
    # start_time = timeit.default_timer()
    # svr_linear = SVR(kernel='linear', cache_size=1000).fit(X, y)
    # print('fit svr_linear: %f' % (timeit.default_timer() - start_time))

    # krr (linear)
    # start_time = timeit.default_timer()
    # krr_linear = KernelRidge(kernel='linear').fit(X, y)
    # print('fit krr_linear: %f' % (timeit.default_timer() - start_time))

    # gpr
    # gpr = GaussianProcessRegressor().fit(X, y)

    # sgd - squared_loss
    start_time = timeit.default_timer()
    sgd = SGDRegressor(loss="squared_loss", random_state=123).fit(X, y)
    print('fit sgd: %f' % (timeit.default_timer() - start_time))

    # sgd - huber
    start_time = timeit.default_timer()
    sgd_huber = SGDRegressor(loss="huber", random_state=123).fit(X, y)
    print('fit sgd_huber: %f' % (timeit.default_timer() - start_time))

    # sgd - epsilon_insensitive - poor performer
    # start_time = timeit.default_timer()
    # sgd_epsi = SGDRegressor(loss="epsilon_insensitive", random_state=123).fit(X, y)
    # print('fit sgd_epsi: %f' % (timeit.default_timer() - start_time))

    # ridge
    start_time = timeit.default_timer()
    ridge = RidgeCV().fit(X, y)
    print('fit ridge: %f' % (timeit.default_timer() - start_time))

    # br
    start_time = timeit.default_timer()
    br = BayesianRidge().fit(X, y)
    print('fit br: %f' % (timeit.default_timer() - start_time))

    # ard - memory error, not scalable
    # start_time = timeit.default_timer()
    # ard = ARDRegression().fit(X, y)
    # print('fit ard: %f' % (timeit.default_timer() - start_time))

    # par - poor performer on pid
    # start_time = timeit.default_timer()
    # par = PassiveAggressiveRegressor(random_state=123).fit(X, y)
    # print('fit par: %f' % (timeit.default_timer() - start_time))

    # knr - just ok performance
    # start_time = timeit.default_timer()
    # knr = KNeighborsRegressor(n_jobs=-1).fit(X, y)
    # print('fit knr: %f' % (timeit.default_timer() - start_time))

    # etr
    start_time = timeit.default_timer()
    etr = ExtraTreesRegressor(random_state=123, n_jobs=-1).fit(X, y)
    print('fit etr: %f' % (timeit.default_timer() - start_time))

    # gbr
    # start_time = timeit.default_timer()
    # gbr = GradientBoostingRegressor(random_state=123).fit(X, y)
    # print('fit gbr: %f' % (timeit.default_timer() - start_time))

    # abr (trees)
    # start_time = timeit.default_timer()
    # abr_trees = AdaBoostRegressor(random_state=123).fit(X, y)
    # print('fit abr_trees: %f' % (timeit.default_timer() - start_time))
    #
    # # abr (br)
    # start_time = timeit.default_timer()
    # abr_br = AdaBoostRegressor(BayesianRidge(), random_state=123).fit(X, y)
    # print('fit abr_br: %f' % (timeit.default_timer() - start_time))

    # mlpr
    # start_time = timeit.default_timer()
    # mlpr = MLPRegressor(random_state=123).fit(X, y)

    # predict on test
    X, y = Xandy(test)
    print(len(X), len(X.columns))
    if scale:
        X = scaler.transform(X)
    y = y - test['norm close 143'].values

    # start_time = timeit.default_timer()
    # y_svr_linear = svr_linear.predict(X)
    # print('predict svr_linear: %f' % (timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # y_krr_linear = krr_linear.predict(X)
    # print('predict krr_linear: %f' % (timeit.default_timer() - start_time))

    # y_gpr = gpr.predict(X)

    start_time = timeit.default_timer()
    y_sgd = sgd.predict(X)
    print('predict sgd: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    y_sgd_huber = sgd_huber.predict(X)
    print('predict sgd_huber: %f' % (timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # y_sgd_epsi = sgd_epsi.predict(X)
    # print('predict sgd_epsi: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    y_ridge = ridge.predict(X)
    print('predict ridge: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    y_br = br.predict(X)
    print('predict br: %f' % (timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # y_ard = ard.predict(X)
    # print('predict ard: %f' % (timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # y_par = par.predict(X)
    # print('predict par: %f' % (timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # y_knr = knr.predict(X)
    # print('predict knr: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    y_etr = etr.predict(X)
    print('predict etr: %f' % (timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # y_gbr = gbr.predict(X)
    # print('predict gbr: %f' % (timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # y_abr_trees = abr_trees.predict(X)
    # print('predict abr_trees: %f' % (timeit.default_timer() - start_time))
    #
    # start_time = timeit.default_timer()
    # y_abr_br = abr_br.predict(X)
    # print('predict abr_br: %f' % (timeit.default_timer() - start_time))

    # y_mlpr = mlpr.predict(X)

    if not train_delta:
        # y_svr_linear = y_svr_linear - test['norm close 143'].values
        # y_krr_linear = y_krr_linear - test['norm close 143'].values
        # y_gpr = gpr.predict(X) - test['norm close 143'].values
        y_sgd = y_sgd - test['norm close 143'].values
        y_sgd_huber = y_sgd_huber - test['norm close 143'].values
        # y_sgd_epsi = y_sgd_epsi - test['norm close 143'].values
        y_ridge = y_ridge - test['norm close 143'].values
        y_br = y_br - test['norm close 143'].values
        # y_ard = y_ard - test['norm close 143'].values
        # y_par = y_par - test['norm close 143'].values
        # y_knr = y_knr - test['norm close 143'].values
        y_etr = y_etr - test['norm close 143'].values
        # y_gbr = y_gbr - test['norm close 143'].values
        # y_abr_trees = y_abr_trees - test['norm close 143'].values
        # y_abr_br = y_abr_br - test['norm close 143'].values
        # y_mlpr = mlpr.predict(X) - test['norm close 143'].values

    # average answer
    a = [
        # y_svr_linear,
        # y_krr_linear,
        y_sgd,
        y_sgd_huber,
        # y_sgd_epsi,
        y_ridge,
        y_br,
        # y_knr,
        y_etr
        # y_gbr
        # y_abr_trees +
        # y_abr_br
    ]
    y_avg = np.mean(a, axis=0)
    y_median = np.median(a, axis=0)

    x_plt = range(len(X))
    plt.figure(figsize=(20,10))
    plt.scatter(x_plt, y, label='data', s=4, color='k')
    # plt.plot(x_plt, y_svr_linear, label='SVR (linear) %f' % r2_score(y, y_svr_linear))
    # plt.plot(x_plt, y_krr_linear, label='KRR (linear) %f' % r2_score(y, y_krr_linear))
    # plt.plot(x_plt, y_gpr, label='GPR %f' % r2_score(y, y_gpr))
    plt.plot(x_plt, y_sgd, label='SGD (squared) %f' % r2_score(y, y_sgd))
    plt.plot(x_plt, y_sgd_huber, label='SGD (huber) %f' % r2_score(y, y_sgd_huber))
    # plt.plot(x_plt, y_sgd_epsi, label='SGD (epsi) %f' % r2_score(y, y_sgd_epsi))
    plt.plot(x_plt, y_ridge, label='Ridge %f' % r2_score(y, y_ridge))
    plt.plot(x_plt, y_br, label='BR %f' % r2_score(y, y_br))
    # plt.plot(x_plt, y_ard, label='ARD %f' % r2_score(y, y_ard))
    # plt.plot(x_plt, y_par, label='PAR %f' % r2_score(y, y_par))
    # plt.plot(x_plt, y_knr, label='KNR %f' % r2_score(y, y_knr))
    plt.plot(x_plt, y_etr, label='ETR %f' % r2_score(y, y_etr))
    # plt.plot(x_plt, y_gbr, label='GBR %f' % r2_score(y, y_gbr))
    # plt.plot(x_plt, y_abr_trees, label='ABR (trees) %f' % r2_score(y, y_abr_trees))
    # plt.plot(x_plt, y_abr_br, label='ABR (BR) %f' % r2_score(y, y_abr_br))
    # plt.plot(x_plt, y_mlpr, label='MLPR %f' % r2_score(y, y_mlpr))
    plt.plot(x_plt, y_avg, label='Mean %f' % r2_score(y, y_avg))
    plt.plot(x_plt, y_median, label='Median %f' % r2_score(y, y_median))
    plt.legend()
    plt.show()
