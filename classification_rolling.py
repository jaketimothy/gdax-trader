import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

def run(column_set='raw', scale=True):
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

        y = (df['next point norm close'] - df['norm close 143']) > 0

        return (X, y)
    X, y = Xandy(train)
    print(len(X), len(X.columns))

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

    # sgd - hinge
    start_time = timeit.default_timer()
    sgd_hinge = SGDClassifier(loss='hinge', class_weight='balanced', random_state=123, n_jobs=-1).fit(X, y)
    print('fit sgd_hinge: %f' % (timeit.default_timer() - start_time))

    # sgd - log
    start_time = timeit.default_timer()
    sgd_log = SGDClassifier(loss='log', class_weight='balanced', random_state=123, n_jobs=-1).fit(X, y)
    print('fit sgd_log: %f' % (timeit.default_timer() - start_time))

    # sgd - huber
    start_time = timeit.default_timer()
    sgd_huber = SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=123, n_jobs=-1).fit(X, y)
    print('fit sgd_huber: %f' % (timeit.default_timer() - start_time))

    # sgd - perceptron
    start_time = timeit.default_timer()
    sgd_perceptron = SGDClassifier(loss='perceptron', class_weight='balanced', random_state=123, n_jobs=-1).fit(X, y)
    print('fit sgd_perceptron: %f' % (timeit.default_timer() - start_time))

    # etr
    start_time = timeit.default_timer()
    etc = ExtraTreesClassifier(class_weight='balanced', random_state=123, n_jobs=-1).fit(X, y)
    print('fit etc: %f' % (timeit.default_timer() - start_time))

    # gbr
    start_time = timeit.default_timer()
    gbc = GradientBoostingClassifier(random_state=123).fit(X, y)
    print('fit gbc: %f' % (timeit.default_timer() - start_time))

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

    # start_time = timeit.default_timer()
    # y_svr_linear = svr_linear.predict(X)
    # print('predict svr_linear: %f' % (timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # y_krr_linear = krr_linear.predict(X)
    # print('predict krr_linear: %f' % (timeit.default_timer() - start_time))

    # y_gpr = gpr.predict(X)

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
    y_sgd_hinge = sgd_hinge.predict(X)
    print('predict sgd_hinge: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    y_sgd_log = sgd_log.predict(X)
    print('predict sgd_log: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    y_sgd_huber = sgd_huber.predict(X)
    print('predict sgd_huber: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    y_sgd_perceptron = sgd_perceptron.predict(X)
    print('predict sgd_perceptron: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    y_etc = etc.predict(X)
    print('predict etc: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    y_gbc = gbc.predict(X)
    print('predict gbc: %f' % (timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # y_abr_trees = abr_trees.predict(X)
    # print('predict abr_trees: %f' % (timeit.default_timer() - start_time))
    #
    # start_time = timeit.default_timer()
    # y_abr_br = abr_br.predict(X)
    # print('predict abr_br: %f' % (timeit.default_timer() - start_time))

    # y_mlpr = mlpr.predict(X)

    # voting ensemble
    # a = [
    #     # y_svr_linear,
    #     # y_krr_linear,
    #     y_ridge,
    #     y_br,
    #     # y_knr,
    #     y_etr
    #     # y_gbr
    #     # y_abr_trees +
    #     # y_abr_br
    # ]
    # y_avg = np.mean(a, axis=0)
    # y_median = np.median(a, axis=0)

    print('')
    print('SGD (hinge) %0.4f' % accuracy_score(y, y_sgd_hinge))
    print(confusion_matrix(y, y_sgd_hinge))

    print('')
    print('SGD (log) %0.4f' % accuracy_score(y, y_sgd_log))
    print(confusion_matrix(y, y_sgd_log))

    print('')
    print('SGD (huber) %0.4f' % accuracy_score(y, y_sgd_huber))
    print(confusion_matrix(y, y_sgd_huber))

    print('')
    print('SGD (perceptron) %0.4f' % accuracy_score(y, y_sgd_perceptron))
    print(confusion_matrix(y, y_sgd_perceptron))

    print('')
    print('ETC %0.4f' % accuracy_score(y, y_etc))
    print(confusion_matrix(y, y_etc))

    print('')
    print('GBC %0.4f' % accuracy_score(y, y_gbc))
    print(confusion_matrix(y, y_gbc))
