import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from hyperopt import fmin, tpe, hp, Trials

def run(column_set='raw', train_delta=True, scale=True, evals=40):
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
    X_test, y_test = Xandy(test)
    print(len(X_test), len(X_test.columns))
    y_test = y_test - test['norm close 143'].values

    # fit scaler
    if scale:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        X_test = scaler.transform(X_test)

    # fmin search
    space = {
        'alpha': hp.uniform('alpha', 0.1, 1e-6),
        'first_layer_size': hp.quniform('first_layer_size', 5, 100, 5),
        'layer_decay': hp.uniform('layer_decay', 0.15, 0.5),
    }

    def make_layers(init, rate):
        layers = []
        layer_size = init
        while layer_size > 1:
            layers.append(layer_size)
            layer_size = int(layer_size - init * rate)
        return tuple(layers)

    def f(x):
        layers = make_layers(int(x['first_layer_size']), x['layer_decay'])

        start_time = timeit.default_timer()
        mlpr = MLPRegressor(
            hidden_layer_sizes=layers,
            activation="relu",
            solver='adam',
            alpha=x['alpha'],
            learning_rate="constant",
            max_iter=200,
            random_state=123
        ).fit(X, y)
        score = mlpr.score(X_test, y_test)
        print(layers)
        print('score: %f; time: %f' % (score, timeit.default_timer() - start_time))

        return -score

    trials = Trials()
    best = fmin(fn=f, space=space, trials=trials, algo=tpe.suggest, max_evals=evals)
    print(best)

    layers = make_layers(int(best['first_layer_size']), best['layer_decay'])
    print(layers)

    mlpr = MLPRegressor(
        hidden_layer_sizes=layers,
        activation="relu",
        solver='adam',
        alpha=best['alpha'],
        learning_rate="constant",
        max_iter=200,
        random_state=123
    ).fit(X, y)
    y_mlpr = mlpr.predict(X_test)

    x_plt = range(len(X_test))
    plt.figure(figsize=(20,10))
    plt.scatter(x_plt, y_test, label='data', s=4, color='k')
    plt.plot(x_plt, y_mlpr, label='MLPR %f' % r2_score(y_test, y_mlpr))
    plt.legend()
    plt.show()

    f, ax = plt.subplots(1)
    xs = [t['misc']['vals']['alpha'] for t in trials.trials]
    ys = [t['result']['loss'] for t in trials.trials]
    ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('$R^2$ $vs$ $alpha$ ', fontsize=18)
    ax.set_xlabel('$alpha$', fontsize=16)
    ax.set_ylabel('$R^2$', fontsize=16)
