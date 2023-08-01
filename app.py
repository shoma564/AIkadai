#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, flash, render_template, request, session
import mysql.connector, re
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, random
import numpy as np
from scipy.stats import norm
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from numpy.random import normal
from scipy.stats import norm, bayes_mvs
import time
import random

#app = Flask(__name__, static_folder='./templates/images')


app = Flask(__name__)

def reg1dim(xlist, ylist):
    n = len(xlist)
    a = ((np.dot(xlist, ylist)- ylist.sum() * xlist.sum()/n)/((xlist ** 2).sum() - xlist.sum()**2 / n))
    b = (ylist.sum() - a * xlist.sum())/n
    return a, b



@app.route("/")
def index():
    return render_template('index.html')


@app.route("/tyokusen")
def tyokusen():
    i = 0
    n = 100
    xlist = []
    ylist = []
    while i < n:
        x=random.uniform(-5,5)
        xlist.append(x)
        e=random.gauss(0,1)
        y=2 * x + 3 + e
        ylist.append(y)
        i += 1
    try:
        os.remove('/usr/src/app/static/figure01.jpg')
    except:
        pass

    xlist = np.array(xlist)
    ylist = np.array(ylist)

    a = ((xlist * ylist).mean() - (xlist.mean() * ylist.mean())) / ((xlist ** 2).mean() - xlist.mean() ** 2)
    b = -(a * xlist.mean()) + ylist.mean()

    z = a * xlist + b

    plt.figure()
    plt.scatter(xlist, ylist, color="pink")
    plt.plot(xlist, z, color="blue") 
    plt.savefig('/usr/src/app/static/figure01.jpg')
    return render_template('tyokusen.html', xlist=xlist, ylist=ylist)


@app.route("/kyokusen")
def kyokusen():
    x1 = []
    y2 = []

    try:
        os.remove('/usr/src/app/static/figure02.jpg')
    except:
        pass

    a1 = -2
    a2 = 1
    a3 = 20
    a4 = 1


    x2 = np.arange(-5, 5, 0.1)                                # 時間軸配列を作成
    noise = np.random.normal(loc=0, scale=20, size=len(x2)) # ガウシアンノイズを生成
    y2 = a1 * x2 ** 3 + a2 * x2 ** 2 + a3 * x2 + a4 + noise     # 3次関数にノイズを重畳

    # 近似パラメータakを算出
    coe = np.polyfit(x2, y2, 3)
    print(coe)

    # 得られたパラメータakからカーブフィット後の波形を作成
    y_fit = coe[0] * x2 ** 3 + coe[1] * x2 ** 2 + coe[2] * x2 + coe[3]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')


    # データプロットの準備。
    ax1.scatter(x2, y2, label='sample', lw=1, marker="o", color="pink")
    ax1.plot(x2, y_fit, label='fitted curve', lw=1, color="blue")
    plt.savefig('/usr/src/app/static/figure02.jpg')
    return render_template('kyokusen.html', xlist2=x2, ylist2=y2)




@app.route("/kadai3")
def kadai3():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    try:
        os.remove('/usr/src/app/static/figure03.jpg')
        os.remove('/usr/src/app/static/figure04.jpg')
    except:
        pass

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Define the derivative of the sigmoid function
    def sigmoid_derivative(x):
        return x * (1 - x)

    # Define the input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Define the output data
    y = np.array([[0], [1], [1], [0]])

    # Define the weights and biases for the first layer
    w1 = np.random.random((2, 3))
    b1 = np.random.random((1, 3))

    # Define the weights and biases for the second layer
    w2 = np.random.random((3, 1))
    b2 = np.random.random((1, 1))

    # Train the neural network
    errors = []
    for i in range(50000):
        # Forward propagation
        z1 = np.dot(X, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)

        # Backpropagation
        error = y - a2
        delta2 = error * sigmoid_derivative(a2)
        delta1 = delta2.dot(w2.T) * sigmoid_derivative(a1)

        # Update the weights and biases
        w2 += a1.T.dot(delta2)
        b2 += np.sum(delta2, axis=0, keepdims=True)
        w1 += X.T.dot(delta1)
        b1 += np.sum(delta1, axis=0)

        # Calculate the mean squared error
        mse = np.mean(np.square(error))
        errors.append(mse)

        # Plot the error over time
    plt.figure()
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.savefig('/usr/src/app/static/figure03.jpg')


    plt.figure()
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    # Plot the predicted output
    plt.scatter(range(len(y)), y, color='red', label='Actual')
    plt.plot(range(len(y)), y, color='red')

    plt.scatter(range(len(a2)), a2, color='blue', label='Predicted')
    plt.plot(range(len(a2)), a2, color='blue')

    plt.xlabel('Data Point')
    plt.ylabel('Output')
    plt.legend()
    plt.savefig('/usr/src/app/static/figure04.jpg')


    time.sleep(1)
    return render_template('kadai3.html')




@app.route("/about")
def about():
    return render_template('about.html')




@app.route("/saiyu")
def saiyu():
    try:
        os.remove('/usr/src/app/static/beizu1.png')
        os.remove('/usr/src/app/static/beizu2.png')
        os.remove('/usr/src/app/static/saiyu1.png')
        os.remove('/usr/src/app/static/saiyu2.png')
    except:
        pass

### ベイズ
    # Fields
    train_size = 32
    noise = 1.0

    # Figure
    ax = []
    plt.clf()
    ax_count = 1
    max_plot_num = 5
    indx = 10
    fig = plt.figure(figsize=(15, 30))
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3)

    # Training data
    desired_sizes = [10]  # 指定するデータセットの数
    for desired_size in desired_sizes:
        print(desired_size)
        dataset = norm(loc=5, scale=1).rvs(size=desired_size)  # データセットを生成

        # Plot
        ax.append(fig.add_subplot((max_plot_num), 1, ax_count))
        ax[-1].set_xlabel('x')
        ax[-1].set_ylabel('y')
        ax[-1].set_title('Epochs [%d]' % desired_size)

        # True distribution
        xdata = np.linspace(0, 10.0, 100)
        true_dist = norm(loc=5, scale=1)
        ax[-1].plot(xdata, true_dist.pdf(xdata), color='green', label='Ground Truth')

        # Dataset
        #ax[-1].scatter(dataset, true_dist.pdf(dataset), marker='x', color='blue', label="Given Datasets")

        # beizu Estimate
        mean_cntr, var_cntr, std_cntr = bayes_mvs(dataset, alpha=0.9)
        estimate = norm(loc=mean_cntr.statistic, scale=std_cntr.statistic)
        ax[-1].plot(xdata, estimate.pdf(xdata), color='red', label="beizu Estimate")

        # Plot histogram of the dataset
        ax[-1].hist(dataset, bins=10, density=True, alpha=0.5, color='gray', label="Histogram of Dataset")

        ### 最尤
        t_hat = np.mean(dataset)  # mu
        beta = np.sqrt(np.var(dataset))
        estimate = norm(loc=t_hat, scale=np.sqrt(beta))
        ax[-1].plot(xdata, estimate.pdf(xdata), color='blue', label=" saiyu Estimate")

        ax_count += 1
        plt.legend()

    plt.savefig('/usr/src/app/static/beizu1.png', bbox_inches='tight', dpi=300)


    ### ベイズ
    # Fields
    train_size = 32
    noise = 1.0

    # Figure
    ax = []
    plt.clf()
    ax_count = 1
    max_plot_num = 5
    indx = 10
    fig = plt.figure(figsize=(15, 30))
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3)

    # Training data
    desired_sizes = [300]  # 指定するデータセットの数
    for desired_size in desired_sizes:
        print(desired_size)
        dataset = norm(loc=5, scale=1).rvs(size=desired_size)  # データセットを生成

        # Plot
        ax.append(fig.add_subplot((max_plot_num), 1, ax_count))
        ax[-1].set_xlabel('x')
        ax[-1].set_ylabel('y')
        ax[-1].set_title('Epochs [%d]' % desired_size)

        # True distribution
        xdata = np.linspace(0, 10.0, 100)
        true_dist = norm(loc=5, scale=1)
        ax[-1].plot(xdata, true_dist.pdf(xdata), color='green', label='Ground Truth')

        # Dataset
        #ax[-1].scatter(dataset, true_dist.pdf(dataset), marker='x', color='blue', label="Given Datasets")

        # Estimate
        mean_cntr, var_cntr, std_cntr = bayes_mvs(dataset, alpha=0.9)
        estimate = norm(loc=mean_cntr.statistic, scale=std_cntr.statistic)
        ax[-1].plot(xdata, estimate.pdf(xdata), color='red', label="beizu Estimate")

        # Plot histogram of the dataset
        ax[-1].hist(dataset, bins=10, density=True, alpha=0.5, color='gray', label="Histogram of Dataset")

        ### 最尤
        t_hat = np.mean(dataset)  # mu
        beta = np.sqrt(np.var(dataset))
        estimate = norm(loc=t_hat, scale=np.sqrt(beta))
        ax[-1].plot(xdata, estimate.pdf(xdata), color='blue', label=" saiyu Estimate")

        ax_count += 1
        plt.legend()

    plt.savefig('/usr/src/app/static/beizu2.png', bbox_inches='tight', dpi=300)



    time.sleep(1)
    return render_template('saiyu.html')


@app.route("/rbfbeizu")
def rbfbeizu():
    try:
        os.remove('/usr/src/app/static/rbfbeizu.png')
    except:
        pass
    # 偏りの割合（-側に何パーセントプロットするか）
    bias_percent = 90

    # サンプルデータの生成（偏りを持たせる）
    np.random.seed(0)

    num_points = 100
    num_points_left = int(num_points * bias_percent / 100)
    num_points_right = num_points - num_points_left

    X_train_left = np.linspace(0, 5, num_points_left)  # -5から0までにnum_points_left個の点を生成
    X_train_right = np.linspace(5, 10, num_points_right)   # 0から5までにnum_points_right個の点を生成
    X_train = np.concatenate([X_train_left, X_train_right])
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, X_train.shape[0])

    # ベイズ推定のための事前分布の設定
    prior_mean = 0
    prior_variance = 1

    # 予測分布の計算
    def predict(X_train, y_train, X_test, prior_mean, prior_variance, noise_variance):
        # データ行列の作成
        Phi_train = np.column_stack([np.exp(-(X_train - mu)**2 / (2 * noise_variance)) for mu in X_train])
        Phi_test = np.column_stack([np.exp(-(X_test - mu)**2 / (2 * noise_variance)) for mu in X_train])

        # 事後分布の計算
        posterior_precision = (1 / noise_variance) * np.dot(Phi_train.T, Phi_train) + (1 / prior_variance) * np.eye(Phi_train.shape[1])
        posterior_covariance = np.linalg.inv(posterior_precision)
        posterior_mean = (1 / noise_variance) * np.dot(np.dot(posterior_covariance, Phi_train.T), y_train)

        # 予測分布の計算
        predictive_mean = np.dot(Phi_test, posterior_mean)
        predictive_variance = noise_variance + np.diag(np.dot(np.dot(Phi_test, posterior_covariance), Phi_test.T))

        return predictive_mean, predictive_variance

    # テストデータの生成
    X_test = np.linspace(-7, 7, 100)

    # 予測分布の計算
    predictive_mean, predictive_variance = predict(X_train, y_train, X_test, prior_mean, prior_variance, noise_variance=0.1)

    # グラフの描画
#    plt.figure(figsize=(10, 6))
    plt.figure()
    plt.scatter(X_train, y_train, color='red', label='Training Data')
    plt.plot(X_test, np.sin(X_test), color='green', linestyle='dashed', label='True Function')
    plt.plot(X_test, predictive_mean, color='blue', label='Predictive Mean')
    plt.fill_between(X_test, predictive_mean - np.sqrt(predictive_variance), predictive_mean + np.sqrt(predictive_variance), color='gray', alpha=0.4, label='Predictive Variance')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Bayesian Regression with RBF Network')

    plt.savefig('/usr/src/app/static/rbfbeizu.png', bbox_inches='tight', dpi=300)
    time.sleep(0.5)

    return render_template('rbfbeizu.html')


@app.route("/meiro")
def meiro():
    maze_data = [
        ['S', '1', '0', '0', '0', '0', '1', '0'],
        ['0', '1', '0', '1', '1', '0', '0', '0'],
        ['0', '0', '0', '0', '1', '0', '1', '0'],
        ['0', '1', '1', '0', '0', '0', '1', '0'],
        ['0', '0', '0', '1', '1', '0', 'G', '0'],
        ['0', '1', '0', '0', '0', '0', '1', '0'],
        ['0', '1', '1', '1', '0', '1', '1', '0'],
        ['0', '0', '0', '0', '0', '0', '1', '0']
    ]

    # 行動に対応する移動方向（上、下、左、右）
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # TD学習法のパラメータ
    learning_rate = 0.1
    discount_factor = 0.9
    num_episodes = 100

    # 初期状態価値関数の設定
    state_value_function = np.zeros_like(maze_data, dtype=float)

    # ゴール地点の価値を非常に高く設定
    state_value_function[np.array(maze_data) == 'G'] = 10000

    # TD学習法による学習
    for episode in range(num_episodes):
        # エージェントの初期位置（スタート地点）を設定
        row, col = 0, 0

        while maze_data[row][col] != 'G':
            # 現在の状態の価値を取得
            current_value = state_value_function[row][col]

            # ランダムに行動を選択
            action = np.random.choice(range(len(actions)))
            next_row, next_col = row + actions[action][0], col + actions[action][1]

            # 壁にぶつかる場合はそのままの位置にとどまる
            if 0 <= next_row < len(maze_data) and 0 <= next_col < len(maze_data[0]) and maze_data[next_row][next_col] != '1':
                # TD学習法による状態価値関数の更新
                reward = 0 if maze_data[next_row][next_col] == '0' else -1
                next_value = state_value_function[next_row][next_col]
                state_value_function[row][col] = current_value + learning_rate * (reward + discount_factor * next_value - current_value)

                # 次の状態へ移動
                row, col = next_row, next_col

    return render_template('meiro.html', maze_data=maze_data, state_value_function=state_value_function)

#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=80)
#    app.run(host="0.0.0.0")


