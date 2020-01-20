# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

input_data = np.array([[1, 1, 0], [1, 4, 0], [3, 1, 0], [4, 5, 1], [0, 7, 1]])
data_number = input_data.shape[0]

epochs = 1000
alpha = 0.1

# 重みの初期設定
b0 = 0.1
b1 = 0.1
b2 = 0.1


# 何回学習を繰り返すのか（エポックスの設定）
## つまりここでより最適なb0、b1、b2を求めるための学習（機械学習）を行っている
for t in range(epochs): 
    # 偏微分して求めた各傾きの初期設定
    db0 = 0
    db1 = 0
    db2 = 0
    L = 1
    
    for i in range(data_number) :
        x1 = input_data[i, 0] # 酒
        x2 = input_data[i, 1] # たばこ
        Y = input_data[i, 2]
        
        # Sはシグモイド関数 
        ## シグモイド関数の定義
        Z = b0 + (b1 * x1) + (b2 * x2)
        S = 1 / (1 + np.exp(-Z))
        
        if Y==1:
            # 生活習慣病の場合
            db0 = db0 + (1 - S)
            db1 = db1 + (1 - S) * x1 
            db2 = db2 + (1 - S) * x2
            
            # 尤度関数の定義
            ## Lは尤度関数
            L = L * (1 / (1 + np.exp(-Z)))
        else:
            # 健康の場合
            db0 = db0 - S
            db1 = db1 - S * x1
            db2 = db2 - S * x2
            
            ## Lは尤度関数
            L = L * (1 - (1 / (1 + np.exp(-Z))))
        
    # 最急降下法（単回帰分析の時とは異なり、alphaをかけた値を足し合わせている）
    b0 = b0 + alpha * (db0)
    b1 = b1 + alpha * (db1)
    b2 = b2 + alpha * (db2)
    
    # epocsの回数分Lの値を描画
    plt.scatter(t, L)

'''
# 機械学習により求めたb0、b1、b2を使用して、図をプロットする
# Zが0より大きい場合は生活習慣病
# 0 = b0 + b1*x + b2*yを式変形すると
#-(b2 * y) = b0 + b1*x
x = np.linspace(0, 7, 100)
y = - (b0 / b2) - (b1 / b2) * x  

# 線を描画（作成された線の上側にいる人たちは生活習慣病）
plt.plot(x, y)

# input_dataの値を図に描画（散布図）
for n in range(data_number):
    plt.scatter(input_data[n, 0], input_data[n, 1])
'''