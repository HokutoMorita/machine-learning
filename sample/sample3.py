# -*- coding: utf-8 -*-

import numpy as np

input_data = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
data_number = input_data.shape[0]

init_weight = np.random.rand(3, 3) # 3x3の行列に0から1の値（小数点も含む）をランダムで代入している

# 重みとバイアスの初期値
w111 = init_weight[0, 0]
w112 = init_weight[0, 1]
b11 = w111 = init_weight[0, 2]

w121 = init_weight[1, 0]
w122 = init_weight[1, 1]
b12 = w111 = init_weight[1, 2]

w211 = init_weight[2, 0]
w212 = init_weight[2, 1]
b21 = w111 = init_weight[2, 2]

epochs = 10000
alpha = 0.1

def S(z):
    return 1 / (1 + np.exp(-z))

for t in range(epochs):
    for i in range(data_number):
        x1 = input_data[i, 0]
        x2 = input_data[i, 1]
        Y = input_data[i, 2]
        
        Z11 = b11 + (w111 * x1) + (w112 * x2)
        Z12 = b12 + (w121 * x1) + (w122 * x2)
        
        # Sはシグモイド関数
        O11 = S(Z11)
        O12 = S(Z12)
        
        Z21 = b21 + w211*O11 + w212*O12
        
        O21 = S(Z21)
        
        # 誤差関数
        error = (Y - O21) ** 2
        
        # 偏微分をして傾きを求める（重みやバイアスを変えて誤差がどれくらい変わるかを求める）
        ## つまり誤差の変化率
        dw111 = 2 * (O21-Y) * S(Z21) * (1 - S(Z21)) * w211 * S(Z11) * (1 - S(Z11)) * x1
        dw112 = 2 * (O21-Y) * S(Z21) * (1 - S(Z21)) * w211 * S(Z11) * (1 - S(Z11)) * x2
        db11 = 2 * (O21-Y) * S(Z21) * (1 - S(Z21)) * w211 * S(Z11) * (1 - S(Z11))
        
        dw121 = 2 * (O21-Y) * S(Z21) * (1 - S(Z21)) * w212 * S(Z12) * (1 - S(Z12)) * x1
        dw122 = 2 * (O21-Y) * S(Z21) * (1 - S(Z21)) * w212 * S(Z12) * (1 - S(Z12)) * x2
        db12 = 2 * (O21-Y) * S(Z21) * (1 - S(Z21)) * w212 * S(Z12) * (1 - S(Z12))
        
        dw211 = 2 * (O21-Y) * S(Z21) * (1 - S(Z21)) * O11
        dw212 = 2 * (O21-Y) * S(Z21) * (1 - S(Z21)) * O21
        db21 = 2 * (O21-Y) * S(Z21) * (1 - S(Z21))
        
        w111 = w111 - alpha * (dw111)
        w112 = w112 - alpha * (dw112)
        b11 = b11 - alpha * (db11)
        
        w121 = w121 - alpha * (dw121)
        w122 = w122 - alpha * (dw122)
        b12 = b12 - alpha * (db12)
        
        w211 = w211 - alpha * (dw211)
        w212 = w212 - alpha * (dw212)
        b21 = b21 - alpha * (db21)

x1 = np.array([0,0,1,1])
x2 = np.array([0,1,0,1])

Z11 = b11 + (w111 * x1) + (w112 * x2)
Z12 = b12 + (w121 * x1) + (w122 * x2)

# Sはシグモイド関数
O11 = S(Z11)
O12 = S(Z12)

Z21 = b21 + w211*O11 + w212*O12

O21 = S(Z21)
print(O21)