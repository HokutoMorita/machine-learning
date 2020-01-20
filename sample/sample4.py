# -*- coding: utf-8 -*-
import numpy as np

input_data = np.array([[0,0], [0,1], [1,0], [1,1]])
# 期待される出力
expected_output = np.array([[0], [1], [1], [0]])

data_number = input_data.shape[0]

# インプットデータの転置行列の数（つまりノードの数）
input_node = input_data.T.shape[0]

# 各ノードの要素の数
first_node = 3
second_node = 1

alpha = 0.1
epochs = 100

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_prime(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

first_weight = np.random.rand(input_node, first_node) - 0.5 # 0.5をマイナスした理由は0から1の範囲にしたくない（ちゃんとデータをばらつかせるため）
second_weight = np.random.rand(first_node, second_node) - 0.5

# first_node_activationに入力するための行列を計算して作成
first_node_input = np.dot(input_data, first_weight)
first_node_activation = sigmoid(first_node_input)

second_node_input = np.dot(first_node_activation, second_weight)
second_node_activation = sigmoid(second_node_input)

# 誤差関数の導関数（偏微分）
## 第二層
d_error = second_node_activation - expected_output
d_second_node_activation = d_error * sigmoid_prime(second_node_input)
d_second_node_weight = first_node_activation.T.dot(d_second_node_activation)

## 第一層
d_second_node_input = d_second_node_activation.dot(second_weight.T)