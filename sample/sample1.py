# -*- coding: utf-8 --

import numpy as np
import matplotlib.pyplot as plt

####
# 誤差関数の導出を行う
####

# データ構造を二元配列にする[[]]
input_data = np.array([[20, 30], [23, 32],[28, 40],[30, 44]])

# データの個数（サイズ）を代入している。
data_number = input_data.shape[0]

# xとyの最大値と最小値をそれぞれ取得
input_data_min = np.min(input_data, axis=0, keepdims=True) # axis=0を指定することで0列目の値から最小の値を取得している。　keepdims=Trueにより、次元を維持している。 詳しくは実行して変数の値をチェックする
x_min = input_data_min[0, 0]
y_min = input_data_min[0, 1]

input_data_max = np.max(input_data, axis=0, keepdims=True) 
x_max = input_data_max[0, 0]
y_max = input_data_max[0, 1]

# 正規化したデータ
## 他の具体的な値を記述しなくてもnumpyが自動で推測して値を代入してくれる、詳しくは実行して変数input_data_normalizedを確認する
## xの最大値と最小値の差とyの最大値と最小値の差で割る。
input_data_normalized = (input_data - np.array([[20, 30]])) / np.array([[(x_max - x_min), (y_max - y_min)]])

''' 
#### 正規化に必要な情報 ####
xの最小値 = 20
xの最大値 = 30

yの最小値 = 30
yの最大値 = 44

xの最大 - 最小 = 10
yの最大 - 最小 = 14
'''

epochs = 100
#alpha = 0.00005
alpha = 0.1 # 正規化したver

# 誤差関数（最小二乗法のやつだと思う）: 偏微分における傾きを求めるために必要な式、しかし実際のプログラムでは傾きのみが必要なため、コメントアウト
# (w0 + w1x1 - y1)^2 = w0^2 + w1^2x1^2 + 2w0w1x1 - 2w1x1y1 - 2y1w0

w0 = 0.1
w1 = 0.1
for t in range(epochs):
    dw0 = 0
    dw1 = 0
    for i in range(data_number):
        #x1 = input_data[i, 0]
        #y1 = input_data[i, 1]

        x1 = input_data_normalized[i, 0] # 正規化したver
        y1 = input_data_normalized[i, 1] # 正規化したver       
        
        # 誤差関数の導関数の導出
        dw0 = dw0 + (2 * w0) + (2 * w1 * x1) - (2 * y1)
        dw1 = dw1 + x1 * ((2 * w1 * x1) + (2 * w0) - (2 * y1))
    
    # 最急降下法
    w0 = w0 - alpha * (dw0)
    w1 = w1 - alpha * (dw1)
    #print(dw0) # 傾きdw0の値がどのように推移しているかを確認するために使用

## 正規化したverを見たい時は正規化verのコメントが付いているコードのコメントアウトを外す。
x = np.linspace(15, 35, 100) # 15から35の間の値を100個用意する。    
#x = np.linspace(0, 1, 100) # 正規化したver  
y = w0 + w1 * x # ハットyの式、もっとも確からしい関数

# 正規化した関数を元に戻す。
# (y - 30) / 14 = w0 + w1 * ((x - 20) / 10)
y = (14 * (w0 + w1 * ((x - 20) / 10))) + 30 # 上の式を等式変形したver

plt.plot(x, y)

for u in range(data_number):
    plt.scatter(input_data[u, 0], input_data[u, 1])
    #plt.scatter(input_data_normalized[u, 0], input_data_normalized[u, 1]) # 正規化したver   
