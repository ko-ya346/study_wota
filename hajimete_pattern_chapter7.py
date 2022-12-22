import os
import sys
from typing import List

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append("./src")
from model import Perceptron
from data import get_sample


# 特徴ベクトルの次元数
dim = 2
# サンプル数
n = 100
eta = 0.0001

# 学習回数
n_iter = 10

x, y = get_sample(dim, n)

print(x)
print(y)

# パーセプトロンのモデル定義
model = Perceptron(dim=dim, eta=eta)


# サンプルを可視化してみる
# クラスで色分け
class1_cond = y >= 0
class2_cond = y < 0

plt.scatter(x[class1_cond, 1], x[class1_cond, 2], color="r", label="c1")
plt.scatter(x[class2_cond, 1], x[class2_cond, 2], color="b", label="c2")

# 初期パラメータの識別境界を可視化
xx = np.linspace(-1, 1, 20)
plt.plot(xx, (model.w[0] + model.w[1] * xx) / model.w[2] * -1)

plt.legend()
plt.show()

accuracy_l = []
w_l = []

for i in range(n_iter):
    print(f"iter: {i + 1}")
    w_l.append(list(model.w))
    output = model.forward(x)
    accuracy = np.sum(output == y) / len(y)
    print(f"accuracy: {accuracy}")
    accuracy_l.append(accuracy)

    # パラメータ更新
    model.update_w(output, x, y)


# 学習後の識別境界を可視化
plt.scatter(x[class1_cond, 1], x[class1_cond, 2], color="r", label="c1")
plt.scatter(x[class2_cond, 1], x[class2_cond, 2], color="b", label="c2")

xx = np.linspace(-1, 1, 20)
plt.plot(xx, (model.w[0] + model.w[1] * xx) / model.w[2] * -1)
plt.legend()
plt.show()


# パラメータの変化の様子

xx = np.linspace(-1, 1, 20)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

ax.scatter(x[class1_cond, 1], x[class1_cond, 2], color="r", label="c1")
ax.scatter(x[class2_cond, 1], x[class2_cond, 2], color="b", label="c2")

ims = []
for w in w_l:
    im = ax.plot(xx, (w[0] + w[1] * xx) / w[2] * -1)
    ims.append(im)
    
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=100)

ani.save("../output/parceptron_params.gif", writer="pillow")


