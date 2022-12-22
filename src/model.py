import numpy as np


class Perceptron:
    def __init__(self, dim=1, eta=1, init_w=None):
        self.w = self.set_init_w(dim, init_w)
        
        assert len(self.w) == dim + 1
        # 学習率
        self.eta = eta
    
    def forward(self, x):
        return np.sign((x * self.w).sum(axis=1))

    def update_w(self, output, x, y):
        # 出力と正解データが異なる場合のみパラメータを更新
        # サンプルデータをwに加える
        self.w += np.dot((output != y) * np.sign(y), x)
        
    def set_init_w(self, dim, init_w):
        if init_w is None:
            # 正規分布の乱数生成
            w = np.random.randn(dim + 1)
        else:
            w = np.zeros(dim + 1)
        return w
            
