import numpy as np

def get_sample(dim, n):
    x = np.random.rand(n, dim) * 2 - 1
    # バイアス項を追加
    x = np.hstack((np.ones((n, 1)), x))
        
    # ラベルは[1, -1]のどっちか
    y = np.where(np.array(x[:, 1] < x[:, 2]), 1, -1)
    return x, y
