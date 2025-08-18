import numpy as np
from pandas import DataFrame
import random, os
from copy import deepcopy

# 均匀地生成权重向量
class Mean_vector:
    # 对m维空间，目标方向个数H
    def __init__(self, H=5, m=3):
        self.H = H
        self.m = m
        self.stepsize = 1 / H

    def perm(self, sequence):
        # ！！！ 序列全排列，且无重复
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_mean_vectors(self):
    #生成权均匀向量
        H = self.H
        m = self.m
        sequence = []
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        ws = []

        pe_seq = self.perm(sequence)
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)
        return ws


def generate_weights_batch_dfs(i, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch):
    if i == obj_num - 1:
        weight.append(1.0 - np.sum(weight[0:i]))
        weights_batch.append(deepcopy(weight))
        weight = weight[0:i]
        return
    w = min_weight
    while w < max_weight + 0.5 * delta_weight and np.sum(weight[0:i]) + w < 1.0 + 0.5 * delta_weight:
        weight.append(w)
        generate_weights_batch_dfs(i + 1, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch)
        weight = weight[0:i]
        w += delta_weight


mv = Mean_vector(1, 2)  # （12， 3）
m_v = mv.get_mean_vectors()

weights = DataFrame(m_v)
weights.to_csv('all_weights.csv', index=False)

# weights_batch = []
# generate_weights_batch_dfs(0, 3, 0, 1, 0.07, [], weights_batch)












