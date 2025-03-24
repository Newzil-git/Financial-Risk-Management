# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:42:59 2025

@author: Lenovo
"""

from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

# 定义参数
n = 6
p = 0.25

# 计算猜对0个或1个的概率
p_0 = binom.pmf(0, n, p)
p_1 = binom.pmf(1, n, p)

# 计算猜对低于两个的概率
p_less_than_2 = p_0 + p_1

# 绘制二项分布图
x = np.arange(0, n+1)
y = binom.pmf(x, n, p)
plt.bar(x, y, color='blue', alpha=0.6, label='Binomial Distribution')
plt.bar([0, 1], [p_0, p_1], color='red', alpha=0.6, label='Less than 2 correct')
plt.title('Binomial Distribution of Correct Answers')
plt.xlabel('Number of Correct Answers')
plt.ylabel('Probability')
plt.legend()
plt.show()

print(f"猜对低于两个的概率为: {p_less_than_2 * 100:.2f}%")