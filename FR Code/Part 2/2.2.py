# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:41:39 2025

@author: Lenovo
"""

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# 定义均值和标准差
mu = 80
sigma = 24

# 计算P(X < 32)和P(X > 116)
p_less_32 = norm.cdf(32, mu, sigma)
p_greater_116 = 1 - norm.cdf(116, mu, sigma)

# 计算不在32到116范围内的比例
p_outside = p_less_32 + p_greater_116

# 绘制正态分布图
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, label='Normal Distribution')
plt.fill_between(x, y, where=(x < 32) | (x > 116), color='red', alpha=0.5, label='Outside 32-116')
plt.title('Normal Distribution with Mean=80, Std=24')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

print(f"不在32到116范围内的比例为: {p_outside * 100:.2f}%")