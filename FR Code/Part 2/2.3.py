# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:42:21 2025

@author: Lenovo
"""

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# 定义参数
mu = 0.1
sigma = np.sqrt(0.2)
S0 = 100

# 计算95%置信区间的上下界
lower_bound = S0 * np.exp(mu - 1.96 * sigma)
upper_bound = S0 * np.exp(mu + 1.96 * sigma)

# 绘制对数正态分布图
x = np.linspace(50, 200, 1000)
y = norm.pdf(np.log(x/S0), mu, sigma) / x
plt.plot(x, y, label='Lognormal Distribution')
plt.fill_between(x, y, where=(x >= lower_bound) & (x <= upper_bound), color='green', alpha=0.5, label='95% Confidence Interval')
plt.title('Lognormal Distribution of Stock Price')
plt.xlabel('Stock Price')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

print(f"95%置信区间为: [{lower_bound:.2f}, {upper_bound:.2f}]")