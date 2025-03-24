# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:39:14 2025

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, kurtosis

# 定义投资组合的偏度和峰度
skewness_A = -1.6
kurtosis_A = 1.9
skewness_B = 0.8
kurtosis_B = 3.2

# 正态分布的峰度为3
normal_kurtosis = 3

# 生成偏度和峰度的可视化数据
data_A = skewnorm.rvs(skewness_A, size=1000)
data_B = skewnorm.rvs(skewness_B, size=1000)

# 绘制直方图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data_A, bins=50, alpha=0.6, color='blue', label=f'Skewness: {skewness_A}, Kurtosis: {kurtosis_A}')
plt.title('Investment Portfolio A')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(data_B, bins=50, alpha=0.6, color='orange', label=f'Skewness: {skewness_B}, Kurtosis: {kurtosis_B}')
plt.title('Investment Portfolio B')
plt.legend()

plt.show()