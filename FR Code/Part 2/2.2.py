# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 19:15:22 2025

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 模拟一个投资组合的历史收益率数据
days = 500
mu = 0.001  # 日均收益率
sigma = 0.015  # 日波动率

# 生成服从正态分布的收益率
returns = np.random.normal(mu, sigma, days)

# 计算累积收益
cumulative_returns = np.cumprod(1 + returns) - 1

# 初始投资金额
initial_investment = 1000000  # 100万元

# 计算历史VaR (95% 置信度)
confidence_level = 0.95
var_percentile = 1 - confidence_level
var_historical = np.percentile(returns, var_percentile * 100) * initial_investment

# 计算参数化VaR (95% 置信度)
var_parametric = stats.norm.ppf(var_percentile, mu, sigma) * initial_investment

# 蒙特卡洛模拟计算VaR
num_simulations = 10000
mc_returns = np.random.normal(mu, sigma, num_simulations)
var_monte_carlo = np.percentile(mc_returns, var_percentile * 100) * initial_investment

print(f"置信水平: {confidence_level*100}%")
print(f"历史模拟VaR: {-var_historical:.2f} 元")
print(f"参数法VaR: {-var_parametric:.2f} 元")
print(f"蒙特卡洛VaR: {-var_monte_carlo:.2f} 元")

# 可视化收益分布和VaR
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.hist(returns, bins=50, density=True, alpha=0.6, color='blue')
plt.axvline(x=np.percentile(returns, var_percentile * 100), color='red', linestyle='--', 
           label=f'历史VaR (95%): {-var_historical:.2f} 元')
plt.title('收益率分布与VaR')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(days), cumulative_returns, color='green')
plt.title('累积收益曲线')
plt.xlabel('交易日')
plt.ylabel('累积收益率')

plt.tight_layout()
plt.show()