import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 无风险收益率
risk_free_rate = 0.02 / 252  # 假设年利率为 2%，转换为日利率

file_path_aapl = "AAPL_data.csv"
file_path_spy = "SPY_data.csv"

# 读取 CSV 并跳过前两行
column_names = ["Date", "Open", "High", "Low", "Close", "Volume"]

data_aapl = pd.read_csv(file_path_aapl, skiprows=3, names=column_names)
data_spy = pd.read_csv(file_path_spy, skiprows=3, names=column_names)

# 统一列名（将 "Close" 重命名为 "Adj Close"）
data_aapl.rename(columns={'Close': 'Adj Close'}, inplace=True)
data_spy.rename(columns={'Close': 'Adj Close'}, inplace=True)

# 转换日期格式并设置索引
data_aapl['Date'] = pd.to_datetime(data_aapl['Date'], format="%Y/%m/%d")
data_spy['Date'] = pd.to_datetime(data_spy['Date'])

data_aapl.set_index('Date', inplace=True)
data_spy.set_index('Date', inplace=True)

# 计算每日收益率
data_aapl['Daily Return'] = data_aapl['Adj Close'].pct_change()
data_spy['Daily Return'] = data_spy['Adj Close'].pct_change()

# 计算 Beta 值（AAPL 与 SPY 的回归分析）
X = sm.add_constant(data_spy['Daily Return'].dropna())  # 加入常数项（截距）
y = data_aapl['Daily Return'].dropna()

# 回归分析
model = sm.OLS(y, X).fit()
beta_aapl = model.params[1]

# 计算 AAPL 和 SPY 的平均回报（日回报）
mu_aapl = data_aapl['Daily Return'].mean()
mu_spy = data_spy['Daily Return'].mean()

# 计算 Treynor 比率
treynor_aapl = (mu_aapl - risk_free_rate) / beta_aapl
treynor_spy = (mu_spy - risk_free_rate) / 1  # SPY 的 Beta 值假定为 1，因为 SPY 是市场本身

# 打印计算结果
print(f"AAPL 的 Beta 值: {beta_aapl:.4f}")
print(f"AAPL 的每日平均回报: {mu_aapl:.4f}")
print(f"AAPL 的 Treynor 比率: {treynor_aapl:.4f}")

print(f"SPY 的 Beta 值: 1.0000")
print(f"SPY 的每日平均回报: {mu_spy:.4f}")
print(f"SPY 的 Treynor 比率: {treynor_spy:.4f}")

