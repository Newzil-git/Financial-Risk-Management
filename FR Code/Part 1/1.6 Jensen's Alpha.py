import pandas as pd
import numpy as np
import statsmodels.api as sm

# 假设的无风险利率
risk_free_rate = 0.02 / 252

# 读取数据
file_path_aapl = "AAPL_data.csv"
file_path_spy = "SPY_data.csv"

# 读取 CSV 数据
column_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
data_aapl = pd.read_csv(file_path_aapl, skiprows=3, names=column_names)
data_spy = pd.read_csv(file_path_spy, skiprows=3, names=column_names)

# 统一列名
data_aapl.rename(columns={'Close': 'Adj Close'}, inplace=True)
data_spy.rename(columns={'Close': 'Adj Close'}, inplace=True)

# 转换日期格式
data_aapl['Date'] = pd.to_datetime(data_aapl['Date'], format="%Y/%m/%d")
data_spy['Date'] = pd.to_datetime(data_spy['Date'])

# 设置索引
data_aapl.set_index('Date', inplace=True)
data_spy.set_index('Date', inplace=True)

# 计算每日收益率
data_aapl['Daily Return'] = data_aapl['Adj Close'].pct_change()
data_spy['Daily Return'] = data_spy['Adj Close'].pct_change()

# 计算 AAPL 的 Beta 值
X = sm.add_constant(data_spy['Daily Return'].dropna())
y = data_aapl['Daily Return'].dropna()

model = sm.OLS(y, X).fit()
beta_aapl = model.params.iloc[1]


# 计算 AAPL 和 SPY 的平均回报
mu_aapl = data_aapl['Daily Return'].mean()
mu_spy = data_spy['Daily Return'].mean()

# 计算 Jensen's Alpha
jensens_alpha_aapl = mu_aapl - risk_free_rate - beta_aapl * (mu_spy - risk_free_rate)

# 打印结果
print(f"AAPL 的 Jensen's Alpha: {jensens_alpha_aapl:.4f}")
