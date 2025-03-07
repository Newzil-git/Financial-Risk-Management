import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

spy_data = pd.read_csv('S&P_500（B）_data.csv')
aapl_data = pd.read_csv('AAPL_data.csv')

# 清理数据，去掉无效的行（空值或无效数据）
aapl_data_clean = aapl_data[pd.to_numeric(aapl_data['Close'], errors='coerce').notnull()].reset_index(drop=True)
spy_data_clean = spy_data[pd.to_numeric(spy_data['Close'], errors='coerce').notnull()].reset_index(drop=True)

# 提取收盘价数据
spy_close_clean = spy_data_clean['Close'].dropna().astype(float)
aapl_close_clean = aapl_data_clean['Close'].dropna().astype(float)

# 计算每日收益率（对数收益率）
spy_returns = np.log(spy_close_clean / spy_close_clean.shift(1)).dropna()
aapl_returns = np.log(aapl_close_clean / aapl_close_clean.shift(1)).dropna()

# 计算平均收益率和标准差
avg_spy = spy_returns.mean()
avg_aapl = aapl_returns.mean()
std_spy = spy_returns.std()
std_aapl = aapl_returns.std()

# 计算信息比率
ir = (avg_aapl - avg_spy) / np.sqrt(std_aapl**2 + std_spy**2)

# 生成结果表格
performance_table = pd.DataFrame({
    'Average': [avg_aapl, avg_spy],
    'Volatility': [std_aapl, std_spy],
    'Performance': [ir, None]
}, index=['Portfolio P (AAPL)', 'Benchmark B (S&P 500)'])

print(performance_table)
