import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 加载数据集
msft_data_df = pd.read_csv('1.6_MSFT_data.csv', skiprows=2)
aapl_data_df = pd.read_csv('1.6_AAPL_data.csv', skiprows=2)
googl_data_df = pd.read_csv('1.6_GOOGL_data.csv', skiprows=2)
market_data_df = pd.read_csv('1.6_market_data.csv', skiprows=2)
risk_free_rate_df = pd.read_csv('1.6_risk_free_rate.csv')
unemployment_rate_df = pd.read_csv('1.6_unemployment_rate.csv')
inflation_rate_df = pd.read_csv('1.6_inflation_rate.csv')
gdp_data_df = pd.read_csv('1.6_gdp_data.csv')

# 统一列名
column_names = ["Date", "Price", "Close", "High", "Low", "Open"]  # 移除 'Volume' 列名
msft_data_df.columns = column_names
aapl_data_df.columns = column_names
googl_data_df.columns = column_names
market_data_df.columns = column_names

# 将 "Close" 列重命名为 "Adj Close" 并转换日期列
msft_data_df.rename(columns={'Close': 'Adj Close'}, inplace=True)
aapl_data_df.rename(columns={'Close': 'Adj Close'}, inplace=True)
googl_data_df.rename(columns={'Close': 'Adj Close'}, inplace=True)
market_data_df.rename(columns={'Close': 'Adj Close'}, inplace=True)

# 转换日期列为日期格式
msft_data_df['Date'] = pd.to_datetime(msft_data_df['Date'])
aapl_data_df['Date'] = pd.to_datetime(aapl_data_df['Date'])
googl_data_df['Date'] = pd.to_datetime(googl_data_df['Date'])
market_data_df['Date'] = pd.to_datetime(market_data_df['Date'])

# 确保所有宏观经济数据的 'Date' 列转换为 datetime 类型
risk_free_rate_df['DATE'] = pd.to_datetime(risk_free_rate_df['DATE'])
unemployment_rate_df['DATE'] = pd.to_datetime(unemployment_rate_df['DATE'])
inflation_rate_df['DATE'] = pd.to_datetime(inflation_rate_df['DATE'])
gdp_data_df['DATE'] = pd.to_datetime(gdp_data_df['DATE'])

# 将 "Date" 设置为索引
msft_data_df.set_index('Date', inplace=True)
aapl_data_df.set_index('Date', inplace=True)
googl_data_df.set_index('Date', inplace=True)
market_data_df.set_index('Date', inplace=True)

# 计算每日回报率
msft_data_df['Return'] = msft_data_df['Adj Close'].pct_change()
aapl_data_df['Return'] = aapl_data_df['Adj Close'].pct_change()
googl_data_df['Return'] = googl_data_df['Adj Close'].pct_change()
market_data_df['Return'] = market_data_df['Adj Close'].pct_change()

# 合并数据
merged_df = msft_data_df[['Return']].merge(aapl_data_df[['Return']], on='Date', suffixes=('_MSFT', '_AAPL'))
merged_df = merged_df.merge(googl_data_df[['Return']], on='Date')
merged_df = merged_df.merge(market_data_df[['Return']], on='Date', suffixes=('', '_Market'))

# 重置索引，将 'Date' 列恢复为普通列
merged_df.reset_index(inplace=True)

# 合并宏观经济数据
merged_df = merged_df.merge(risk_free_rate_df[['DATE', 'GS10']], left_on='Date', right_on='DATE', how='left', suffixes=('', '_RiskFree'))
merged_df = merged_df.merge(unemployment_rate_df[['DATE', 'UNRATE']], left_on='Date', right_on='DATE', how='left', suffixes=('', '_Unemployment'))
merged_df = merged_df.merge(inflation_rate_df[['DATE', 'CPIAUCSL']], left_on='Date', right_on='DATE', how='left', suffixes=('', '_Inflation'))
merged_df = merged_df.merge(gdp_data_df[['DATE', 'GDP']], left_on='Date', right_on='DATE', how='left', suffixes=('', '_GDP'))

# 删除重复的 'DATE' 列
merged_df = merged_df.drop(columns=['DATE'])

# 删除缺失值
merged_df = merged_df.dropna()

# 计算超额收益
merged_df['Excess_Return_MSFT'] = merged_df['Return_MSFT'] - merged_df['GS10'] / 100
merged_df['Excess_Return_AAPL'] = merged_df['Return_AAPL'] - merged_df['GS10'] / 100
merged_df['Excess_Return_GOOGL'] = merged_df['Return'] - merged_df['GS10'] / 100

# 设置回归的自变量（宏观经济因子）
factors = merged_df[['GS10', 'UNRATE', 'CPIAUCSL', 'GDP']]
factors = sm.add_constant(factors)  # 加入常数项

# 因变量：超额回报
y_msft = merged_df['Excess_Return_MSFT']
y_aapl = merged_df['Excess_Return_AAPL']
y_googl = merged_df['Excess_Return_GOOGL']

# 进行回归分析
model_msft = sm.OLS(y_msft, factors).fit()
model_aapl = sm.OLS(y_aapl, factors).fit()
model_googl = sm.OLS(y_googl, factors).fit()

# 输出回归结果
print("MSFT回归结果：\n", model_msft.summary())
print("\nAAPL回归结果：\n", model_aapl.summary())
print("\nGOOGL回归结果：\n", model_googl.summary())

