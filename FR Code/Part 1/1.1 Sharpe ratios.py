import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 无风险收益率
risk_free_rate = 0.02
# 文件路径
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

# 计算 Sharpe Ratio
def calculate_sharpe_ratio(data, risk_free_rate=0.02):
    avg_daily_return = data['Daily Return'].mean()
    std_daily_return = data['Daily Return'].std()

    # 年化收益率 标准差
    annual_return = avg_daily_return * 252
    annual_std = std_daily_return * np.sqrt(252)

    # 计算 Sharpe Ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_std
    return sharpe_ratio, annual_return, annual_std

# 计算 AAPL 和 SPY 的 Sharpe Ratio
sharpe_aapl, return_aapl, std_aapl = calculate_sharpe_ratio(data_aapl)
sharpe_spy, return_spy, std_spy = calculate_sharpe_ratio(data_spy)

# 打印计算结果
print(f"AAPL 夏普比率: {sharpe_aapl:.2f}, 年化收益率: {return_aapl:.2%}, 年化标准差: {std_aapl:.2%}")
print(f"SPY 夏普比率: {sharpe_spy:.2f}, 年化收益率: {return_spy:.2%}, 年化标准差: {std_spy:.2%}")

# 可视化
# 1. 价格走势
plt.figure(figsize=(12, 5))
plt.plot(data_aapl.index, data_aapl['Adj Close'], label="AAPL Price", color='blue')
plt.plot(data_spy.index, data_spy['Adj Close'], label="SPY Price", color='orange')
plt.title("AAPL vs SPY Price Trend (2022-2025)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()

# 2. 收益率分布
plt.figure(figsize=(10, 5))
plt.hist(data_aapl['Daily Return'].dropna(), bins=50, alpha=0.6, label="AAPL", color='blue')
plt.hist(data_spy['Daily Return'].dropna(), bins=50, alpha=0.6, label="SPY", color='orange')
plt.axvline(data_aapl['Daily Return'].mean(), color='blue', linestyle='dashed', linewidth=2, label="AAPL Mean")
plt.axvline(data_spy['Daily Return'].mean(), color='orange', linestyle='dashed', linewidth=2, label="SPY Mean")
plt.title("Daily Return Distribution: AAPL vs SPY")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# 3. 累计收益率
data_aapl['Cumulative Return'] = (1 + data_aapl['Daily Return']).cumprod()
data_spy['Cumulative Return'] = (1 + data_spy['Daily Return']).cumprod()

plt.figure(figsize=(12, 5))
plt.plot(data_aapl.index, data_aapl['Cumulative Return'], label="AAPL Cumulative Return", color='blue')
plt.plot(data_spy.index, data_spy['Cumulative Return'], label="SPY Cumulative Return", color='orange')
plt.title("Cumulative Return: AAPL vs SPY (2022-2025)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# 资产数据
assets = ['AAPL', 'SPY']
annual_volatilities = [std_aapl, std_spy]  # 年波动率 (X 轴)
annual_returns = [return_aapl, return_spy]  # 年期望收益率 (Y 轴)

# 计算夏普比率斜率
sharpe_ratios = [(r - risk_free_rate) / s for r, s in zip(annual_returns, annual_volatilities)]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制资本市场线 (CML)，从无风险收益率开始
x_values = np.linspace(0, max(annual_volatilities) * 1.2, 100)
y_values = risk_free_rate + sharpe_aapl * x_values  # 使用 AAPL 的 Sharpe Ratio 画斜率
plt.plot(x_values, y_values, linestyle="--", color="black", label="Capital Market Line (CML)")

# 绘制 AAPL 和 SPY 的点
for i, asset in enumerate(assets):
    plt.scatter(annual_volatilities[i], annual_returns[i], s=100, label=asset, edgecolors='black')
    plt.text(annual_volatilities[i], annual_returns[i], f"  {asset}", fontsize=12, verticalalignment='bottom', horizontalalignment='left')

# 绘制无风险资产（现金）的点
plt.scatter(0, risk_free_rate, color='red', s=100, label="Risk-Free Asset", edgecolors='black')
plt.text(0, risk_free_rate, "  Cash", fontsize=12, verticalalignment='bottom', horizontalalignment='left')

# 设置图表标题和标签
plt.title("Sharpe Ratio Comparison (Annual Return vs. Risk)")
plt.xlabel("Annual Volatility (%)")
plt.ylabel("Annual Expected Return (%)")

# 添加图例和网格
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 展示图表
plt.show()
