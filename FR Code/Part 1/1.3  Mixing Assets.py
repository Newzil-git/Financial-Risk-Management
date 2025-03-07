import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 读取AAPL数据和US国债数据
aapl_data = pd.read_csv('AAPL_data.csv')  # AAPL 股票数据
debt_data = pd.read_csv('US_national_debt _data.csv')  # 美国国债数据

# 提取收盘价数据并计算对数收益率
aapl_data['Close'] = pd.to_numeric(aapl_data['Close'], errors='coerce')
debt_data['Close'] = pd.to_numeric(debt_data['Close'], errors='coerce')

# 计算每日收益率（对数收益率）
aapl_returns = np.log(aapl_data['Close'] / aapl_data['Close'].shift(1)).dropna()
debt_returns = np.log(debt_data['Close'] / debt_data['Close'].shift(1)).dropna()

# 计算平均收益率和波动率（标准差）
avg_aapl = aapl_returns.mean()
avg_debt = debt_returns.mean()
std_aapl = aapl_returns.std()
std_debt = debt_returns.std()

# 计算资产间的相关系数
correlation = aapl_returns.corr(debt_returns)

# 计算不同权重下的投资组合收益率和波动率
weights = np.linspace(0, 1, 100)
portfolio_returns = []
portfolio_volatility = []

for w in weights:
    # 组合的预期回报
    portfolio_return = w * avg_aapl + (1 - w) * avg_debt
    portfolio_returns.append(portfolio_return)

    # 组合的波动率
    portfolio_vol = np.sqrt(
        w ** 2 * std_aapl ** 2 + (1 - w) ** 2 * std_debt ** 2 + 2 * w * (1 - w) * correlation * std_aapl * std_debt)
    portfolio_volatility.append(portfolio_vol)

# 绘制风险-回报曲线
plt.figure(figsize=(10, 6))
plt.plot(portfolio_volatility, portfolio_returns, label="Portfolio Mix")
plt.scatter(std_aapl, avg_aapl, color='blue', label="AAPL (Stock)", marker='o')
plt.scatter(std_debt, avg_debt, color='red', label="US Debt (Bond)", marker='x')
plt.title('Risk-Return Profile of Portfolio Mix')
plt.xlabel('Volatility (%)')
plt.ylabel('Expected Return (%)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# 输出计算的结果
print(f"AAPL Expected Return: {avg_aapl * 100:.2f}%")
print(f"AAPL Volatility: {std_aapl * 100:.2f}%")
print(f"US Debt Expected Return: {avg_debt * 100:.2f}%")
print(f"US Debt Volatility: {std_debt * 100:.2f}%")
print(f"Correlation between AAPL and US Debt: {correlation:.2f}")
