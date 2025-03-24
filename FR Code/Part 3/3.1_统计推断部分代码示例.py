import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ======================
# 1. 数据生成与基本统计
# ======================
np.random.seed(2023)

# 生成模拟数据：假设真实μ=-0.18%，σ=3.24%，T=240
T = 240
true_mu = -0.0018
true_sigma = 0.0324

# 生成厚尾数据（使用t分布模拟非正态性）
returns = true_mu + true_sigma * np.random.standard_t(5, T)/np.sqrt(5/3)

# 转换为DataFrame便于分析
df = pd.DataFrame({'Return': returns})

# 计算基本统计量
mu_hat = df['Return'].mean()
sigma_hat = df['Return'].std(ddof=1)
skew = stats.skew(df['Return'])
kurt = stats.kurtosis(df['Return'], fisher=False)  # Fisher=False得到Pearson峰度

print(f"样本均值: {mu_hat:.4f}")
print(f"样本标准差: {sigma_hat:.4f}")
print(f"样本偏度: {skew:.2f}")
print(f"样本峰度: {kurt:.2f}")

# ======================
# 2. 均值检验 (t检验)
# ======================
# 原假设 H0: μ = 0
t_stat = (mu_hat - 0)/(sigma_hat/np.sqrt(T))
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), T-1))

print(f"\nt统计量: {t_stat:.2f}")
print(f"P值: {p_value:.4f}")

# 置信区间计算
conf_level = 0.95
t_critical = stats.t.ppf((1 + conf_level)/2, T-1)
ci_low = mu_hat - t_critical * sigma_hat/np.sqrt(T)
ci_high = mu_hat + t_critical * sigma_hat/np.sqrt(T)
print(f"95%置信区间: [{ci_low:.4f}, {ci_high:.4f}]")

# ======================
# 3. 方差检验 (卡方检验)
# ======================
# 原假设 H0: σ^2 = true_sigma^2
chi2_stat = (T-1)*sigma_hat**2 / true_sigma**2
chi2_critical_low = stats.chi2.ppf(0.025, T-1)
chi2_critical_high = stats.chi2.ppf(0.975, T-1)

print(f"\n卡方统计量: {chi2_stat:.2f}")
print(f"卡方临界值区间: [{chi2_critical_low:.2f}, {chi2_critical_high:.2f}]")

# 方差置信区间
ci_var_low = (T-1)*sigma_hat**2 / chi2_critical_high
ci_var_high = (T-1)*sigma_hat**2 / chi2_critical_low
ci_vol_low = np.sqrt(ci_var_low)
ci_vol_high = np.sqrt(ci_var_high)
print(f"波动率95%置信区间: [{ci_vol_low:.4f}, {ci_vol_high:.4f}]")

# ======================
# 4. 正态性检验 (Jarque-Bera)
# ======================
jb_stat = T * (skew**2/6 + (kurt-3)**2/24)
jb_pvalue = 1 - stats.chi2.cdf(jb_stat, 2)

print(f"\nJB统计量: {jb_stat:.2f}")
print(f"P值: {jb_pvalue:.4e}")

# ======================
# 5. Visualization
# ======================
plt.figure(figsize=(12, 8))

# 直方图与正态分布对比
plt.subplot(2,2,1)
x = np.linspace(-0.1, 0.1, 100)
plt.hist(df['Return'], bins=30, density=True, alpha=0.6, label='Empirical')
plt.plot(x, stats.norm.pdf(x, mu_hat, sigma_hat), 'r-', lw=2, label='Normal Fit')
plt.title('Return Distribution vs Normal Fit')
plt.legend()

# Q-Q图
plt.subplot(2,2,2)
sm.qqplot(df['Return'], line='s', ax=plt.gca())
plt.title('Q-Q Plot against Normal Distribution')

# 均值抽样分布模拟
plt.subplot(2,2,3)
sample_means = [np.random.choice(df['Return'], 100).mean() for _ in range(1000)]
plt.hist(sample_means, bins=30, density=True)
plt.title('Sampling Distribution of Sample Mean')
plt.xlabel('Sample Means')

# 波动率置信区间
plt.subplot(2,2,4)
plt.errorbar(0, sigma_hat, yerr=[[sigma_hat - ci_vol_low], [ci_vol_high - sigma_hat]],
             fmt='o', capsize=10)
plt.xlim(-0.5, 0.5)
plt.title('Volatility Estimate with 95% CI')
plt.yticks()
plt.tight_layout()
plt.show()