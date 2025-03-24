import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
import seaborn as sns

# 设置随机种子保证可重复性
np.random.seed(2023)

# ======================
# 步骤1：数据生成过程
# ======================
T = 100  # 样本量
alpha_true = 0.1   # 真实截距项
beta_true = 1.5    # 真实斜率系数
sigma = 0.2        # 误差标准差

# 生成解释变量（市场收益率）
x = np.random.normal(0, 0.3, T)
# 生成误差项
epsilon = np.random.normal(0, sigma, T)
# 生成被解释变量（个股收益率）
y = alpha_true + beta_true * x + epsilon

# 创建DataFrame
df = pd.DataFrame({'Market_Return': x, 'Stock_Return': y})

# ======================
# 步骤2：回归模型估计
# ======================
# 添加常数项
X = sm.add_constant(df['Market_Return'])
model = sm.OLS(df['Stock_Return'], X)
results = model.fit()

# 提取关键结果
alpha_est = results.params[0]
beta_est = results.params[1]
se_alpha = results.bse[0]
se_beta = results.bse[1]
t_alpha = results.tvalues[0]
t_beta = results.tvalues[1]
p_alpha = results.pvalues[0]
p_beta = results.pvalues[1]
r_squared = results.rsquared
adj_r_squared = results.rsquared_adj

# ======================
# 步骤3：统计检验
# ======================
# 残差分析
residuals = results.resid
fitted = results.fittedvalues

# Jarque-Bera正态性检验
jb_stat, jb_pval = normal_ad(residuals)

# Breusch-Pagan异方差检验
bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X)

# Durbin-Watson自相关检验
dw_stat = sm.stats.durbin_watson(residuals)

# ======================
# 步骤4：可视化分析
# ======================
plt.style.use('seaborn')

# 图1：散点图与回归线
plt.figure(figsize=(10, 6))
sns.regplot(x='Market_Return', y='Stock_Return', data=df,
            line_kws={'color':'red', 'lw':2},
            scatter_kws={'alpha':0.6})
plt.title('Stock Return vs Market Return', fontsize=14)
plt.xlabel('Market Return', fontsize=12)
plt.ylabel('Stock Return', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')

# 图2：残差分布图
plt.figure(figsize=(10, 6))
plt.scatter(fitted, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Analysis', fontsize=14)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')

# ======================
# 结果输出
# ======================
print("="*50)
print("Regression Results Summary")
print("="*50)
print(f"Sample Size: {T}")
print(f"Intercept Estimate (α): {alpha_est:.4f} (SE: {se_alpha:.4f})")
print(f"Slope Coefficient (β): {beta_est:.4f} (SE: {se_beta:.4f})")
print(f"R-squared: {r_squared:.4f}")
print(f"Adjusted R-squared: {adj_r_squared:.4f}\n")

print("Statistical Tests:")
print(f"Intercept t-statistic: {t_alpha:.2f} (p-value: {p_alpha:.4f})")
print(f"Slope t-statistic: {t_beta:.2f} (p-value: {p_beta:.4f})")
print(f"Jarque-Bera Test: Statistic={jb_stat:.2f} (p-value={jb_pval:.4f})")
print(f"Breusch-Pagan Test: Statistic={bp_stat:.2f} (p-value={bp_pval:.4f})")
print(f"Durbin-Watson Statistic: {dw_stat:.2f}")

print("\nVisualizations saved as:")
print("scatter_plot.png (Scatter plot with regression line)")
print("residual_plot.png (Residual plot)")