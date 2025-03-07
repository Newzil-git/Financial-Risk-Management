import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 使用内建的中文字体设置
matplotlib.rcParams['font.family'] = 'SimHei'  
matplotlib.rcParams['axes.unicode_minus'] = False
# 读取数据并清理
data_file = '4.1 AAPL_data.csv'
data = pd.read_csv(data_file)

# 将 'Close' 列转换为数字类型
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# 清理掉含有NaN的行
data_clean = data.dropna(subset=['Close'])
prices_clean = data_clean['Close'].values

# 计算日收益率
returns_clean = np.diff(prices_clean) / prices_clean[:-1]

# 估计均值(mu)和波动率(sigma)
mu_clean = np.mean(returns_clean)
sigma_clean = np.std(returns_clean)

# 设置模拟参数
n_steps = 100  # 模拟的步数
delta_t = 1 / 252  # 每步的时间间隔（假设每年252个交易日）
initial_price = prices_clean[-1]


# 模拟多条路径的函数
def simulate_multiple_paths(n_paths, n_steps, mu, sigma, initial_price, delta_t):
    all_paths = []
    for _ in range(n_paths):
        simulated_prices = [initial_price]
        for i in range(n_steps):
            Z = np.random.normal(0, 1)
            next_price = simulated_prices[-1] * np.exp((mu - 0.5 * sigma ** 2) * delta_t + sigma * np.sqrt(delta_t) * Z)
            simulated_prices.append(next_price)
        all_paths.append(simulated_prices)
    return all_paths


# 模拟3条路径
n_paths = 3
paths = simulate_multiple_paths(n_paths, n_steps, mu_clean, sigma_clean, initial_price, delta_t)

# 绘制多条路径
for i, path in enumerate(paths):
    plt.plot(path, label=f'Path #{i + 1}')

plt.title('模拟价格路径')
plt.xlabel('未来步骤')
plt.ylabel('价格')
plt.legend()
plt.show()

# 将模拟数据存储到DataFrame中，并生成表格
simulated_prices_clean = np.array(paths[0])
steps = list(range(n_steps + 1))
uniform_random = [None] + [np.random.uniform(0, 1) for _ in range(n_steps)]
normal_random = [None] + [
    (mu_clean - 0.5 * sigma_clean ** 2) * delta_t + sigma_clean * np.sqrt(delta_t) * np.random.normal(0, 1) for _ in
    range(n_steps)]
simulated_data = pd.DataFrame({
    "Step": steps,
    "Uniform": uniform_random,
    "Normal": normal_random,
    "Price Increment": [None] + [
        (mu_clean - 0.5 * sigma_clean ** 2) * delta_t + sigma_clean * np.sqrt(delta_t) * np.random.normal(0, 1) for _ in
        range(n_steps)],
    "Price": simulated_prices_clean
})

# 输出结果数据集
simulated_data.to_csv('4.1_模拟.csv', index=False)
