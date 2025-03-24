# Financial Risk Management

## Project Overview

This project is a comprehensive implementation of financial risk management models, covering various aspects from asset pricing theory to risk simulation. The project is divided into four main parts, encompassing modern portfolio theory, arbitrage pricing theory, statistical inference, regression analysis, and stochastic process simulation.

## Learning Materials

- **Study Notes (阅读笔记.pdf)**: Notes taken while studying the textbooks. Sections marked with【代码...】have code implementations
- **Code Results Analysis (代码结果展示分析.pdf)**: Contains code execution results and detailed analysis

## Directory Structure

```
Financial-Risk-Management/
├── FR Code/
│   ├── Part 1/    # Asset Pricing Models
│   ├── Part 2/    # Portfolio Optimization
│   ├── Part 3/    # Statistical Inference & Regression
│   ├── Part 4/    # Stochastic Processes & Risk Simulation
│   └── 任务要求.md  # Task Requirements
├── 教材/          # Reference Materials
├── 阅读笔记.pdf    # Study Notes
├── 代码结果展示分析.pdf # Code Results Analysis
├── README.md
├── README.en.md
└── requirements.txt
```

## Detailed Content

### Part 1: Asset Pricing Models

This part contains implementations of various asset pricing models to understand the relationship between asset risk and return.

 Main Contents:

1. **Sharpe Ratio (1.1)**
   - Calculates and analyzes risk-adjusted returns of assets
   
2. **Information Ratio (1.2)**
   - Measures excess return per unit of risk relative to a benchmark
   
3. **Asset Mixing (1.3)**
   - Multi-asset allocation and portfolio construction
   
4. **Treynor Ratio (1.4)**
   - Calculates the ratio of excess return to systematic risk
   
5. **Jensen's Alpha (1.5)**
   - Measures risk-adjusted excess return of a portfolio
   
6. **Arbitrage Pricing Theory (1.6)**
   - Implementation of multi-factor pricing model to analyze how economic factors affect returns

### Part 2:  Portfolio Optimization

This part focuses on portfolio theory and optimization techniques to help investors construct optimal asset allocations.

Main Contents:

1.  **Efficient Frontier Construction (2.1)**
   - Calculation and visualization of efficient portfolio sets
   
2. **Optimal Portfolio Selection (2.2)**
   - Determining optimal asset allocation based on risk preferences
   
3. **Capital Market Line (2.3)**
   - Optimal allocation between risk-free and risky assets
   
4.  **Portfolio Performance Evaluation (2.4)**
   - Calculation of various metrics for evaluating portfolio performance

### Part 3: Statistical Inference & Regression

This part provides implementations of statistical methods needed for financial data analysis to validate financial theories and models.

 Main Contents:

1.  **Statistical Inference (3.1)**
   - Hypothesis testing, confidence intervals, and their applications in finance
   
2. **Regression Analysis (3.2)**
   - Applications of univariate and multivariate regression in financial data analysis

### Part 4: Stochastic Processes & Risk Simulation

This part implements stochastic process models commonly used in financial markets for asset price simulation and risk assessment.

 Main Contents:

1.  **Geometric Brownian Motion (4.1)**
   - Stock price stochastic process simulation

## Environment Requirements

The project code is mainly implemented in Python, with the following dependencies:

```
pandas==1.4.4
numpy==1.21.5
matplotlib==3.5.2
scipy==1.9.1
statsmodels==0.13.2
scikit-learn==1.0.2
seaborn==0.11.2
```

## Installation Guide

1. Clone the repository
   ```
   git clone https://github.com/Newzil-git/Financial-Risk-Management.git
   cd Financial-Risk-Management
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Run Jupyter notebooks or Python scripts
   ```
   jupyter notebook "FR Code/Part 1/1.1 Sharpe ratios.ipynb"
   ```
    or
   ```
   python "FR Code/Part 1/1.1 Sharpe ratios.py"
   ```

##  References

- Textbooks and reference materials used in this project can be found in the `教材/` directory
