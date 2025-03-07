import os
import pandas as pd

def get_data_path(file_name):
    """
    获取数据文件的路径，支持本地和GitHub两种方式
    """
    # 首先尝试本地路径
    local_paths = [
        file_name,  # 当前目录
        os.path.join("FR Code", "Part 1", file_name),  # Part 1目录
        os.path.join("FR Code", "Part 2", file_name),  # Part 2目录
        os.path.join("..", "FR Code", "Part 1", file_name),  # 相对于notebook的Part 1目录
        os.path.join("..", "FR Code", "Part 2", file_name),  # 相对于notebook的Part 2目录
        os.path.join("Financial-Risk-Management", "FR Code", "Part 1", file_name),  # Colab克隆目录
        os.path.join("Financial-Risk-Management", "FR Code", "Part 2", file_name),  # Colab克隆目录
        os.path.join("/content/Financial-Risk-Management", "FR Code", "Part 1", file_name),  # Colab完整路径
        os.path.join("/content/Financial-Risk-Management", "FR Code", "Part 2", file_name)  # Colab完整路径
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            print(f"找到本地文件：{path}")
            return path
            
    # 如果本地文件不存在，使用GitHub路径
    print(f"未找到本地文件{file_name}，尝试从GitHub加载...")
    github_base = "https://raw.githubusercontent.com/Newzil-git/Financial-Risk-Management/main/FR%20Code/"
    if file_name.startswith("4.1"):
        url = f"{github_base}Part%202/{file_name.replace(' ', '%20')}"
    else:
        url = f"{github_base}Part%201/{file_name.replace(' ', '%20')}"
    print(f"使用URL：{url}")
    return url

def load_stock_data(file_name, skiprows=3, names=None):
    """
    加载股票数据
    
    Parameters:
    -----------
    file_name : str
        数据文件名
    skiprows : int, optional
        跳过的行数，默认为3
    names : list, optional
        列名列表，默认为["Date", "Open", "High", "Low", "Close", "Volume"]
        
    Returns:
    --------
    pd.DataFrame
        处理后的股票数据
    """
    if names is None:
        names = ["Date", "Open", "High", "Low", "Close", "Volume"]
    
    path = get_data_path(file_name)
    print(f"正在从{path}加载数据...")
    
    # 如果是GitHub URL，不需要skiprows
    if path.startswith("http"):
        skiprows = 0
        
    try:
        df = pd.read_csv(path, skiprows=skiprows, names=names)
        print(f"数据加载成功，共{len(df)}行")
    except Exception as e:
        print(f"Error reading file {file_name} from path {path}")
        print(f"Error details: {str(e)}")
        raise
    
    # 统一列名（将 "Close" 重命名为 "Adj Close"）
    if "Close" in df.columns:
        df.rename(columns={'Close': 'Adj Close'}, inplace=True)
    
    # 转换日期格式
    if "Date" in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
        except:
            df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def load_macro_data(file_name):
    """
    加载宏观经济数据
    
    Parameters:
    -----------
    file_name : str
        数据文件名
        
    Returns:
    --------
    pd.DataFrame
        处理后的宏观经济数据
    """
    path = get_data_path(file_name)
    print(f"正在从{path}加载数据...")
    
    try:
        df = pd.read_csv(path)
        print(f"数据加载成功，共{len(df)}行")
    except Exception as e:
        print(f"Error reading file {file_name} from path {path}")
        print(f"Error details: {str(e)}")
        raise
    
    # 转换日期格式
    if "DATE" in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    return df 