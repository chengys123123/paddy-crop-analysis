# data_preprocessing.py
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data():
    """Load the rice data set"""
    return pd.read_csv(r"D:\Desktop\Data Visualization\final_presentation\paddydataset.csv")

def basic_quality_check(df):
    """Basic data quality check"""
    st.subheader("📊 Data quality check")
    
    # 缺失值统计
    # Statistics of missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Number of missing': missing, 'Missing proportion %': missing_percent})
    missing_df = missing_df[missing_df['Number of missing'] > 0]
    
    if len(missing_df) > 0:
        st.write("**Missing values:**")
        st.dataframe(missing_df)
    else:
        st.success("✅ No missing values")
    
    # 数据类型
    # Data type
    st.write("Data type:")
    st.write(pd.DataFrame({
        'Column name': df.columns,
        'Type': df.dtypes,
        'Unique value': [df[col].nunique() for col in df.columns]
    }))

def handle_outliers(df):
    """Outlier handling"""
    st.subheader("📈 Outlier handling")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if st.checkbox("Detect outliers"):
        outlier_counts = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            outlier_counts[col] = len(outliers)
        
        st.write("Number of outliers:", outlier_counts)

def validate_data(df):
    """Data validation"""
    st.subheader("✅ Data validation")
    
    checks = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 检查负值
    # Check for negative values
    for col in numeric_cols:
        if (df[col] < 0).any():
            checks.append(f"❌ {col} Has a negative value")
        else:
            checks.append(f"✅ {col} No negative values")
    
    # 检查重复
    # Check for duplicates
    if df.duplicated().sum() > 0:
        checks.append(f"❌ There are {df.duplicated().sum()} duplicate rows")
    else:
        checks.append("✅ No duplicate rows")
    
    for check in checks:
        st.write(check)

def data_preprocessing_page():
    """Main function of data preprocessing page"""
    st.title("🌾 Paddy data preprocessing")
    
    # 加载数据
    # Load data
    df = load_data()
    st.write(f"Original data: {df.shape[0]} Lines × {df.shape[1]} Columns")
    
    # 数据质量检查
    # Data quality check
    basic_quality_check(df)
    
    # 异常值处理
    # Outlier handling
    handle_outliers(df)
    
    # 数据验证
    # Data validation
    validate_data(df)
    
    # 基础清洗
    # Basic cleaning
    df_clean = df.dropna().drop_duplicates()
    
    # 结果显示
    # The results show
    st.subheader("🎯 Processing results")
    col1, col2 = st.columns(2)
    col1.metric("Original data", len(df))
    col2.metric("Data after cleaning", len(df_clean))
    
    # 保存数据
    # Save the data
    if st.button("Download the cleaned data"):
        df_clean.to_csv("paddy_cleaned.csv", index=False)
        st.success("Data has been saved")
    
    # 存储供后续使用
    # Store for later up
    st.session_state.df = df_clean
    st.success("✅ The data has been loaded into the application and can be switched to the univariate analysis module")
    
    return df_clean

# 保留单独运行功能
# Retain the function of running separately
if __name__ == "__main__":
    data_preprocessing_page()