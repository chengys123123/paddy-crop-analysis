import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np

def pca_preprocessing():
    """PCA data preprocessing"""
    
    # 检查基础数据是否已加载
    # Check whether the basic data has been loaded
    if 'df' not in st.session_state:
        st.error("Please run data preprocessing step to load basic data first!")
        return
    
    df = st.session_state.df.copy()
    
    st.title("🔧 PCA data preprocessing")
    st.markdown("Preparing data for PCA analysis: one-hot coding of categorical variables + normalization of numerical variables")
    
    # 分类变量列表
    # List of categorical variables
    categorical_columns = [
        'Agriblock', 'Variety', 'Soil Types', 'Nursery', 'Wind Direction_D1_D30', 'Wind Direction_D31_D60', 'Wind Direction_D61_D90', 'Wind Direction_D91_D120'
    ]
    
    # 只保留实际存在的分类变量
    # Keep only categorical variables that actually exist
    existing_categorical = [col for col in categorical_columns if col in df.columns]
    
    # 显示处理前信息
    # Display pre-process information
    st.subheader("Data information before processing")
    st.write(f"- Data shape: {df.shape}")
    st.write(f"- Categorical variable: {len(existing_categorical)}")
    
    # 1. 独热编码处理分类变量
    # 1. One-hot coding deals with categorical variables
    if existing_categorical:
        st.subheader("1. Categorical variable one-hot coding")
        df = pd.get_dummies(df, columns=existing_categorical, drop_first=True)
        st.success(f"✅ Complete independent hot coding and add feature column")
    
    # 2. 标准化数值变量
    # 2. Standardized numerical variables
    st.subheader("2. Normalization of numerical variables")
    
    # 识别数值列（排除目标变量Yield）
    # Identify numeric columns (exclude the target variable Yield)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Paddy yield(in Kg)' in numeric_columns:
        numeric_columns.remove('Paddy yield(in Kg)')
    
    if numeric_columns:
        # 标准化
        # Standardization
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        st.success(f"✅ Completed{len(numeric_columns)}numerical variables normalization")
    
    # 显示处理后信息
    # Display post-processing information
    # Display post-processing information
    st.subheader("Processed data information")
    st.write(f"- Data shape: {df.shape}")
    st.write(f"- Total number of features: {df.shape[1]}")
    
    # 数据预览
    # Data preview
    # Data preview
    st.subheader("Preview of processed data")
    st.dataframe(df.head())
    
    # 保存到session state
    # Save to session state
    st.session_state.df_pca = df
    st.success("✅ PCA data preprocessing has been completed!")
    
    # 下载按钮
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download PCA preprocessed data",
        data=csv,
        file_name="paddy_pca_preprocessed.csv",
        mime="text/csv"
    )

# 主程序
# Main program
if __name__ == "__main__":
    pca_preprocessing()