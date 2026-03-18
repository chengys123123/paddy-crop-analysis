import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import pearsonr

def growth_stage_analysis():
    """Growth Stage Analysis Module"""
    
    # Check if data is loaded
    # 检查数据是否已加载
    if 'df' not in st.session_state:
        st.error("Please run data preprocessing step first!")
        return
    
    df = st.session_state.df
    
    st.title("🌱 Paddy Growth Stage Analysis")
    
    # Rainfall and yield relationship analysis
    # 降雨量与产量关系分析
    st.header("🌧️ Rainfall and Yield Relationship Analysis")
    
    # Assumed rainfall column names
    # 假设的降雨量字段名
    rainfall_columns = ['30DRain( in mm)', '30_50DRain( in mm)', '51_70DRain(in mm)', '71_105DRain(in mm)']  # 4 growth stage rainfall amounts
    
    # Check if columns exist
    # 检查字段是否存在
    available_rainfall_cols = [col for col in rainfall_columns if col in df.columns]
    
    if not available_rainfall_cols:
        st.error("Rainfall columns not found, please check data column names")
        return
    
    # Average rainfall by stage
    # 各阶段平均降雨量
    st.subheader("Average Rainfall by Growth Stage")
    avg_rainfall = df[available_rainfall_cols].mean()
    
    fig_rainfall = px.bar(
        x=avg_rainfall.values,
        y=avg_rainfall.index,
        orientation='h',
        title="Average Rainfall by Growth Stage",
        labels={'x': 'Average Rainfall', 'y': 'Growth Stage'}
    )
    st.plotly_chart(fig_rainfall, use_container_width=True)
    
    # Rainfall vs yield scatter plot
    # 降雨量与产量散点图
    st.subheader("Rainfall vs Yield Relationship")
    
    # Create subplots
    # 创建子图
    fig_rain_scatter = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{col} vs Yield" for col in available_rainfall_cols]
    )
    
    for i, col in enumerate(available_rainfall_cols):
        row = i // 2 + 1
        col_num = i % 2 + 1
        
        # Calculate correlation coefficient
        # 计算相关系数
        corr_coef, p_value = pearsonr(df[col], df['Paddy yield(in Kg)'])
        
        fig_rain_scatter.add_trace(
            go.Scatter(
                x=df[col],
                y=df['Paddy yield(in Kg)'],
                mode='markers',
                name=f"{col} (r={corr_coef:.3f})",
                marker=dict(size=6, opacity=0.6)
            ),
            row=row, col=col_num
        )
        
        # Add trend line
        # 添加趋势线
        z = np.polyfit(df[col], df['Paddy yield(in Kg)'], 1)
        p = np.poly1d(z)
        fig_rain_scatter.add_trace(
            go.Scatter(
                x=df[col],
                y=p(df[col]),
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2)
            ),
            row=row, col=col_num
        )
    
    fig_rain_scatter.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_rain_scatter, use_container_width=True)
    
    # Identify key rainfall periods
    # 识别关键降雨期
    st.subheader("Key Rainfall Period Identification")
    
    correlation_results = []
    for col in available_rainfall_cols:
        corr_coef, p_value = pearsonr(df[col], df['Paddy yield(in Kg)'])
        correlation_results.append({
            'Growth Stage': col,
            'Correlation Coefficient': corr_coef,
            'P Value': p_value
        })
    
    corr_df = pd.DataFrame(correlation_results)
    strongest_stage = corr_df.loc[corr_df['Correlation Coefficient'].abs().idxmax()]
    
    st.write(f"**Rainfall stage with greatest impact on yield**: {strongest_stage['Growth Stage']}")
    st.write(f"**Correlation Coefficient**: {strongest_stage['Correlation Coefficient']:.4f}")
    
    # Temperature and yield relationship analysis
    # 温度与产量关系分析
    st.header("🌡️ Temperature and Yield Relationship Analysis")
    
    # Assumed temperature column names - adjust according to actual data
    # 假设的温度字段名 - 请根据实际数据调整
    temp_columns = ['Min temp_D1_D30', 'Max temp_D1_D30', 'Min temp_D31_D60', 'Max temp_D31_D60', 'Min temp_D61_D90', 'Max temp_D61_D90', 'Min temp_D91_D120', 'Max temp_D91_D120']
    
    # Check if columns exist
    # 检查字段是否存在
    available_temp_cols = [col for col in temp_columns if col in df.columns]
    
    if not available_temp_cols:
        st.error("Temperature columns not found, please check data column names")
        return
    
    # Calculate average temperature by stage
    # 计算各阶段平均温度
    st.subheader("Average Temperature by Growth Stage")
    
    # Separate min and max temperatures
    # 分离最低和最高温度
    min_temp_cols = [col for col in available_temp_cols if 'Min' in col]
    max_temp_cols = [col for col in available_temp_cols if 'Max' in col]
    
    avg_temp_data = {}
    for i in range(min(len(min_temp_cols), len(max_temp_cols))):
        stage_num = i + 1
        avg_col_name = f'Stage{stage_num}'
        df[avg_col_name] = (df[min_temp_cols[i]] + df[max_temp_cols[i]]) / 2
        avg_temp_data[avg_col_name] = df[avg_col_name].mean()
    
    avg_temp_series = pd.Series(avg_temp_data)
    
    fig_temp = px.bar(
        x=avg_temp_series.values,
        y=avg_temp_series.index,
        orientation='h',
        title="Average Temperature by Growth Stage",
        labels={'x': 'Average Temperature (°C)', 'y': 'Growth Stage'}
    )
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Temperature vs yield scatter plot
    # 温度与产量散点图
    st.subheader("Temperature vs Yield Relationship")
    
    # Create subplots
    # 创建子图
    avg_temp_cols = [f'Stage{i+1}' for i in range(len(avg_temp_data))]
    
    fig_temp_scatter = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{col} vs Yield" for col in avg_temp_cols]
    )
    
    for i, col in enumerate(avg_temp_cols):
        if i >= 4:  # Maximum 4 subplots
            break
            
        row = i // 2 + 1
        col_num = i % 2 + 1
        
        # Calculate correlation coefficient
        # 计算相关系数
        corr_coef, p_value = pearsonr(df[col], df['Paddy yield(in Kg)'])
        
        fig_temp_scatter.add_trace(
            go.Scatter(
                x=df[col],
                y=df['Paddy yield(in Kg)'],
                mode='markers',
                name=f"{col} (r={corr_coef:.3f})",
                marker=dict(size=6, opacity=0.6)
            ),
            row=row, col=col_num
        )
        
        # Add trend line
        # 添加趋势线
        z = np.polyfit(df[col], df['Paddy yield(in Kg)'], 1)
        p = np.poly1d(z)
        fig_temp_scatter.add_trace(
            go.Scatter(
                x=df[col],
                y=p(df[col]),
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2)
            ),
            row=row, col=col_num
        )
    
    fig_temp_scatter.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_temp_scatter, use_container_width=True)
    
    # Identify suitable temperature ranges
    # 识别适宜温度范围
    st.subheader("Optimal Temperature Range Analysis")
    
    for col in avg_temp_cols:
        # Find temperature range for highest yield
        # 找到产量最高的温度范围
        high_yield = df['Paddy yield(in Kg)'].quantile(0.8)  # Top 20% as high yield
        high_yield_temp = df[df['Paddy yield(in Kg)'] >= high_yield][col]
        
        if len(high_yield_temp) > 0:
            optimal_min = high_yield_temp.min()
            optimal_max = high_yield_temp.max()
            st.write(f"**{col}**: Temperature range for high-yield samples {optimal_min:.1f}°C - {optimal_max:.1f}°C")

# Main program
# 主程序
if __name__ == "__main__":
    growth_stage_analysis()