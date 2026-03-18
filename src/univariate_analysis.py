import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def univariate_analysis():
    """Univariate Analysis Module"""
    # 单变量分析模块
    
    # Check if data is loaded
    # 检查数据是否已加载
    if 'df' not in st.session_state:
        st.error("Please run data preprocessing step first!")
        st.info("Please switch to 'Data Preprocessing' module, load and process data before univariate analysis.")
        return
    
    df = st.session_state.df
    
    st.title("📊 Paddy Data Univariate Analysis")
    # 水稻数据单变量分析
    
    # Create tabs to organize different analyses
    # 创建选项卡组织不同的分析
    tab1, tab2, tab3 = st.tabs(["🌾 Variety Distribution", "🗺️ Region Distribution", "📈 Yield Distribution"])
    
    with tab1:
        st.header("Paddy Variety Distribution Analysis")
        # 水稻品种分布分析
        
        # Calculate variety frequency and percentage
        # 计算品种频数和百分比
        variety_counts = df['Variety'].value_counts()
        variety_percent = (variety_counts / len(df) * 100).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Variety Statistics Table")
            # 品种统计表
            variety_df = pd.DataFrame({
                'Variety': variety_counts.index,
                'Count': variety_counts.values,
                'Percentage(%)': variety_percent.values
            })
            st.dataframe(variety_df, use_container_width=True)
        
        with col2:
            st.subheader("Variety Distribution Pie Chart")
            # 品种分布饼图
            fig_pie = px.pie(
                names=variety_counts.index,
                values=variety_counts.values,
                title="Paddy Variety Distribution Ratio"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Horizontal bar chart
        # 水平条形图
        st.subheader("Variety Quantity Distribution")
        fig_bar = px.bar(
            x=variety_counts.values,
            y=variety_counts.index,
            orientation='h',
            title="Sample Count by Variety",
            labels={'x': 'Sample Count', 'y': 'Variety'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.header("Region Distribution Analysis")
        # 地区分布分析
        
        # Calculate region frequency
        # 计算地区频数
        region_counts = df['Agriblock'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Region Statistics Table")
            # 地区统计表
            region_df = pd.DataFrame({
                'Agriblock': region_counts.index,
                'Sample Count': region_counts.values
            })
            st.dataframe(region_df, use_container_width=True)
        
        with col2:
            st.subheader("Region Distribution Chart")
            # 地区分布图
            fig_region_bar = px.bar(
                x=region_counts.values,
                y=region_counts.index,
                orientation='h',
                title="Sample Count Distribution by Agriblock",
                labels={'x': 'Sample Count', 'y': 'Agriblock'}
            )
            st.plotly_chart(fig_region_bar, use_container_width=True)
        
        # Add descriptive statistics for region distribution
        # 添加地区分布的描述性统计
        st.subheader("Region Distribution Statistics")
        st.write(f"**Total Agriblocks:** {len(region_counts)}")
        st.write(f"**Agriblock with Most Samples:** {region_counts.index[0]} ({region_counts.iloc[0]} samples)")
        st.write(f"**Agriblock with Least Samples:** {region_counts.index[-1]} ({region_counts.iloc[-1]} samples)")
    
    with tab3:
        st.header("Yield Distribution Analysis")
        # 产量分布分析
        
        # Basic statistics
        # 基本统计量
        yield_stats = {
            'Mean': df['Paddy yield(in Kg)'].mean(),
            'Median': df['Paddy yield(in Kg)'].median(),
            'Std Dev': df['Paddy yield(in Kg)'].std(),
            'Min': df['Paddy yield(in Kg)'].min(),
            'Max': df['Paddy yield(in Kg)'].max(),
            'Skewness': df['Paddy yield(in Kg)'].skew()
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Yield Statistics")
            # 产量统计量
            for stat, value in yield_stats.items():
                st.metric(
                    label=stat,
                    value=f"{value:.2f}" if stat != 'Skewness' else f"{value:.4f}"
                )
        
        with col2:
            st.subheader("Yield Distribution Histogram")
            # 产量分布直方图
            fig_hist = px.histogram(
                df, 
                x='Paddy yield(in Kg)',
                nbins=30,
                title="Yield Distribution Histogram",
                labels={'Paddy yield(in Kg)': 'Yield'}
            )
            # Add mean and std dev lines
            # 添加均值和标准差线
            fig_hist.add_vline(
                x=yield_stats['Mean'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {yield_stats['Mean']:.2f}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Box plot to show outliers
        # 箱线图显示异常值
        st.subheader("Yield Box Plot")
        fig_box = px.box(
            df, 
            y='Paddy yield(in Kg)',
            title="Yield Distribution Box Plot",
            labels={'Paddy yield(in Kg)': 'Yield'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Distribution characteristics analysis
        # 分布特征分析
        st.subheader("Distribution Characteristics Analysis")
        if yield_stats['Skewness'] > 0.5:
            st.info("📊 **Distribution Characteristics:** Yield distribution shows right skew (positive skew), most samples have low yield, few samples have high yield")
        elif yield_stats['Skewness'] < -0.5:
            st.info("📊 **Distribution Characteristics:** Yield distribution shows left skew (negative skew), most samples have high yield")
        else:
            st.info("📊 **Distribution Characteristics:** Yield distribution is close to normal distribution")
        
        # Identify outliers
        # 识别异常值
        Q1 = df['Paddy yield(in Kg)'].quantile(0.25)
        Q3 = df['Paddy yield(in Kg)'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['Paddy yield(in Kg)'] < lower_bound) | (df['Paddy yield(in Kg)'] > upper_bound)]
        st.write(f"**Outlier Count:** {len(outliers)} samples")

# Main program
# 主程序
if __name__ == "__main__":
    univariate_analysis()