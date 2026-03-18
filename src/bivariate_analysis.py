import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import f_oneway
import numpy as np

def bivariate_analysis():
    """Bivariate Analysis Module"""
    # 双变量分析模块
    
    # Check if data is loaded
    # 检查数据是否已加载
    if 'df' not in st.session_state:
        st.error("Please run data preprocessing step first!")
        return
    
    df = st.session_state.df
    
    st.title("🔗 Paddy Data Bivariate Analysis")
    # 水稻数据双变量分析
    
    # Create tabs to organize different analyses
    # 创建选项卡组织不同的分析
    tab1, tab2, tab3 = st.tabs(["🌾 Variety & Yield", "💧 Nursery Method & Yield", "🗺️ Region & Yield"])
    
    with tab1:
        st.header("Variety and Yield Relationship Analysis")
        # 品种与产量关系分析
        
        # Variety and yield descriptive statistics
        # 品种与产量描述性统计
        st.subheader("Yield Statistics by Variety")
        variety_yield_stats = df.groupby('Variety')['Paddy yield(in Kg)'].agg([
            ('Sample Count', 'count'),
            ('Average Yield', 'mean'),
            ('Yield Median', 'median'),
            ('Yield Std Dev', 'std'),
            ('Min Yield', 'min'),
            ('Max Yield', 'max')
        ]).round(2)
        
        # Sort by average yield
        # 按平均产量排序
        variety_yield_stats = variety_yield_stats.sort_values('Average Yield', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(variety_yield_stats, use_container_width=True)
        
        with col2:
            st.subheader("Key Findings")
            best_variety = variety_yield_stats.index[0]
            best_yield = variety_yield_stats.iloc[0]['Average Yield']
            worst_variety = variety_yield_stats.index[-1]
            worst_yield = variety_yield_stats.iloc[-1]['Average Yield']
            
            st.metric(
                "Highest Yield Variety",
                f"{best_variety}",
                f"{best_yield:.1f}"
            )
            st.metric(
                "Most Stable Variety",
                f"{variety_yield_stats.loc[variety_yield_stats['Yield Std Dev'].idxmin()].name}",
                f"Std Dev: {variety_yield_stats['Yield Std Dev'].min():.1f}"
            )
        
        # Box plot showing yield distribution by variety
        # 箱线图展示品种产量分布
        st.subheader("Yield Distribution Box Plot by Variety")
        fig_box = px.box(
            df, 
            x='Variety', 
            y='Paddy yield(in Kg)',
            title="Yield Distribution Comparison by Variety",
            labels={'Variety': 'Variety', 'Paddy yield(in Kg)': 'Yield'}
        )
        fig_box.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # ANOVA analysis
        # 方差分析（ANOVA）
        st.subheader("Significance Test for Yield Differences Between Varieties")
        # 品种间产量差异显著性检验
        varieties = df['Variety'].unique()
        variety_groups = [df[df['Variety'] == var]['Paddy yield(in Kg)'] for var in varieties]
        
        f_stat, p_value = f_oneway(*variety_groups)
        
        col1, col2 = st.columns(2)
        col1.metric("F Statistic", f"{f_stat:.4f}")
        col2.metric("P Value", f"{p_value:.4f}")
        
        if p_value < 0.05:
            st.success("✅ **Statistical Significance**: Yield differences between varieties are statistically significant (p < 0.05)")
        else:
            st.warning("❌ **Statistical Significance**: Yield differences between varieties are not statistically significant (p ≥ 0.05)")
        
        # Add variety yield comparison bar chart
        # 添加品种产量对比条形图
        st.subheader("Average Yield Comparison by Variety")
        avg_yield_by_variety = df.groupby('Variety')['Paddy yield(in Kg)'].mean().sort_values(ascending=True)
        
        fig_bar = px.bar(
            x=avg_yield_by_variety.values,
            y=avg_yield_by_variety.index,
            orientation='h',
            title="Average Yield Ranking by Variety",
            labels={'x': 'Average Yield', 'y': 'Variety'},
            color=avg_yield_by_variety.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.header("Nursery Method and Yield Relationship Analysis")
        # 育苗方式与产量关系分析
        
        # Nursery method statistics
        # 育苗方式统计
        seedling_counts = df['Nursery'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Nursery Method Distribution")
            # 育苗方式分布
            fig_pie = px.pie(
                names=seedling_counts.index,
                values=seedling_counts.values,
                title="Nursery Method Ratio"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Yield Comparison Statistics")
            # 产量对比统计
            seedling_stats = df.groupby('Nursery')['Paddy yield(in Kg)'].agg([
                ('Average Yield', 'mean'),
                ('Yield Median', 'median'),
                ('Yield Std Dev', 'std')
            ]).round(2)
            st.dataframe(seedling_stats, use_container_width=True)
        
        with col3:
            st.subheader("Key Metrics")
            # 关键指标
            dry_yield = seedling_stats.loc['dry', 'Average Yield']
            wet_yield = seedling_stats.loc['wet', 'Average Yield']
            yield_diff = wet_yield - dry_yield
            yield_diff_percent = (yield_diff / dry_yield) * 100
            
            st.metric(
                "Wet vs Dry Yield Increase",
                f"{yield_diff:.1f}",
                f"{yield_diff_percent:.1f}%"
            )
        
        # Box plot comparison
        # 箱线图对比
        st.subheader("Yield Distribution Comparison by Nursery Method")
        fig_box_seedling = px.box(
            df, 
            x='Nursery', 
            y='Paddy yield(in Kg)',
            title="Yield Distribution by Nursery Method",
            labels={'Nursery': 'Nursery Method', 'Paddy yield(in Kg)': 'Yield'}
        )
        st.plotly_chart(fig_box_seedling, use_container_width=True)
        
        # Violin plot showing distribution density
        # 小提琴图展示分布密度
        st.subheader("Yield Density Distribution by Nursery Method")
        fig_violin = px.violin(
            df, 
            x='Nursery', 
            y='Paddy yield(in Kg)',
            box=True,
            points="all",
            title="Yield Density Distribution by Nursery Method (Violin Plot)",
            labels={'Nursery': 'Nursery Method', 'Paddy yield(in Kg)': 'Yield'}
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # T-test
        # T检验
        st.subheader("Statistical Significance Test")
        # 统计显著性检验
        dry_yield_data = df[df['Nursery'] == 'dry']['Paddy yield(in Kg)']
        wet_yield_data = df[df['Nursery'] == 'wet']['Paddy yield(in Kg)']
        
        t_stat, p_value = stats.ttest_ind(dry_yield_data, wet_yield_data)
        
        col1, col2 = st.columns(2)
        col1.metric("T Statistic", f"{t_stat:.4f}")
        col2.metric("P Value", f"{p_value:.4f}")
        
        if p_value < 0.05:
            st.success("✅ **Statistical Significance**: Yield differences between nursery methods are statistically significant")
            better_method = "wet" if wet_yield > dry_yield else "dry"
            st.info(f"📊 **Conclusion**: {better_method} nursery method has significantly higher average yield")
        else:
            st.warning("❌ **Statistical Significance**: Yield differences between nursery methods are not statistically significant")
    
    with tab3:
        st.header("Region and Yield Relationship Analysis")
        # 地区与产量关系分析
        
        # Region yield statistics
        # 地区产量统计
        st.subheader("Yield Statistics by Region")
        location_yield_stats = df.groupby('Agriblock')['Paddy yield(in Kg)'].agg([
            ('Sample Count', 'count'),
            ('Average Yield', 'mean'),
            ('Yield Median', 'median'),
            ('Yield Std Dev', 'std'),
            ('Min Yield', 'min'),
            ('Max Yield', 'max')
        ]).round(2)
        
        # Sort by average yield
        # 按平均产量排序
        location_yield_stats = location_yield_stats.sort_values('Average Yield', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(location_yield_stats, use_container_width=True)
        
        with col2:
            st.subheader("Region Ranking")
            # 地区排名
            best_location = location_yield_stats.index[0]
            best_location_yield = location_yield_stats.iloc[0]['Average Yield']
            worst_location = location_yield_stats.index[-1]
            worst_location_yield = location_yield_stats.iloc[-1]['Average Yield']
            
            st.metric(
                "Highest Yield Region",
                f"{best_location}",
                f"{best_location_yield:.1f}"
            )
            st.metric(
                "Least Yield Variation Region",
                f"{location_yield_stats.loc[location_yield_stats['Yield Std Dev'].idxmin()].name}",
                f"Std Dev: {location_yield_stats['Yield Std Dev'].min():.1f}"
            )
        
        # Region average yield bar chart
        # 地区平均产量条形图
        st.subheader("Average Yield Comparison by Region")
        avg_yield_by_location = df.groupby('Agriblock')['Paddy yield(in Kg)'].mean().sort_values(ascending=True)
        
        fig_location_bar = px.bar(
            x=avg_yield_by_location.values,
            y=avg_yield_by_location.index,
            orientation='h',
            title="Average Yield Ranking by Region",
            labels={'x': 'Average Yield', 'y': 'Region'},
            color=avg_yield_by_location.values,
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig_location_bar, use_container_width=True)
        
        # Region yield distribution box plot
        # 地区产量分布箱线图
        st.subheader("Yield Distribution by Region")
        fig_location_box = px.box(
            df, 
            x='Agriblock', 
            y='Paddy yield(in Kg)',
            title="Yield Distribution Comparison by Region",
            labels={'Agriblock': 'Region', 'Paddy yield(in Kg)': 'Yield'}
        )
        fig_location_box.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_location_box, use_container_width=True)
        
        # Region yield variation analysis
        # 地区产量变异分析
        st.subheader("Within-Region Yield Variation Analysis")
        
        # Calculate coefficient of variation for each region (std dev/mean)
        # 计算每个地区的变异系数（标准差/均值）
        location_cv = (df.groupby('Agriblock')['Paddy yield(in Kg)'].std() / df.groupby('Agriblock')['Paddy yield(in Kg)'].mean()).round(3)
        location_cv_sorted = location_cv.sort_values(ascending=False)
        
        fig_cv = px.bar(
            x=location_cv_sorted.values,
            y=location_cv_sorted.index,
            orientation='h',
            title="Yield Coefficient of Variation (CV) by Region",
            labels={'x': 'Coefficient of Variation (Std Dev/Mean)', 'y': 'Region'},
            color=location_cv_sorted.values,
            color_continuous_scale='rdylbu_r'
        )
        st.plotly_chart(fig_cv, use_container_width=True)
        
        st.info("📊 **Coefficient of Variation Interpretation**: Higher CV indicates greater yield fluctuation within the region, poorer stability")
        # 变异系数解读
        
        # ANOVA for regional differences
        # 地区间差异的方差分析
        st.subheader("Significance Test for Yield Differences Between Regions")
        # 地区间产量差异显著性检验
        locations = df['Agriblock'].unique()
        location_groups = [df[df['Agriblock'] == loc]['Paddy yield(in Kg)'] for loc in locations]
        
        f_stat_loc, p_value_loc = f_oneway(*location_groups)
        
        col1, col2 = st.columns(2)
        col1.metric("F Statistic", f"{f_stat_loc:.4f}")
        col2.metric("P Value", f"{p_value_loc:.4f}")
        
        if p_value_loc < 0.05:
            st.success("✅ Statistical Significance: Yield differences between regions are statistically significant")
            st.info("🌍 Geographic Factor Impact: Regional environmental conditions significantly affect rice yield")
        else:
            st.warning("❌ **Statistical Significance**: Yield differences between regions are not statistically significant")

# Main program
# 主程序
if __name__ == "__main__":
    bivariate_analysis()