import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np

def pca_analysis():
    """Principal Component Analysis"""
    
    # Check if PCA data is prepared
    # 检查PCA数据是否已准备
    if 'df_pca' not in st.session_state:
        st.error("Please run PCA data preprocessing step first!")
        return
    
    df_pca = st.session_state.df_pca
    
    st.title("🔍 Principal Component Analysis (PCA)")
    
    # Prepare data (exclude target variable)
    # 准备数据（排除目标变量）
    if 'Paddy yield(in Kg)' in df_pca.columns:
        features_df = df_pca.drop('Paddy yield(in Kg)', axis=1)
        target = df_pca['Paddy yield(in Kg)']
    else:
        features_df = df_pca
        target = None
    
    # Step 2: Check variable correlation
    # 步骤2：检查变量相关性
    st.header("📊 Variable Correlation Analysis")
    
    # Calculate correlation matrix
    # 计算相关系数矩阵
    corr_matrix = features_df.corr().round(3)
    
    # Correlation heatmap
    # 相关性热力图
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Variable Correlation Heatmap"
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Correlation analysis conclusion
    # 相关性分析结论
    high_corr_count = (corr_matrix.abs() > 0.7).sum().sum() - len(corr_matrix)
    st.write(f"**Correlation Analysis**: Found {high_corr_count} highly correlated relationships (|r|>0.7)")
    
    # Step 3: Perform PCA
    # 步骤3：执行PCA
    st.header("🔄 PCA Execution and Results")
    
    # Perform PCA
    # 执行PCA
    pca = PCA()
    principal_components = pca.fit_transform(features_df)
    
    # Calculate explained variance ratio
    # 计算方差解释率
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Step 4: Select number of principal components
    # 步骤4：选择主成分数量
    # Find number of components explaining 85% variance
    # 找到达到85%方差的主成分数量
    n_components_85 = np.argmax(cumulative_variance >= 0.85)
    
    # Scree plot
    # 碎石图
    fig_scree = go.Figure()
    
    # Add variance for each principal component
    # 添加各主成分方差
    fig_scree.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(len(explained_variance))],
        y=explained_variance,
        name="Variance per Principal Component",
        marker_color='lightblue'
    ))
    
    # Add cumulative variance line
    # 添加累计方差线
    fig_scree.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
        y=cumulative_variance,
        name="Cumulative Variance",
        line=dict(color='red', width=3),
        yaxis='y2'
    ))
    
    # Add 85% variance line
    # 添加85%方差线
    fig_scree.add_hline(
        y=0.85, 
        line_dash="dash", 
        line_color="green",
        annotation_text="85% Variance Line"
    )
    
    fig_scree.update_layout(
        title="PCA Scree Plot and Cumulative Variance Contribution Rate",
        xaxis_title="Principal Component",
        yaxis=dict(title="Variance per Principal Component", side='left'),
        yaxis2=dict(title="Cumulative Variance", side='right', overlaying='y'),
        showlegend=True
    )
    
    st.plotly_chart(fig_scree, use_container_width=True)
    
    # Principal component selection result
    # 主成分选择结果
    st.write(f"**Principal Component Selection**: First {n_components_85} principal components explain {cumulative_variance[n_components_85-1]:.1%} of variance")
    
    # Step 5: Interpret principal components
    # 步骤5：解释主成分
    st.header("📈 Principal Component Interpretation")
    
    # Create loading matrix
    # 创建载荷矩阵
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Show main contributing variables for first 3 principal components
    # 显示前3个主成分的主要贡献变量
    for i in range(min(3, len(pca.components_))):
        # Get loadings for i-th principal component
        # 获取第i个主成分的载荷
        pc_loadings = pd.Series(loadings[:, i], index=features_df.columns)
        # Sort by absolute value and take top 5
        # 按绝对值排序并取前5个
        top_vars = pc_loadings.abs().sort_values(ascending=False).head(5)
        
        st.subheader(f"Principal Component {i+1} (Explained Variance: {explained_variance[i]:.2%})")
        
        # Create horizontal bar chart showing main variables
        # 创建水平条形图显示主要变量
        fig_loadings = px.bar(
            x=top_vars.values,
            y=top_vars.index,
            orientation='h',
            title=f"PC{i+1} - Main Contributing Variables",
            labels={'x': 'Loading Absolute Value', 'y': 'Variable'}
        )
        st.plotly_chart(fig_loadings, use_container_width=True)
    
    # Step 6: Visualize principal components
    # 步骤6：可视化主成分
    st.header("🎨 Principal Component Visualization")
    
    # Create PCA result DataFrame
    # 创建PCA结果DataFrame
    pca_df = pd.DataFrame(
        principal_components[:, :2], 
        columns=['PC1', 'PC2']
    )
    
    # Add target variable to DataFrame if available
    # 如果有目标变量，添加到DataFrame中
    if target is not None:
        pca_df['Paddy yield(in Kg)'] = target.values
    
    # Try to add categorical variables for grouping colors
    # 尝试添加分类变量用于分组着色
    original_df = st.session_state.df  # Original data (contains categorical variables)
    for col in ['Variety', 'Soil Types', 'Location']:
        if col in original_df.columns and len(pca_df) == len(original_df):
            pca_df[col] = original_df[col].values
            break
    
    # Plot PC1 vs PC2 scatter plot
    # 绘制PC1 vs PC2散点图
    color_col = None
    for col in ['Paddy yield(in Kg)', 'Variety', 'Soil Types', 'Location']:
        if col in pca_df.columns:
            color_col = col
            break
    
    fig_pca = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color=color_col,
        title="Principal Component Analysis: PC1 vs PC2",
        labels={
            'PC1': f'PC1 ({explained_variance[0]:.1%} variance)',
            'PC2': f'PC2 ({explained_variance[1]:.1%} variance)'
        }
    )
    
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Step 7: Relationship between principal components and target variable
    # 步骤7：主成分与目标变量的关系
    if target is not None:
        st.header("🔗 Principal Components and Yield Relationship")
        
        # Create DataFrame of principal components with target
        # 创建主成分与产量的相关性DataFrame
        pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
        pca_with_target = pd.DataFrame(principal_components, columns=pca_columns)
        pca_with_target['Paddy yield(in Kg)'] = target.values
        
        # Calculate correlation between principal components and yield
        # 计算主成分与产量的相关性
        corr_with_yield = pca_with_target.corr()['Paddy yield(in Kg)'].drop('Paddy yield(in Kg)')
        
        # Display correlations
        # 显示相关性
        st.subheader("Correlation Between Principal Components and Yield")
        for i, (pc, corr) in enumerate(corr_with_yield.items()):
            st.write(f"- {pc}: r = {corr:.3f}")
        
        # Visualize relationship with yield
        # 可视化与产量的关系
        if len(corr_with_yield) > 0:
            # Find principal component with strongest correlation to yield
            # 找到与产量相关性最强的主成分
            strongest_pc = corr_with_yield.abs().idxmax()
            strongest_corr = corr_with_yield[strongest_pc]

            # Get principal component index
            # 获取主成分索引
            pc_index = int(strongest_pc[2:]) - 1  # Extract number from "PC1", subtract 1 to get index 0
            
            # Create scatter plot
            # 创建散点图
            scatter_data = pd.DataFrame({
                'PC': principal_components[:, pc_index],
                'Paddy yield(in Kg)': target
            })
            
            fig_yield = px.scatter(
                scatter_data,
                x='PC',
                y='Paddy yield(in Kg)',
                title=f"{strongest_pc} vs Yield (r = {strongest_corr:.3f})",
                labels={'PC': strongest_pc, 'Paddy yield(in Kg)': 'Yield'}
            )
            
            # Manually add trend line
            # 手动添加趋势线
            z = np.polyfit(principal_components[:, pc_index], target, 1)
            p = np.poly1d(z)
            
            # Add trend line to chart
            # 添加趋势线到图表
            x_range = np.linspace(
                principal_components[:, pc_index].min(),
                principal_components[:, pc_index].max(),
                100
            )
            fig_yield.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Trend Line'
                )
            )
            st.plotly_chart(fig_yield, use_container_width=True)
    
    # Analysis summary
    # 分析总结
    st.header("💡 Analysis Summary")
    
    st.markdown(f"""
    ### PCA Analysis Results
    
    **Main Findings:**
    - First **{n_components_85} principal components** explain **{cumulative_variance[n_components_85-1]:.1%}** of total variance
    - **PC1** and **PC2** explain **{explained_variance[0]:.1%}** and **{explained_variance[1]:.1%}** of variance respectively
    - Found **{high_corr_count}** highly correlated relationships in data
    
    **Application Suggestions:**
    - Can use first {n_components_85} principal components for dimensionality reduction
    - First two principal components can be used for data visualization
    - Identified main contributing variables can be used for feature engineering
    """)

if __name__ == "__main__":
    pca_analysis()