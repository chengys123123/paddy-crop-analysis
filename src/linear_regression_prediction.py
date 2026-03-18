import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression_prediction():
    """Linear Regression Yield Prediction"""
    
    if 'df_pca' not in st.session_state:
        st.error("Please run PCA data preprocessing step first!")
        return
    
    df_pca = st.session_state.df_pca
    
    st.title("📈 Linear Regression Yield Prediction")
    
    # Prepare data
    # 准备数据
    if 'Paddy yield(in Kg)' not in df_pca.columns:
        st.error("Yield column (Paddy yield(in Kg)) not found in dataset!")
        return
    
    # Use PCA features
    # 使用PCA特征
    X = df_pca.drop('Paddy yield(in Kg)', axis=1)
    y = df_pca['Paddy yield(in Kg)']
    
    # Data split
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    # 预测
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Display results
    # 显示结果
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Set Sample Count", len(X_test))
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.4f}")
    
    # Visualization 1: Predicted vs actual values
    # 可视化1: 预测vs真实值
    st.subheader("Prediction Performance Chart")
    # 预测效果图
    fig1 = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Yield', 'y': 'Predicted Yield'},
        title=f"Predicted vs Actual Values (R² = {r2:.3f})"
    )
    # Add diagonal line
    # 添加对角线
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig1.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(dash='dash', color='red'),
            name='Perfect Prediction Line'
        )
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Visualization 2: Feature importance
    # 可视化2: 特征重要性
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Importance': np.abs(model.coef_)
    }).sort_values('Importance', ascending=False).head(10)
    
    fig2 = px.bar(
        importance, 
        x='Importance', y='Feature',
        orientation='h',
        title="Top 10 Important Features"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Visualization 3: Residual analysis
    # 可视化3: 残差分析
    st.subheader("Residual Analysis")
    residuals = y_test - y_pred
    fig3 = px.histogram(
        x=residuals,
        nbins=30,
        title="Residual Distribution",
        labels={'x': 'Residual'}
    )
    fig3.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Dynamic training demonstration (simplified version)
    # 动态训练演示（简化版）
    st.subheader("Model Training Demonstration")
    # 模型训练演示
    
    # Create dynamic chart container
    # 创建动态图表容器
    dynamic_chart = st.empty()
    
    # Simulate training process
    # 模拟训练过程
    for i in range(1, 101, 20):
        # Update progress
        # 更新进度
        
        # Create simple dynamic effect
        # 创建简单动态效果
        if i == 100:
            # Final result
            # 最终结果
            fig_dynamic = go.Figure()
            fig_dynamic.add_trace(
                go.Scatter(x=y_test, y=y_pred, mode='markers', name='Prediction Points')
            )
            fig_dynamic.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', line=dict(dash='dash', color='red'),
                    name='Perfect Prediction Line'
                )
            )
            fig_dynamic.update_layout(
                title="Final Prediction Result",
                xaxis_title="Actual Yield",
                yaxis_title="Predicted Yield"
            )
        else:
            # Intermediate state during training
            # 训练过程中的中间状态
            fig_dynamic = go.Figure()
            # Simulate partial data points
            # 模拟部分数据点
            sample_size = int(len(y_test) * (i/100))
            fig_dynamic.add_trace(
                go.Scatter(
                    x=y_test[:sample_size], 
                    y=y_pred[:sample_size], 
                    mode='markers', 
                    name=f'Trained {i}%'
                )
            )
            fig_dynamic.update_layout(
                title=f"Training... {i}%",
                xaxis_title="Actual Yield",
                yaxis_title="Predicted Yield"
            )
        
        dynamic_chart.plotly_chart(fig_dynamic, use_container_width=True)
        
        # Brief pause to show dynamic effect
        # 短暂暂停以显示动态效果
    
    # Prediction example
    # 预测示例
    st.subheader("Yield Prediction")
    
    # Select top 3 important features for example
    # 选择前3个重要特征用于示例
    top_features = importance.head(3)['Feature'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Input Feature Values:")
        # 输入特征值
        input_vals = {}
        for feature in top_features:
            min_val = X[feature].min()
            max_val = X[feature].max()
            mean_val = X[feature].mean()
            input_vals[feature] = st.slider(
                feature, min_val, max_val, mean_val
            )
    
    with col2:
        if st.button("Predict Yield"):
            # 预测产量
            # Prepare input data
            # 准备输入数据
            input_data = [X[col].mean() for col in X.columns]  # Initialize with mean values
            for feature, value in input_vals.items():
                if feature in X.columns:
                    idx = X.columns.get_loc(feature)
                    input_data[idx] = value
            
            prediction = model.predict([input_data])[0]
            st.success(f"Predicted Yield: **{prediction:.2f}**")

if __name__ == "__main__":
    linear_regression_prediction()