import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

def random_forest_prediction():
    """Random Forest Prediction with Dynamic Visualization (Includes Comparison with Linear Regression)"""
    # 随机森林预测与动态可视化（包含与线性回归对比）
    
    if 'df_pca' not in st.session_state:
        st.error("Please run PCA data preprocessing step first!")
        return
    
    df_pca = st.session_state.df_pca
    
    st.title("🌲 Random Forest Yield Prediction")
    st.markdown("Using Random Forest model to predict Paddy yield, and compare with Linear Regression model")
    # 使用随机森林模型预测水稻产量，并与线性回归模型进行对比
    
    # Prepare data
    # 准备数据
    if 'Paddy yield(in Kg)' not in df_pca.columns:
        st.error("Yield column (Paddy yield(in Kg)) not found in dataset!")
        return
    
    X = df_pca.drop('Paddy yield(in Kg)', axis=1)
    y = df_pca['Paddy yield(in Kg)']
    
    # Data split
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model parameter settings
    # 模型参数设置
    st.sidebar.subheader("Model Parameters")
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
    max_depth = st.sidebar.slider("Maximum Depth", 3, 20, 10)
    
    # Train Random Forest model
    # 训练随机森林模型
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Also train Linear Regression model for comparison
    # 同时训练线性回归模型用于对比
    lr_model = LinearRegression()
    
    # Dynamic training process
    # 动态训练过程
    st.subheader("🌱 Random Forest Training Process")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Modification point 1: Change training progress to 10%, 20%, 40%, 60%, 80%, 100%
    # 修改点1: 将训练进度改为10%, 20%, 40%, 60%, 80%, 100%
    progress_steps = [10, 20, 40, 60, 80, 100]
    
    # Simulate step-by-step training
    # 模拟逐步训练
    for i, progress in enumerate(progress_steps):
        # Modification point 2: Use new progress values
        # 修改点2: 使用新的进度值
        progress_bar.progress(progress)
        status_text.text(f"Training Progress: {progress}%")
        
        # Actual training (completed at once)
        # 实际训练（一次性完成）
        if progress == 100:
            rf_model.fit(X_train, y_train)
            lr_model.fit(X_train, y_train)  # Also train linear regression
        
        # Create dynamic chart
        # 创建动态图表
        fig = go.Figure()
        
        # Simulate predictions during training
        # 模拟训练过程中的预测
        if progress == 100:
            # Modification point 3: Only show Random Forest prediction results at 100%
            # 修改点3: 在100%时只显示随机森林预测结果
            rf_pred = rf_model.predict(X_test)
            
            fig.add_trace(go.Scatter(
                x=y_test, y=rf_pred, mode='markers', 
                name='Random Forest Prediction', marker=dict(color='blue')
            ))
        else:
            # Simulate partial predictions
            # 模拟部分预测
            sample_size = int(len(X_test) * (progress/100))
            if sample_size > 0:
                # Only show Random Forest model predictions before 100%
                # 在100%之前也只显示随机森林模型的预测
                rf_pred = rf_model.predict(X_test[:sample_size]) if progress == 100 else np.random.normal(
                    y_test[:sample_size].mean(), 
                    y_test[:sample_size].std(), 
                    sample_size
                )
                
                fig.add_trace(go.Scatter(
                    x=y_test[:sample_size], y=rf_pred, 
                    mode='markers', name='Random Forest Prediction',
                    marker=dict(color='blue')
                ))
        
        # Add diagonal line
        # 添加对角线
        min_val = y_test.min()
        max_val = y_test.max()
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(dash='dash', color='red'),
            name='Perfect Prediction Line'
        ))
        
        fig.update_layout(
            title=f"Training Progress: {progress}%",
            xaxis_title="Actual Yield",
            yaxis_title="Predicted Yield"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Only pause when progress is not 100%, show results immediately at 100%
        # 只在非100%进度时暂停，100%时立即显示结果
        if progress < 100:
            time.sleep(0.5)
    
    # Final predictions
    # 最终预测
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    
    # Calculate metrics
    # 计算指标
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)
    
    # Display results
    # 显示结果
    st.subheader("📊 Model Performance Comparison")
    # 模型性能对比
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Sample Count", len(X_test))
    col2.metric("Random Forest R²", f"{rf_r2:.4f}")
    col3.metric("Linear Regression R²", f"{lr_r2:.4f}")
    col4.metric("R² Difference", f"{(rf_r2 - lr_r2):.4f}", 
                delta=f"{(rf_r2 - lr_r2):.4f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Random Forest RMSE", f"{rf_rmse:.2f}")
    col2.metric("Linear Regression RMSE", f"{lr_rmse:.2f}")
    col3.metric("RMSE Difference", f"{(lr_rmse - rf_rmse):.2f}")
    
    # Model comparison visualization
    # 模型对比可视化
    st.subheader("🔄 Model Prediction Comparison")
    # 模型预测对比
    
    # Create comparison scatter plot
    # 创建对比散点图
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Scatter(
        x=y_test, y=rf_pred, 
        mode='markers', name='Random Forest',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=y_test, y=lr_pred, 
        mode='markers', name='Linear Regression',
        marker=dict(color='orange', opacity=0.6)
    ))
    
    # Add diagonal line
    # 添加对角线
    min_val = min(y_test.min(), rf_pred.min(), lr_pred.min())
    max_val = max(y_test.max(), rf_pred.max(), lr_pred.max())
    fig_comparison.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', line=dict(dash='dash', color='red'),
        name='Perfect Prediction Line'
    ))
    
    fig_comparison.update_layout(
        title="Random Forest vs Linear Regression Prediction Performance Comparison",
        xaxis_title="Actual Yield",
        yaxis_title="Predicted Yield"
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Feature importance (Random Forest specific)
    # 特征重要性（随机森林特有）
    st.subheader("🌳 Random Forest Feature Importance")
    # 随机森林特征重要性
    
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    fig_importance = px.treemap(
        importance,
        path=['Feature'],
        values='Importance',
        title="Random Forest Feature Importance (Tree Map)",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Linear regression coefficient comparison
    # 线性回归系数对比
    st.subheader("📈 Linear Regression Coefficients vs Random Forest Importance")
    # 线性回归系数 vs 随机森林重要性
    
    # Get linear regression coefficients
    # 获取线性回归系数
    lr_coef = pd.DataFrame({
        'Feature': X.columns,
        'Linear Regression Coefficient': lr_model.coef_,
        'Coefficient Absolute Value': np.abs(lr_model.coef_)
    }).sort_values('Coefficient Absolute Value', ascending=False).head(10)
    
    # Merge feature importance from both models
    # 合并两种模型的特征重要性
    comparison_df = pd.merge(
        importance.head(10),
        lr_coef[['Feature', 'Coefficient Absolute Value']],
        on='Feature',
        how='inner'
    )
    
    # Create comparison chart
    # 创建对比图
    fig_coef_compare = go.Figure()
    
    fig_coef_compare.add_trace(go.Bar(
        x=comparison_df['Feature'],
        y=comparison_df['Importance'],
        name='Random Forest Importance',
        marker_color='blue'
    ))
    
    fig_coef_compare.add_trace(go.Bar(
        x=comparison_df['Feature'],
        y=comparison_df['Coefficient Absolute Value'],
        name='Linear Regression Coefficient Absolute Value',
        marker_color='orange'
    ))
    
    fig_coef_compare.update_layout(
        title="Feature Importance Comparison",
        xaxis_title="Feature",
        yaxis_title="Importance/Coefficient",
        barmode='group'
    )
    
    st.plotly_chart(fig_coef_compare, use_container_width=True)
    
    # Prediction uncertainty analysis
    # 预测不确定性分析
    st.subheader("📈 Random Forest Prediction Uncertainty")
    
    # Collect predictions from all trees
    # 收集所有树的预测
    all_predictions = []
    for tree in rf_model.estimators_:
        all_predictions.append(tree.predict(X_test[:5]))  # Only take first 5 samples
    
    all_predictions = np.array(all_predictions)
    
    fig_uncertainty = go.Figure()
    
    for i in range(min(3, len(all_predictions[0]))):
        fig_uncertainty.add_trace(go.Scatter(
            x=list(range(len(all_predictions))),
            y=all_predictions[:, i],
            mode='lines',
            name=f'Sample {i+1}',
            line=dict(width=2),
            opacity=0.7
        ))
    
    fig_uncertainty.update_layout(
        title="Prediction Variation Across Different Decision Trees (First 3 Samples)",
        xaxis_title="Tree Number",
        yaxis_title="Predicted Yield"
    )
    st.plotly_chart(fig_uncertainty, use_container_width=True)
    
    # Real-time prediction
    # 实时预测
    st.subheader("🔮 Yield Prediction")
    
    # Select important features for input
    # 选择重要特征用于输入
    top_3_features = importance.head(3)['Feature'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Input Feature Values:")
        # 输入特征值
        input_vals = {}
        for feature in top_3_features:
            min_val = X[feature].min()
            max_val = X[feature].max()
            mean_val = X[feature].mean()
            input_vals[feature] = st.slider(
                f"{feature}", min_val, max_val, mean_val, key=f"rf_{feature}"
            )
    
    with col2:
        if st.button("Predict Yield"):
            # 预测产量
            # Prepare input data
            # 准备输入数据
            input_data = [X[col].mean() for col in X.columns]
            for feature, value in input_vals.items():
                if feature in X.columns:
                    idx = X.columns.get_loc(feature)
                    input_data[idx] = value
            
            rf_prediction = rf_model.predict([input_data])[0]
            lr_prediction = lr_model.predict([input_data])[0]
            
            # Calculate prediction uncertainty
            # 计算预测不确定性
            tree_predictions = [tree.predict([input_data])[0] for tree in rf_model.estimators_]
            uncertainty = np.std(tree_predictions)
            
            st.success(f"Random Forest Prediction: **{rf_prediction:.2f}**")
            st.info(f"Linear Regression Prediction: **{lr_prediction:.2f}**")
            st.info(f"Prediction Uncertainty: ±{uncertainty:.2f}")
    
    # Analysis summary
    # 分析总结
    st.subheader("💡 Model Comparison Summary")
    # 模型对比总结
    
    st.markdown(f"""
    ### Random Forest vs Linear Regression Analysis
    
    **Performance Comparison:**
    - Random Forest R²: **{rf_r2:.4f}**
    - Linear Regression R²: **{lr_r2:.4f}**
    - Performance Difference: **{(rf_r2 - lr_r2):.4f}**
    
    **Model Characteristics Comparison:**
    
    | Characteristic | Random Forest | Linear Regression |
    |------|----------|----------|
    | Feature Relationships | Handles nonlinear | Linear assumption |
    | Feature Importance | Gini importance | Coefficient absolute value |
    | Prediction Stability | Higher (ensemble) | Lower |
    | Interpretability | Medium | High |
    | Training Speed | Slower | Faster |
    
    **Business Recommendations:**
    - If prediction accuracy is most important: **{"Random Forest" if rf_r2 > lr_r2 else "Linear Regression"}**
    - If model interpretability is needed: **Linear Regression**
    - If data relationships are complex: **Random Forest**
    - If computational resources are limited: **Linear Regression**
    
    **Most Important Feature:** **{importance.iloc[0]['Feature']}**
    """)

if __name__ == "__main__":
    random_forest_prediction()