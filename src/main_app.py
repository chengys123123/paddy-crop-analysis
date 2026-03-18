import streamlit as st

# 页面配置
# Page configuration
st.set_page_config(
    page_title="Paddy Data Analysis Platform",
    page_icon="🌾",
    layout="wide"
)

# 应用标题
# The title of the application
st.sidebar.title("🌾 Paddy Data Analysis Platform")
st.sidebar.markdown("---")

# 导航菜单
# The menu for navigation
page = st.sidebar.radio(
    "Choose Analysis Module:",
    ["🏠 Home", "📋 data_preprocessing", "🔧 PCA_preprocessing", "📊 univariate_analysis", "🔗 bivariate_analysis", "🌱 growth_stage_analysis", "🔍 PCA_analysis", "📈 linear_regression_prediction", "🌲 random_forest_prediction", "🙏 acknowledgements"]
)

# 首页
# Home page
if page == "🏠 Home":

    # 创建三列布局：左侧为个人信息，中间为空白间隔，右侧为Logo
    # Create a three-column layout with personal information on the left, a blank space in the middle, and a Logo on the right
    col1, col_space, col2 = st.columns([2, 0.5, 1])
    
    with col1:
        st.title("🌾 Welcome to the Paddy Data Analysis Platform")
        
        st.markdown("""
        ### 👤 Personal information
        
        - name：YanshuCheng
        - studentID：4320251017
        - Course: Data Visualization 2025
        - major：Data, Artificial Intelligence, and Cloud
        - Efrei Email：yanshu.cheng@efrei.net
        - Advisor：Prof. Mano Mathew
        - Project Date：Nov18-2025
        
        ---
        """)

     # 空白间隔列 - 增加左右间距
     # Blank Interval Columns-Increase Left and Right Spacing
    with col_space:
        st.write("")  # 空内容，仅用于创建间距
    
    with col2:
        # 创建两列用于放置校徽，增加间距
        logo_col1, logo_col2 = st.columns([1, 1])
    
        with logo_col1:
            # 显示第一个校徽
            try:
                st.image("WUT-Logo.png", width=150)
            except:
                st.info("School Emblem Picture 1 Loading...")
        
        with logo_col2:
            # 显示第二个校徽
            # 添加空行来下移校徽
            for _ in range(2):  # 循环2次，添加2个空行
                st.write("")
            try:
                st.image("Efrei-Logo.png", width=150)  
            except:
                st.info("School Emblem Picture 2 Loading...")
    
    st.markdown("""
    ### Platform functions
    
    - Data preprocessing: data loading, quality inspection, abnormal value processing and data cleaning
    - PCA data preprocessing: one-hot coding of categorical variables and normalization of numerical variables
    - Univariate analysis: analysis of variety distribution, regional distribution and yield distribution
    - Bivariate analysis: variety vs yield, region vs yield, etc.
    - Growth stage analysis: correlation analysis of rice growth stage
    - PCA analysis: find out the main change direction in high-dimensional data through dimensionality reduction, which is convenient for visualization and interpretation.
    - Linear Regression Prediction: Rice Yield Prediction
    - Random Forest Prediction: Rice Yield Prediction
                                    
    ### Instructions for use
    
    1. First, enter the "data_preprocessing" module to load and clean data.
    2. Next enter the "PCA_preprocessing" module to prepare data for PCA analysis.
    3. Then switch to the "univariate_analysis" module for visual analysis.
    4. Enter the "bivariate_analysis" module to explore the relationship between variables.
    5. Enter the "growth_stage_analysis" module to explore the effects of rainfall and temperature on yield.
    6. Enter the "PCA_analysis" module to identify the environmental and management factors that have the greatest impact on yield.
    7. Enter "linear_regression_prediction" and "random_forest_prediction", predict rice yield through two machine learning methods, and make dynamic visualization to show the prediction effect.
    8. Data will be automatically shared between different modules.
    """)
    
    # 显示数据状态
    # Displays the data status
    if 'df' in st.session_state:
        st.success(f"✅ Data loaded: {st.session_state.df.shape[0]} Line × {st.session_state.df.shape[1]} Column")
    else:
        st.info("📝 Please enter the data preprocessing module to load data first")

    if 'df_pca' in st.session_state:
        st.success(f"✅ PCA data prepared: {st.session_state.df_pca.shape[0]} Line × {st.session_state.df_pca.shape[1]} Column")
    else:
        st.info("📝 To perform PCA analysis, please enter the PCA data preprocessing module first.")

# 数据预处理页面
# Data preprocessing page
elif page == "📋 data_preprocessing":
    from data_preprocessing import data_preprocessing_page
    data_preprocessing_page()

# PCA数据预处理页面 
# PCA Data Preprocessing Page
elif page == "🔧 PCA_preprocessing":
    # 检查基础数据是否已加载
    if 'df' not in st.session_state:
        st.error("Please run the data_preprocessing step first!")
        st.stop()
        
    from PCA_preprocessing import pca_preprocessing
    pca_preprocessing()

# 单变量分析页面
# Univariate analysis page
elif page == "📊 univariate_analysis":
    # 检查数据是否已加载
    if 'df' not in st.session_state:
        st.error("Please run the data_preprocessing step first!")
        st.stop()
        
    from univariate_analysis import univariate_analysis
    univariate_analysis()

# 双变量分析页面
# Bivariate Analysis Page 
elif page == "🔗 bivariate_analysis":
    # 检查数据是否已加载
    if 'df' not in st.session_state:
        st.error("Please run the data_preprocessing step first!")
        st.stop()
    
    # 导入双变量分析模块
    from bivariate_analysis import bivariate_analysis
    bivariate_analysis()

# 生长阶段分析页面
# Growth Stage Analysis Page 
elif page == "🌱 growth_stage_analysis":
    # 检查数据是否已加载
    if 'df' not in st.session_state:
        st.error("Please run the data_preprocessing step first!")
        st.stop()
    
    # 导入生长阶段分析模块
    from growth_stage_analysis import growth_stage_analysis
    growth_stage_analysis()

# PCA分析页面
# PCA Analysis Page 
elif page == "🔍 PCA_analysis":
    # 检查数据是否已加载
    if 'df' not in st.session_state:
        st.error("Please run the PCA_preprocessing steps first!")
        st.stop()
    
    # 导入多变量分析模块
    from PCA_analysis import pca_analysis
    pca_analysis()

# 线性回归预测页面
# Linear Regression Prediction Page 
elif page == "📈 linear_regression_prediction":
    # 检查数据是否已准备
    if 'df_pca' not in st.session_state:
        st.error("Please run the PCA_preprocessing steps first!")
        st.stop()
        
    from linear_regression_prediction import linear_regression_prediction
    linear_regression_prediction()

# 随机森林预测页面
# Random Forest Prediction Page 
elif page == "🌲 random_forest_prediction":
    if 'df_pca' not in st.session_state:
        st.error("Please run the PCA_preprocessing steps first!")
        st.stop()
        
    from random_forest_prediction import random_forest_prediction
    random_forest_prediction()

# 致谢页面
# Acknowledgements page 
elif page == "🙏 acknowledgements":
    from acknowledgements import acknowledgements_page
    acknowledgements_page()

# 页脚
# Footer 
st.sidebar.markdown("---")
st.sidebar.markdown("🌾 Paddy Data Analysis Platform")