import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(
    page_title="Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Seaborn & Matplotlib Visualization Dashboard")
st.markdown("### Create beautiful visualizations without writing code")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None

st.sidebar.header("📁 Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload your data file",
    type=['csv', 'xlsx', 'xls', 'json'],
    help="Supported formats: CSV, Excel, JSON"
)

if uploaded_file is not None:
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        
        st.session_state.data = df
        st.session_state.filename = uploaded_file.name
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
        
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

use_sample_data = st.sidebar.checkbox("Use Sample Data", value=False)

if use_sample_data:
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 200),
        'Value1': np.random.randn(200) * 10 + 50,
        'Value2': np.random.randn(200) * 15 + 75,
        'Value3': np.random.randint(1, 100, 200),
        'Group': np.random.choice(['Group1', 'Group2', 'Group3'], 200),
        'Score': np.random.uniform(0, 100, 200)
    })
    st.session_state.data = sample_df
    st.session_state.filename = "Sample Data"
    st.sidebar.info("📊 Using sample dataset")

st.markdown("---")

if st.session_state.data is not None:
    df = st.session_state.data
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Data Source", st.session_state.filename)
    
    with st.expander("📋 View Data Preview", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
    
    with st.expander("📊 Data Summary", expanded=False):
        st.write("**Column Information:**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)
        
        st.write("**Numeric Columns Statistics:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🎨 Choose a visualization type from the sidebar")
    st.markdown("""
    **Available visualization pages:**
    - **1D Plots**: Histograms, KDE plots, Box plots, Violin plots, Strip plots
    - **2D Plots**: Scatter plots, Line plots, Bar charts, Heatmaps, Regression plots
    - **3D Plots**: 3D Scatter, Surface plots, Contour plots
    - **Specialty Plots**: Pair plots, Correlation matrices, Categorical plots
    """)
    
else:
    st.info("👆 Please upload a data file or use sample data from the sidebar to begin creating visualizations.")
    
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Upload your data** using the sidebar (CSV, Excel, or JSON format)
    2. **Or use sample data** to explore the dashboard features
    3. **Navigate to different plot pages** from the sidebar menu
    4. **Customize your visualizations** using interactive controls
    5. **Download your plots** in various formats (PNG, SVG, PDF)
    """)
    
    st.markdown("### 📊 Supported Plot Types")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1D Visualizations:**")
        st.markdown("- Histogram")
        st.markdown("- KDE Plot")
        st.markdown("- Box Plot")
        st.markdown("- Violin Plot")
        st.markdown("- Strip Plot")
        st.markdown("- Swarm Plot")
        
        st.markdown("**2D Visualizations:**")
        st.markdown("- Scatter Plot")
        st.markdown("- Line Plot")
        st.markdown("- Bar Chart")
        st.markdown("- Heatmap")
        st.markdown("- Regression Plot")
        st.markdown("- Hexbin Plot")
    
    with col2:
        st.markdown("**3D Visualizations:**")
        st.markdown("- 3D Scatter Plot")
        st.markdown("- 3D Surface Plot")
        st.markdown("- Contour Plot")
        st.markdown("- 3D Line Plot")
        
        st.markdown("**Specialty Plots:**")
        st.markdown("- Pair Plot")
        st.markdown("- Correlation Matrix")
        st.markdown("- Distribution Plot")
        st.markdown("- Count Plot")
        st.markdown("- Point Plot")
        st.markdown("- Factor Plot")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.info("This dashboard provides access to all Seaborn and Matplotlib plotting capabilities through an easy-to-use interface.")
