import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')

st.title("📚 Visualization Cheat Sheet")
st.markdown("Quick reference guide with examples and documentation links for all plot types")

st.sidebar.markdown("### Navigation")
plot_category = st.sidebar.radio(
    "Select Category",
    ["1D Plots", "2D Plots", "3D Plots", "Specialty Plots", "Quick Templates"]
)

def create_sample_data():
    np.random.seed(42)
    return pd.DataFrame({
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'Value1': np.random.randn(100) * 10 + 50,
        'Value2': np.random.randn(100) * 15 + 75,
        'Value3': np.random.randint(1, 100, 100),
        'Group': np.random.choice(['X', 'Y', 'Z'], 100),
    })

sample_df = create_sample_data()

if plot_category == "1D Plots":
    st.header("📈 1D Plots - Cheat Sheet")
    st.markdown("Single variable visualizations for distributions and frequencies")
    
    st.markdown("---")
    st.subheader("📊 Histogram")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Show frequency distribution of a numeric variable
        - Understand data spread and central tendency
        - Identify outliers and patterns
        
        **Best for:**
        - Continuous numeric data
        - Understanding data distribution shape
        - Comparing distributions across groups
        
        **Documentation:**
        - [Seaborn Histogram](https://seaborn.pydata.org/generated/seaborn.histplot.html)
        - [Matplotlib Histogram](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=sample_df, x='Value1', bins=20, kde=True, color='skyblue')
        ax.set_title("Example: Distribution of Values", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("📉 KDE Plot")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Smooth estimate of probability distribution
        - Compare distributions without binning artifacts
        - Overlay multiple distributions
        
        **Best for:**
        - Large datasets (>50 points)
        - Comparing multiple groups
        - Smooth visualization needs
        
        **Documentation:**
        - [Seaborn KDE Plot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=sample_df, x='Value1', fill=True, color='coral')
        ax.set_title("Example: Smooth Density Curve", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("📦 Box Plot")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Show quartiles and outliers
        - Compare distributions across categories
        - Identify median and spread
        
        **Best for:**
        - Small to medium datasets
        - Outlier detection
        - Categorical comparisons
        
        **Key Elements:**
        - Box: 25th to 75th percentile (IQR)
        - Line: Median (50th percentile)
        - Whiskers: 1.5 × IQR
        - Points: Outliers
        
        **Documentation:**
        - [Seaborn Box Plot](https://seaborn.pydata.org/generated/seaborn.boxplot.html)
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=sample_df, x='Category', y='Value1', palette='Set2')
        ax.set_title("Example: Values by Category", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("🎻 Violin Plot")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Combine box plot and KDE
        - Show full distribution shape
        - Compare multiple groups
        
        **Best for:**
        - Medium to large datasets
        - Multi-modal distributions
        - Rich distribution information
        
        **Documentation:**
        - [Seaborn Violin Plot](https://seaborn.pydata.org/generated/seaborn.violinplot.html)
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=sample_df, x='Category', y='Value2', palette='muted')
        ax.set_title("Example: Distribution Shapes", fontweight='bold')
        st.pyplot(fig)
        plt.close()

elif plot_category == "2D Plots":
    st.header("📊 2D Plots - Cheat Sheet")
    st.markdown("Two-variable visualizations for relationships and patterns")
    
    st.markdown("---")
    st.subheader("🔵 Scatter Plot")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Show relationship between two numeric variables
        - Identify correlations and patterns
        - Detect clusters and outliers
        
        **Best for:**
        - Continuous numeric data
        - Correlation analysis
        - Pattern recognition
        
        **Documentation:**
        - [Seaborn Scatter Plot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)
        - [Matplotlib Scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=sample_df, x='Value1', y='Value2', hue='Group', palette='deep', s=100, alpha=0.7)
        ax.set_title("Example: Relationship Between Variables", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("📈 Line Plot")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Show trends over time or sequences
        - Connect sequential data points
        - Display continuous change
        
        **Best for:**
        - Time series data
        - Sequential measurements
        - Trend analysis
        
        **Documentation:**
        - [Seaborn Line Plot](https://seaborn.pydata.org/generated/seaborn.lineplot.html)
        - [Matplotlib Line Plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
        """)
    with col2:
        time_data = pd.DataFrame({
            'Time': range(50),
            'Value': np.cumsum(np.random.randn(50))
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=time_data, x='Time', y='Value', marker='o', color='green')
        ax.set_title("Example: Trend Over Time", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("📊 Bar Chart")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Compare values across categories
        - Show aggregated statistics
        - Display categorical data
        
        **Best for:**
        - Categorical comparisons
        - Summary statistics
        - Group comparisons
        
        **Documentation:**
        - [Seaborn Bar Plot](https://seaborn.pydata.org/generated/seaborn.barplot.html)
        - [Matplotlib Bar Chart](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html)
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=sample_df, x='Category', y='Value1', palette='viridis', estimator=np.mean)
        ax.set_title("Example: Mean Values by Category", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("🔥 Heatmap")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Visualize matrix data with colors
        - Show correlations between variables
        - Display patterns in 2D data
        
        **Best for:**
        - Correlation matrices
        - Pivot tables
        - Grid-based data
        
        **Documentation:**
        - [Seaborn Heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
        """)
    with col2:
        corr_data = sample_df[['Value1', 'Value2', 'Value3']].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
        ax.set_title("Example: Correlation Matrix", fontweight='bold')
        st.pyplot(fig)
        plt.close()

elif plot_category == "3D Plots":
    st.header("🎲 3D Plots - Cheat Sheet")
    st.markdown("Three-dimensional visualizations for complex relationships")
    
    st.markdown("---")
    st.subheader("🔵 3D Scatter Plot")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Show relationships between three variables
        - Identify 3D patterns and clusters
        - Explore multidimensional data
        
        **Best for:**
        - Scientific data analysis
        - Multi-variable relationships
        - Spatial data visualization
        
        **Tips:**
        - Use interactive Plotly for rotation
        - Color by additional variable for 4D view
        - Keep point count reasonable (<5000)
        
        **Documentation:**
        - [Plotly 3D Scatter](https://plotly.com/python/3d-scatter-plots/)
        - [Matplotlib 3D](https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html)
        """)
    with col2:
        st.info("💡 3D plots work best in interactive mode. Visit the 3D Plots page to explore with rotation controls!")
    
    st.markdown("---")
    st.subheader("🏔️ Surface Plot")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Visualize functions of two variables
        - Show continuous surfaces
        - Display topographical data
        
        **Best for:**
        - Mathematical functions
        - Elevation/terrain data
        - Response surfaces
        
        **Documentation:**
        - [Plotly Surface](https://plotly.com/python/3d-surface-plots/)
        - [Matplotlib Surface](https://matplotlib.org/stable/gallery/mplot3d/surface3d.html)
        """)
    with col2:
        st.info("💡 Try mathematical functions like sin(x)*cos(y) on the 3D Plots page!")
    
    st.markdown("---")
    st.subheader("🗺️ Contour Plot")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Show 3D data in 2D
        - Visualize elevation or intensity
        - Display gradients and levels
        
        **Best for:**
        - Topographic maps
        - Density visualization
        - Gradient analysis
        
        **Documentation:**
        - [Matplotlib Contour](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)
        - [Plotly Contour](https://plotly.com/python/contour-plots/)
        """)
    with col2:
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        
        fig, ax = plt.subplots(figsize=(6, 4))
        cs = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
        plt.colorbar(cs, ax=ax)
        ax.set_title("Example: Contour Levels", fontweight='bold')
        st.pyplot(fig)
        plt.close()

elif plot_category == "Specialty Plots":
    st.header("✨ Specialty Plots - Cheat Sheet")
    st.markdown("Advanced visualizations for comprehensive analysis")
    
    st.markdown("---")
    st.subheader("🔗 Pair Plot")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Explore all pairwise relationships
        - Initial data exploration
        - Identify interesting variable pairs
        
        **Best for:**
        - Multiple numeric variables (2-10)
        - Exploratory data analysis
        - Finding correlations
        
        **Shows:**
        - Scatter plots for variable pairs
        - Distributions on diagonal
        - Group comparisons
        
        **Documentation:**
        - [Seaborn Pair Plot](https://seaborn.pydata.org/generated/seaborn.pairplot.html)
        """)
    with col2:
        st.info("💡 Pair plots create a grid of scatter plots for all variable combinations. Best with 2-8 variables!")
    
    st.markdown("---")
    st.subheader("📊 Correlation Matrix")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Show correlations between all variables
        - Identify strongly correlated features
        - Feature selection for modeling
        
        **Best for:**
        - Numeric data analysis
        - Feature engineering
        - Multicollinearity detection
        
        **Correlation Types:**
        - Pearson: Linear relationships
        - Spearman: Monotonic relationships
        - Kendall: Rank-based correlation
        
        **Documentation:**
        - [Pandas Correlation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        corr_data = sample_df[['Value1', 'Value2', 'Value3']].corr()
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, 
                   square=True, linewidths=1, vmin=-1, vmax=1)
        ax.set_title("Example: Variable Correlations", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("🎨 Categorical Plot (Cat Plot)")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Flexible categorical data visualization
        - Multiple plot types in one interface
        - Faceted visualizations
        
        **Plot Types:**
        - strip, swarm: Individual points
        - box, violin, boxen: Distributions
        - bar, point, count: Aggregates
        
        **Features:**
        - Row and column faceting
        - Hue grouping
        - Multiple estimators
        
        **Documentation:**
        - [Seaborn Cat Plot](https://seaborn.pydata.org/generated/seaborn.catplot.html)
        """)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=sample_df, x='Category', y='Value1', hue='Group', palette='Set2', split=False)
        ax.set_title("Example: Grouped Categorical Data", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("🗺️ Cluster Map")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **When to use:**
        - Hierarchical clustering visualization
        - Group similar observations/features
        - Pattern discovery in high-dimensional data
        
        **Best for:**
        - Gene expression data
        - Customer segmentation
        - Feature grouping
        
        **Methods:**
        - Ward: Minimize variance
        - Average: Average linkage
        - Complete: Maximum distance
        - Single: Minimum distance
        
        **Documentation:**
        - [Seaborn Cluster Map](https://seaborn.pydata.org/generated/seaborn.clustermap.html)
        """)
    with col2:
        st.info("💡 Cluster maps automatically group similar rows and columns using hierarchical clustering!")

elif plot_category == "Quick Templates":
    st.header("⚡ Quick Templates")
    st.markdown("Common visualization patterns for typical use cases")
    
    template_type = st.selectbox(
        "Select Template",
        ["Sales Analysis", "Survey Results", "Scientific Data", "Time Series Analysis", 
         "Comparison Study", "Distribution Analysis"]
    )
    
    st.markdown("---")
    
    if template_type == "Sales Analysis":
        st.subheader("📊 Sales Analysis Template")
        st.markdown("""
        **Recommended Plots:**
        1. **Line Plot**: Sales over time
        2. **Bar Chart**: Sales by category/region
        3. **Heatmap**: Sales by month and category
        4. **Box Plot**: Sales distribution by product
        
        **Data Requirements:**
        - Date/time column
        - Sales/revenue numeric column
        - Category/product column
        - Optional: Region, customer segment
        """)
        
        st.code("""
# Example workflow:
1. Upload CSV with: Date, Sales, Category, Region
2. Use Line Plot: X=Date, Y=Sales (check trend)
3. Use Bar Chart: X=Category, Y=Sales (compare categories)
4. Use Heatmap: Pivot table with Category vs Month
        """, language="text")
    
    elif template_type == "Survey Results":
        st.subheader("📋 Survey Results Template")
        st.markdown("""
        **Recommended Plots:**
        1. **Count Plot**: Response frequencies
        2. **Bar Chart**: Average ratings by question
        3. **Violin Plot**: Rating distributions
        4. **Heatmap**: Correlation between questions
        
        **Data Requirements:**
        - Categorical responses (Yes/No, ratings)
        - Multiple choice columns
        - Numeric rating scales
        """)
        
        st.code("""
# Example workflow:
1. Upload survey CSV with response columns
2. Use Count Plot: Show frequency of each response
3. Use Bar Chart: Compare average ratings across groups
4. Use Correlation Matrix: Find related questions
        """, language="text")
    
    elif template_type == "Scientific Data":
        st.subheader("🔬 Scientific Data Template")
        st.markdown("""
        **Recommended Plots:**
        1. **Scatter Plot**: Variable relationships
        2. **Regression Plot**: Trend with confidence interval
        3. **Pair Plot**: Explore all relationships
        4. **Box Plot**: Group comparisons with outliers
        
        **Data Requirements:**
        - Multiple numeric measurements
        - Experimental conditions/groups
        - Replicate data points
        """)
        
        st.code("""
# Example workflow:
1. Upload experimental data CSV
2. Use Scatter Plot: Check for correlations
3. Use Regression Plot: Model relationships
4. Use Box Plot: Compare treatment groups
5. Use Pair Plot: Full exploratory analysis
        """, language="text")
    
    elif template_type == "Time Series Analysis":
        st.subheader("📈 Time Series Template")
        st.markdown("""
        **Recommended Plots:**
        1. **Line Plot**: Primary trend visualization
        2. **Histogram**: Value distribution
        3. **Heatmap**: Seasonal patterns (month vs year)
        4. **Box Plot**: Compare periods (monthly, quarterly)
        
        **Data Requirements:**
        - DateTime column
        - Numeric value column
        - Optional: Category for multiple series
        """)
        
        st.code("""
# Example workflow:
1. Upload time series CSV with Date and Value columns
2. Use Line Plot: X=Date, Y=Value, show trend
3. Use Box Plot: Group by month to see seasonality
4. Use Heatmap: Create pivot with Month vs Year
        """, language="text")
    
    elif template_type == "Comparison Study":
        st.subheader("⚖️ Comparison Study Template")
        st.markdown("""
        **Recommended Plots:**
        1. **Bar Chart**: Side-by-side group comparison
        2. **Violin Plot**: Distribution comparison
        3. **Point Plot**: Mean with confidence intervals
        4. **Heatmap**: Multi-factor comparison matrix
        
        **Data Requirements:**
        - Group/category column
        - Numeric measurement column
        - Optional: Multiple factors
        """)
        
        st.code("""
# Example workflow:
1. Upload data with groups and measurements
2. Use Bar Chart: Compare means across groups
3. Use Violin Plot: Compare full distributions
4. Use Point Plot: Show trends with error bars
        """, language="text")
    
    elif template_type == "Distribution Analysis":
        st.subheader("📊 Distribution Analysis Template")
        st.markdown("""
        **Recommended Plots:**
        1. **Histogram**: Overall distribution shape
        2. **KDE Plot**: Smooth density estimate
        3. **Box Plot**: Quartiles and outliers
        4. **ECDF Plot**: Cumulative distribution
        
        **Data Requirements:**
        - Numeric column(s)
        - Optional: Group column for comparison
        """)
        
        st.code("""
# Example workflow:
1. Upload numeric data
2. Use Histogram: Check distribution shape
3. Use KDE Plot: Smooth visualization
4. Use Box Plot: Identify outliers and quartiles
5. Use ECDF Plot: See cumulative probabilities
        """, language="text")

st.markdown("---")
st.markdown("### 📖 Additional Resources")
st.markdown("""
**Official Documentation:**
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html) - Official examples
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) - Matplotlib examples
- [Plotly Python](https://plotly.com/python/) - Interactive plots

**Learning Resources:**
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Python Graph Gallery](https://python-graph-gallery.com/) - Code examples

**Color Palettes:**
- Use sidebar to experiment with different color schemes
- ColorBrewer palettes: great for categorical data
- Sequential palettes: good for continuous data
- Diverging palettes: ideal for data with meaningful center (e.g., correlation)
""")
