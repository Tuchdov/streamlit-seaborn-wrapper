import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from utils import (
    get_plot_customization_sidebar, apply_plot_formatting,
    download_plot, check_data_loaded, get_numeric_columns,
    get_categorical_columns, get_all_columns, MATPLOTLIB_COLORMAPS,
    get_display_data
)

st.title("✨ Specialty Visualizations")
st.markdown("Advanced and specialized plot types for comprehensive data analysis")

if not check_data_loaded():
    st.stop()

df = get_display_data()

numeric_cols = get_numeric_columns(df)
categorical_cols = get_categorical_columns(df)
all_cols = get_all_columns(df)

plot_type = st.selectbox(
    "Select Plot Type",
    ["Pair Plot", "Correlation Matrix", "Distribution Plot", "Cat Plot", 
     "Point Plot", "Factor Plot", "Cluster Map", "Andrews Curves"]
)

config = get_plot_customization_sidebar()

st.markdown("---")

if plot_type == "Pair Plot":
    st.subheader("🔗 Pair Plot")
    st.markdown("Show pairwise relationships in a dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns")
            st.stop()
        
        if len(numeric_cols) > 10:
            st.warning("Too many columns may slow down the plot. Select specific columns below.")
            selected_cols = st.multiselect("Select Columns", numeric_cols, 
                                          default=numeric_cols[:5])
        else:
            selected_cols = st.multiselect("Select Columns", numeric_cols, 
                                          default=numeric_cols)
        
        hue_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
        diag_kind = st.selectbox("Diagonal Plot Type", ["auto", "hist", "kde"])
        corner = st.checkbox("Show only lower triangle", value=False)
        
        if len(selected_cols) < 2:
            st.error("Select at least 2 columns")
            st.stop()
    
    with col1:
        if hue_col:
            g = sns.pairplot(df[selected_cols + [hue_col]], hue=hue_col, 
                           palette=config['palette'], diag_kind=diag_kind, 
                           corner=corner, height=2.5)
        else:
            g = sns.pairplot(df[selected_cols], palette=config['palette'], 
                           diag_kind=diag_kind, corner=corner, height=2.5)
        
        if config.get('title'):
            g.fig.suptitle(config['title'], y=1.02, fontsize=14, fontweight='bold')
        
        st.pyplot(g.fig)
        plt.close()
    
    download_plot(g.fig, "pairplot")

elif plot_type == "Correlation Matrix":
    st.subheader("📊 Correlation Matrix")
    st.markdown("Visualize correlations between all numeric variables")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns")
            st.stop()
        
        selected_cols = st.multiselect("Select Columns", numeric_cols, 
                                      default=numeric_cols[:min(15, len(numeric_cols))])
        
        if len(selected_cols) < 2:
            st.error("Select at least 2 columns")
            st.stop()
        
        method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
        cmap = st.selectbox("Color Map", MATPLOTLIB_COLORMAPS, index=9)
        annot = st.checkbox("Show Values", value=True)
        mask_upper = st.checkbox("Mask Upper Triangle", value=False)
        square = st.checkbox("Square Cells", value=True)
    
    with col1:
        corr = df[selected_cols].corr(method=method)
        
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, 
                   vmin=-1, vmax=1, center=0, square=square,
                   mask=mask, linewidths=0.5, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, "correlation_matrix")

elif plot_type == "Distribution Plot":
    st.subheader("📈 Distribution Plot")
    st.markdown("Compare distributions of multiple variables")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 1:
            st.error("Need at least 1 numeric column")
            st.stop()
        
        selected_cols = st.multiselect("Select Columns", numeric_cols, 
                                      default=numeric_cols[:min(5, len(numeric_cols))])
        
        if len(selected_cols) < 1:
            st.error("Select at least 1 column")
            st.stop()
        
        plot_style = st.selectbox("Plot Style", ["overlaid", "stacked"])
        kde = st.checkbox("Show KDE", value=True)
        hist = st.checkbox("Show Histogram", value=True)
        rug = st.checkbox("Show Rug Plot", value=False)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        colors = sns.color_palette(config['palette'], len(selected_cols))
        
        for i, col in enumerate(selected_cols):
            if hist and kde:
                sns.histplot(data=df, x=col, kde=kde, color=colors[i], 
                           label=col, alpha=0.5 if plot_style == "overlaid" else 1, ax=ax)
            elif kde:
                sns.kdeplot(data=df, x=col, color=colors[i], label=col, 
                          fill=True, alpha=0.5, ax=ax)
            elif hist:
                sns.histplot(data=df, x=col, color=colors[i], label=col, 
                           alpha=0.5 if plot_style == "overlaid" else 1, ax=ax)
            
            if rug:
                sns.rugplot(data=df, x=col, color=colors[i], ax=ax)
        
        ax.legend()
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, "distribution")

elif plot_type == "Cat Plot":
    st.subheader("🎨 Categorical Plot")
    st.markdown("Flexible categorical data visualization")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(categorical_cols) < 1 or len(numeric_cols) < 1:
            st.error("Need at least 1 categorical and 1 numeric column")
            st.stop()
        
        kind = st.selectbox("Plot Kind", ["strip", "swarm", "box", "violin", "boxen", 
                                         "point", "bar", "count"])
        
        x_col = st.selectbox("X-axis", categorical_cols + numeric_cols)
        
        if kind != "count":
            y_col = st.selectbox("Y-axis", numeric_cols)
        else:
            y_col = None
        
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        col_col = st.selectbox("Column facet (optional)", [None] + categorical_cols)
        row_col = st.selectbox("Row facet (optional)", [None] + categorical_cols)
    
    with col1:
        if kind == "count":
            g = sns.catplot(data=df, x=x_col, kind=kind, hue=hue_col, 
                          col=col_col, row=row_col, palette=config['palette'], 
                          height=4, aspect=1.5)
        else:
            g = sns.catplot(data=df, x=x_col, y=y_col, kind=kind, hue=hue_col, 
                          col=col_col, row=row_col, palette=config['palette'], 
                          height=4, aspect=1.5)
        
        if config.get('title'):
            g.fig.suptitle(config['title'], y=1.02, fontsize=14, fontweight='bold')
        
        st.pyplot(g.fig)
        plt.close()
    
    download_plot(g.fig, f"catplot_{kind}")

elif plot_type == "Point Plot":
    st.subheader("📍 Point Plot")
    st.markdown("Show point estimates and confidence intervals")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(categorical_cols) < 1 or len(numeric_cols) < 1:
            st.error("Need at least 1 categorical and 1 numeric column")
            st.stop()
        
        x_col = st.selectbox("X-axis (Categories)", categorical_cols)
        y_col = st.selectbox("Y-axis (Values)", numeric_cols)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        estimator = st.selectbox("Estimator", ["mean", "median", "sum", "min", "max"])
        ci = st.slider("Confidence Interval (%)", 0, 99, 95)
        markers = st.checkbox("Show Markers", value=True)
        linestyles = st.checkbox("Different Line Styles", value=False)
        dodge = st.checkbox("Dodge Points", value=False)
        join = st.checkbox("Join Points with Lines", value=True)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        estimator_func = None
        if estimator == "mean":
            estimator_func = np.mean
        elif estimator == "median":
            estimator_func = np.median
        elif estimator == "sum":
            estimator_func = np.sum
        elif estimator == "min":
            estimator_func = np.min
        elif estimator == "max":
            estimator_func = np.max
        
        sns.pointplot(data=df, x=x_col, y=y_col, hue=hue_col, 
                     palette=config['palette'], estimator=estimator_func,
                     errorbar=('ci', ci) if ci > 0 else None, markers='o' if markers else '',
                     linestyles=['-', '--', '-.'] if linestyles and hue_col else '-',
                     dodge=dodge, join=join, ax=ax)
        
        ax.tick_params(axis='x', rotation=45)
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"pointplot_{x_col}_vs_{y_col}")

elif plot_type == "Factor Plot":
    st.subheader("🎭 Factor Plot")
    st.markdown("Multi-panel categorical plots")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(categorical_cols) < 1 or len(numeric_cols) < 1:
            st.error("Need at least 1 categorical and 1 numeric column")
            st.stop()
        
        st.info("Factor Plot is now called catplot in newer versions of Seaborn")
        
        kind = st.selectbox("Plot Kind", ["point", "bar", "strip", "swarm", "box", "violin", "boxen"])
        x_col = st.selectbox("X-axis", categorical_cols)
        y_col = st.selectbox("Y-axis", numeric_cols)
        hue_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
        col_col = st.selectbox("Column facet", [None] + categorical_cols)
        row_col = st.selectbox("Row facet (optional)", [None] + categorical_cols)
        col_wrap = st.slider("Columns per row (if using column facet)", 1, 5, 3)
    
    with col1:
        g = sns.catplot(data=df, x=x_col, y=y_col, kind=kind, hue=hue_col, 
                       col=col_col, row=row_col, palette=config['palette'], 
                       col_wrap=col_wrap if col_col and not row_col else None,
                       height=4, aspect=1.2)
        
        if config.get('title'):
            g.fig.suptitle(config['title'], y=1.02, fontsize=14, fontweight='bold')
        
        st.pyplot(g.fig)
        plt.close()
    
    download_plot(g.fig, f"factorplot_{kind}")

elif plot_type == "Cluster Map":
    st.subheader("🗺️ Cluster Map")
    st.markdown("Hierarchical clustering heatmap")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns")
            st.stop()
        
        selected_cols = st.multiselect("Select Columns", numeric_cols, 
                                      default=numeric_cols[:min(10, len(numeric_cols))])
        
        if len(selected_cols) < 2:
            st.error("Select at least 2 columns")
            st.stop()
        
        if len(df) > 100:
            st.warning("Large datasets may be slow. Consider using fewer rows.")
            max_rows = st.slider("Max rows to display", 10, min(500, len(df)), 100)
            data_to_plot = df[selected_cols].head(max_rows)
        else:
            data_to_plot = df[selected_cols]
        
        method = st.selectbox("Clustering Method", ["average", "single", "complete", "ward"])
        metric = st.selectbox("Distance Metric", ["euclidean", "cityblock", "cosine", "correlation"])
        cmap = st.selectbox("Color Map", MATPLOTLIB_COLORMAPS, index=0)
        standard_scale = st.selectbox("Standardize", [None, "rows", "columns"])
        z_score = st.selectbox("Z-score", [None, "rows", "columns"])
        row_cluster = st.checkbox("Cluster Rows", value=True)
        col_cluster = st.checkbox("Cluster Columns", value=True)
    
    with col1:
        try:
            g = sns.clustermap(data_to_plot, method=method, metric=metric, 
                             cmap=cmap, standard_scale=standard_scale if standard_scale else None,
                             z_score=z_score if z_score else None,
                             row_cluster=row_cluster, col_cluster=col_cluster,
                             figsize=config['figsize'], cbar_pos=(0.02, 0.8, 0.03, 0.15))
            
            if config.get('title'):
                g.fig.suptitle(config['title'], y=0.98, fontsize=14, fontweight='bold')
            
            st.pyplot(g.fig)
            plt.close()
        except Exception as e:
            st.error(f"Error creating cluster map: {str(e)}")
    
    download_plot(g.fig, "clustermap")

elif plot_type == "Andrews Curves":
    st.subheader("🌊 Andrews Curves")
    st.markdown("Visualize multivariate data as curves")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(categorical_cols) < 1 or len(numeric_cols) < 2:
            st.error("Need at least 1 categorical and 2 numeric columns")
            st.stop()
        
        class_col = st.selectbox("Class Column (categories)", categorical_cols)
        
        selected_cols = st.multiselect("Select Numeric Columns", numeric_cols, 
                                      default=numeric_cols[:min(5, len(numeric_cols))])
        
        if len(selected_cols) < 2:
            st.error("Select at least 2 numeric columns")
            st.stop()
        
        if len(df) > 500:
            st.warning("Large datasets may be cluttered. Consider sampling.")
            sample_size = st.slider("Sample size", 50, min(1000, len(df)), 500)
            data_to_plot = df.sample(n=sample_size, random_state=42)
        else:
            data_to_plot = df
        
        alpha = st.slider("Line Transparency", 0.1, 1.0, 0.5, 0.1)
    
    with col1:
        from pandas.plotting import andrews_curves
        
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        plot_data = data_to_plot[selected_cols + [class_col]].copy()
        
        andrews_curves(plot_data, class_col, color=sns.color_palette(config['palette']), 
                      alpha=alpha, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, "andrews_curves")

st.markdown("---")
st.info("💡 Tip: Specialty plots work best when you have the right data structure. Make sure your data has appropriate categorical and numeric columns for the plot type you choose.")
