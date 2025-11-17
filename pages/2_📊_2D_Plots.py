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
    add_data_filters
)

st.title("📊 2D Visualizations")
st.markdown("Create two-dimensional plots to visualize relationships between variables")

if not check_data_loaded():
    st.stop()

df = st.session_state.data

df = add_data_filters(df)

numeric_cols = get_numeric_columns(df)
categorical_cols = get_categorical_columns(df)
all_cols = get_all_columns(df)

plot_type = st.selectbox(
    "Select Plot Type",
    ["Scatter Plot", "Line Plot", "Bar Chart", "Heatmap", "Regression Plot", 
     "Hexbin Plot", "Joint Plot", "Residual Plot"]
)

config = get_plot_customization_sidebar()

st.markdown("---")

if plot_type == "Scatter Plot":
    st.subheader("🔵 Scatter Plot")
    st.markdown("Show relationship between two numeric variables")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns")
            st.stop()
        
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
        hue_col = st.selectbox("Color by (optional)", [None] + categorical_cols + numeric_cols)
        size_col = st.selectbox("Size by (optional)", [None] + numeric_cols)
        style_col = st.selectbox("Style by (optional)", [None] + categorical_cols)
        alpha = st.slider("Transparency", 0.1, 1.0, 0.7, 0.1)
        size = st.slider("Point Size", 10, 500, 100)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, size=size_col, 
                       style=style_col, palette=config['palette'], 
                       alpha=alpha, s=size, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"scatter_{x_col}_vs_{y_col}")

elif plot_type == "Line Plot":
    st.subheader("📈 Line Plot")
    st.markdown("Show trends over continuous or categorical variables")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 1:
            st.error("Need at least 1 numeric column")
            st.stop()
        
        x_col = st.selectbox("X-axis", all_cols)
        y_col = st.selectbox("Y-axis", numeric_cols)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        style_col = st.selectbox("Style by (optional)", [None] + categorical_cols)
        markers = st.checkbox("Show Markers", value=True)
        dashes = st.checkbox("Use Dashed Lines", value=False)
        sort = st.checkbox("Sort X values", value=True)
        estimator = st.selectbox("Aggregation", ["mean", "median", "sum", "min", "max", None])
        ci = st.slider("Confidence Interval (%)", 0, 99, 95)
    
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
        
        sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, style=style_col,
                    palette=config['palette'], markers=markers, dashes=dashes,
                    estimator=estimator_func, errorbar=('ci', ci) if ci > 0 else None,
                    sort=sort, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"line_{x_col}_vs_{y_col}")

elif plot_type == "Bar Chart":
    st.subheader("📊 Bar Chart")
    st.markdown("Compare values across categories")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        x_col = st.selectbox("X-axis (Categories)", all_cols)
        y_col = st.selectbox("Y-axis (Values)", numeric_cols)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        orient = st.radio("Orientation", ["Vertical", "Horizontal"])
        estimator = st.selectbox("Aggregation", ["mean", "sum", "median", "min", "max", "count"])
        ci = st.slider("Error bars (% CI)", 0, 99, 95)
        order = st.checkbox("Sort by value", value=False)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        estimator_func = None
        if estimator == "mean":
            estimator_func = np.mean
        elif estimator == "sum":
            estimator_func = np.sum
        elif estimator == "median":
            estimator_func = np.median
        elif estimator == "min":
            estimator_func = np.min
        elif estimator == "max":
            estimator_func = np.max
        elif estimator == "count":
            estimator_func = len
        
        order_list = None
        if order and x_col in categorical_cols:
            grouped = df.groupby(x_col)[y_col].apply(estimator_func)
            order_list = grouped.sort_values(ascending=False).index.tolist()
        
        if orient == "Vertical":
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, 
                       palette=config['palette'], estimator=estimator_func,
                       errorbar=('ci', ci) if ci > 0 else None, order=order_list, ax=ax)
            ax.tick_params(axis='x', rotation=45)
        else:
            sns.barplot(data=df, y=x_col, x=y_col, hue=hue_col, 
                       palette=config['palette'], estimator=estimator_func,
                       errorbar=('ci', ci) if ci > 0 else None, order=order_list, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"bar_{x_col}_vs_{y_col}")

elif plot_type == "Heatmap":
    st.subheader("🔥 Heatmap")
    st.markdown("Visualize matrix data with colors")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        plot_mode = st.radio("Data Mode", ["Pivot Table", "Correlation Matrix", "Custom Matrix"])
        
        if plot_mode == "Pivot Table":
            if len(categorical_cols) < 2 or len(numeric_cols) < 1:
                st.error("Need at least 2 categorical columns and 1 numeric column")
                st.stop()
            
            index_col = st.selectbox("Rows", categorical_cols)
            columns_col = st.selectbox("Columns", [c for c in categorical_cols if c != index_col])
            values_col = st.selectbox("Values", numeric_cols)
            aggfunc = st.selectbox("Aggregation", ["mean", "sum", "count", "median", "min", "max"])
        
        elif plot_mode == "Correlation Matrix":
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns")
                st.stop()
            
            selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])
            method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
        
        cmap = st.selectbox("Color Map", MATPLOTLIB_COLORMAPS, index=0)
        annot = st.checkbox("Show Values", value=True)
        fmt = st.selectbox("Number Format", [".2f", ".1f", ".0f", ".2g"])
        linewidths = st.slider("Cell Spacing", 0.0, 2.0, 0.5, 0.1)
        center = st.checkbox("Center colormap at 0", value=False)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        if plot_mode == "Pivot Table":
            pivot_data = df.pivot_table(values=values_col, index=index_col, 
                                       columns=columns_col, aggfunc=aggfunc)
            sns.heatmap(pivot_data, annot=annot, fmt=fmt, cmap=cmap, 
                       linewidths=linewidths, center=0 if center else None, ax=ax)
        
        elif plot_mode == "Correlation Matrix":
            if len(selected_cols) < 2:
                st.error("Select at least 2 columns")
                st.stop()
            corr = df[selected_cols].corr(method=method)
            sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, 
                       linewidths=linewidths, center=0 if center else None, 
                       vmin=-1, vmax=1, ax=ax)
        
        else:
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns")
                st.stop()
            data_matrix = df[numeric_cols].head(50)
            sns.heatmap(data_matrix.T, annot=annot, fmt=fmt, cmap=cmap, 
                       linewidths=linewidths, center=0 if center else None, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"heatmap_{plot_mode}")

elif plot_type == "Regression Plot":
    st.subheader("📉 Regression Plot")
    st.markdown("Scatter plot with regression line")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns")
            st.stop()
        
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
        order = st.slider("Polynomial Order", 1, 5, 1)
        ci = st.slider("Confidence Interval (%)", 0, 99, 95)
        scatter = st.checkbox("Show Scatter Points", value=True)
        robust = st.checkbox("Robust Regression", value=False)
        truncate = st.checkbox("Truncate Regression Line", value=False)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        sns.regplot(data=df, x=x_col, y=y_col, order=order, ci=ci, 
                   scatter=scatter, robust=robust, truncate=truncate,
                   color=sns.color_palette(config['palette'])[0], ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"regression_{x_col}_vs_{y_col}")

elif plot_type == "Hexbin Plot":
    st.subheader("🔷 Hexbin Plot")
    st.markdown("2D histogram using hexagonal bins")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns")
            st.stop()
        
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
        gridsize = st.slider("Grid Size", 10, 50, 20)
        cmap = st.selectbox("Color Map", MATPLOTLIB_COLORMAPS, index=0)
        reduce_func = st.selectbox("Aggregation", ["mean", "sum", "min", "max", "count"])
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        reduce_c = None
        if reduce_func == "mean":
            reduce_c = np.mean
        elif reduce_func == "sum":
            reduce_c = np.sum
        elif reduce_func == "min":
            reduce_c = np.min
        elif reduce_func == "max":
            reduce_c = np.max
        elif reduce_func == "count":
            reduce_c = len
        
        hb = ax.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap=cmap, 
                      reduce_C_function=reduce_c, mincnt=1)
        plt.colorbar(hb, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"hexbin_{x_col}_vs_{y_col}")

elif plot_type == "Joint Plot":
    st.subheader("🎯 Joint Plot")
    st.markdown("Bivariate plot with marginal distributions")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns")
            st.stop()
        
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
        kind = st.selectbox("Plot Kind", ["scatter", "kde", "hex", "reg", "resid"])
        hue_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
    
    with col1:
        if hue_col and kind in ["scatter", "kde"]:
            g = sns.jointplot(data=df, x=x_col, y=y_col, kind=kind, hue=hue_col, 
                            palette=config['palette'], height=config['figsize'][1])
        else:
            g = sns.jointplot(data=df, x=x_col, y=y_col, kind=kind, 
                            color=sns.color_palette(config['palette'])[0], 
                            height=config['figsize'][1])
        
        if config.get('title'):
            g.fig.suptitle(config['title'], fontsize=14, fontweight='bold')
            g.fig.tight_layout()
        
        st.pyplot(g.fig)
        plt.close()
    
    download_plot(g.fig, f"joint_{x_col}_vs_{y_col}")

elif plot_type == "Residual Plot":
    st.subheader("📐 Residual Plot")
    st.markdown("Plot residuals of regression to check model fit")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns")
            st.stop()
        
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
        order = st.slider("Polynomial Order", 1, 5, 1)
        lowess = st.checkbox("Add LOWESS Smoother", value=True)
        robust = st.checkbox("Robust Regression", value=False)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        sns.residplot(data=df, x=x_col, y=y_col, order=order, lowess=lowess,
                     robust=robust, color=sns.color_palette(config['palette'])[0], ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"residual_{x_col}_vs_{y_col}")
