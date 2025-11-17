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
    get_categorical_columns, get_all_columns, get_display_data
)

st.title("📈 1D Visualizations")
st.markdown("Create one-dimensional plots to visualize distributions and single variables")

if not check_data_loaded():
    st.stop()

df = get_display_data()

numeric_cols = get_numeric_columns(df)
categorical_cols = get_categorical_columns(df)
all_cols = get_all_columns(df)

plot_type = st.selectbox(
    "Select Plot Type",
    ["Histogram", "KDE Plot", "Box Plot", "Violin Plot", "Strip Plot", "Swarm Plot", "Count Plot", "ECDF Plot"]
)

config = get_plot_customization_sidebar()

st.markdown("---")

if plot_type == "Histogram":
    st.subheader("📊 Histogram")
    st.markdown("Show the distribution of a numeric variable using bars")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the dataset")
            st.stop()
        
        column = st.selectbox("Select Column", numeric_cols)
        bins = st.slider("Number of Bins", 5, 100, 30)
        kde = st.checkbox("Show KDE", value=False)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        stat = st.selectbox("Statistic", ["count", "frequency", "probability", "percent", "density"])
        multiple = st.selectbox("Multiple groups display", ["layer", "dodge", "stack", "fill"])
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        if hue_col:
            sns.histplot(data=df, x=column, bins=bins, kde=kde, hue=hue_col, 
                        palette=config['palette'], stat=stat, multiple=multiple, ax=ax)
        else:
            sns.histplot(data=df, x=column, bins=bins, kde=kde, 
                        color=sns.color_palette(config['palette'])[0], stat=stat, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"histogram_{column}")

elif plot_type == "KDE Plot":
    st.subheader("📉 KDE Plot (Kernel Density Estimate)")
    st.markdown("Smooth estimate of the probability density function")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the dataset")
            st.stop()
        
        column = st.selectbox("Select Column", numeric_cols)
        fill = st.checkbox("Fill Area", value=True)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        bw_adjust = st.slider("Bandwidth Adjustment", 0.1, 3.0, 1.0, 0.1)
        common_norm = st.checkbox("Common Normalization", value=True)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        if hue_col:
            sns.kdeplot(data=df, x=column, fill=fill, hue=hue_col, 
                       palette=config['palette'], bw_adjust=bw_adjust, 
                       common_norm=common_norm, ax=ax)
        else:
            sns.kdeplot(data=df, x=column, fill=fill, 
                       color=sns.color_palette(config['palette'])[0], 
                       bw_adjust=bw_adjust, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"kde_{column}")

elif plot_type == "Box Plot":
    st.subheader("📦 Box Plot")
    st.markdown("Show distribution with quartiles and outliers")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the dataset")
            st.stop()
        
        column = st.selectbox("Select Numeric Column", numeric_cols)
        x_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        orient = st.radio("Orientation", ["Vertical", "Horizontal"])
        show_points = st.checkbox("Show All Points", value=False)
        notch = st.checkbox("Show Notch", value=False)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        if x_col:
            if orient == "Vertical":
                sns.boxplot(data=df, x=x_col, y=column, palette=config['palette'], 
                           notch=notch, ax=ax)
                if show_points:
                    sns.stripplot(data=df, x=x_col, y=column, color='black', 
                                 alpha=0.3, size=3, ax=ax)
            else:
                sns.boxplot(data=df, y=x_col, x=column, palette=config['palette'], 
                           notch=notch, ax=ax)
                if show_points:
                    sns.stripplot(data=df, y=x_col, x=column, color='black', 
                                 alpha=0.3, size=3, ax=ax)
        else:
            if orient == "Vertical":
                sns.boxplot(data=df, y=column, color=sns.color_palette(config['palette'])[0], 
                           notch=notch, ax=ax)
                if show_points:
                    sns.stripplot(data=df, y=column, color='black', 
                                 alpha=0.3, size=3, ax=ax)
            else:
                sns.boxplot(data=df, x=column, color=sns.color_palette(config['palette'])[0], 
                           notch=notch, ax=ax)
                if show_points:
                    sns.stripplot(data=df, x=column, color='black', 
                                 alpha=0.3, size=3, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"boxplot_{column}")

elif plot_type == "Violin Plot":
    st.subheader("🎻 Violin Plot")
    st.markdown("Combination of box plot and KDE")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the dataset")
            st.stop()
        
        column = st.selectbox("Select Numeric Column", numeric_cols)
        x_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        orient = st.radio("Orientation", ["Vertical", "Horizontal"])
        split = st.checkbox("Split Violins", value=False)
        inner = st.selectbox("Inner Display", ["box", "quartile", "point", "stick", None])
        bw_adjust = st.slider("Bandwidth Adjustment", 0.1, 3.0, 1.0, 0.1)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        if x_col:
            if orient == "Vertical":
                sns.violinplot(data=df, x=x_col, y=column, palette=config['palette'], 
                              split=split, inner=inner, bw_adjust=bw_adjust, ax=ax)
            else:
                sns.violinplot(data=df, y=x_col, x=column, palette=config['palette'], 
                              split=split, inner=inner, bw_adjust=bw_adjust, ax=ax)
        else:
            if orient == "Vertical":
                sns.violinplot(data=df, y=column, color=sns.color_palette(config['palette'])[0], 
                              inner=inner, bw_adjust=bw_adjust, ax=ax)
            else:
                sns.violinplot(data=df, x=column, color=sns.color_palette(config['palette'])[0], 
                              inner=inner, bw_adjust=bw_adjust, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"violin_{column}")

elif plot_type == "Strip Plot":
    st.subheader("📍 Strip Plot")
    st.markdown("Show individual data points")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the dataset")
            st.stop()
        
        column = st.selectbox("Select Numeric Column", numeric_cols)
        x_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        orient = st.radio("Orientation", ["Vertical", "Horizontal"])
        jitter = st.slider("Jitter Amount", 0.0, 0.5, 0.2, 0.05)
        size = st.slider("Point Size", 1, 10, 5)
        alpha = st.slider("Transparency", 0.1, 1.0, 0.7, 0.1)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        if x_col:
            if orient == "Vertical":
                sns.stripplot(data=df, x=x_col, y=column, palette=config['palette'], 
                             jitter=jitter, size=size, alpha=alpha, ax=ax)
            else:
                sns.stripplot(data=df, y=x_col, x=column, palette=config['palette'], 
                             jitter=jitter, size=size, alpha=alpha, ax=ax)
        else:
            if orient == "Vertical":
                sns.stripplot(data=df, y=column, color=sns.color_palette(config['palette'])[0], 
                             jitter=jitter, size=size, alpha=alpha, ax=ax)
            else:
                sns.stripplot(data=df, x=column, color=sns.color_palette(config['palette'])[0], 
                             jitter=jitter, size=size, alpha=alpha, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"strip_{column}")

elif plot_type == "Swarm Plot":
    st.subheader("🐝 Swarm Plot")
    st.markdown("Show all points without overlap")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the dataset")
            st.stop()
        
        if len(df) > 1000:
            st.warning("⚠️ Swarm plots work best with smaller datasets (<1000 rows). Consider using Strip Plot for larger datasets.")
        
        column = st.selectbox("Select Numeric Column", numeric_cols)
        x_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        orient = st.radio("Orientation", ["Vertical", "Horizontal"])
        size = st.slider("Point Size", 1, 10, 5)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        try:
            if x_col:
                if orient == "Vertical":
                    sns.swarmplot(data=df, x=x_col, y=column, palette=config['palette'], 
                                 size=size, ax=ax)
                else:
                    sns.swarmplot(data=df, y=x_col, x=column, palette=config['palette'], 
                                 size=size, ax=ax)
            else:
                if orient == "Vertical":
                    sns.swarmplot(data=df, y=column, color=sns.color_palette(config['palette'])[0], 
                                 size=size, ax=ax)
                else:
                    sns.swarmplot(data=df, x=column, color=sns.color_palette(config['palette'])[0], 
                                 size=size, ax=ax)
            
            apply_plot_formatting(fig, ax, config)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating swarm plot: {str(e)}. Try reducing the dataset size or use Strip Plot instead.")
        
        plt.close()
    
    download_plot(fig, f"swarm_{column}")

elif plot_type == "Count Plot":
    st.subheader("🔢 Count Plot")
    st.markdown("Show counts of categorical variables")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(categorical_cols) == 0:
            st.error("No categorical columns found in the dataset")
            st.stop()
        
        column = st.selectbox("Select Categorical Column", categorical_cols)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        orient = st.radio("Orientation", ["Vertical", "Horizontal"])
        order = st.checkbox("Sort by frequency", value=False)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        order_list = None
        if order:
            order_list = df[column].value_counts().index.tolist()
        
        if orient == "Vertical":
            sns.countplot(data=df, x=column, hue=hue_col, palette=config['palette'], 
                         order=order_list, ax=ax)
            ax.tick_params(axis='x', rotation=45)
        else:
            sns.countplot(data=df, y=column, hue=hue_col, palette=config['palette'], 
                         order=order_list, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"countplot_{column}")

elif plot_type == "ECDF Plot":
    st.subheader("📊 ECDF Plot (Empirical Cumulative Distribution)")
    st.markdown("Show cumulative distribution of values")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the dataset")
            st.stop()
        
        column = st.selectbox("Select Column", numeric_cols)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        complementary = st.checkbox("Show complementary CDF", value=False)
    
    with col1:
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        if hue_col:
            sns.ecdfplot(data=df, x=column, hue=hue_col, palette=config['palette'], 
                        complementary=complementary, ax=ax)
        else:
            sns.ecdfplot(data=df, x=column, color=sns.color_palette(config['palette'])[0], 
                        complementary=complementary, ax=ax)
        
        apply_plot_formatting(fig, ax, config)
        st.pyplot(fig)
        plt.close()
    
    download_plot(fig, f"ecdf_{column}")
