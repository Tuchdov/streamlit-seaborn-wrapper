import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import sys
sys.path.append('..')
from utils import (
    get_plot_customization_sidebar, check_data_loaded, 
    get_numeric_columns, get_categorical_columns, MATPLOTLIB_COLORMAPS
)

st.title("🎲 3D Visualizations")
st.markdown("Create three-dimensional plots for advanced data exploration")

if not check_data_loaded():
    st.stop()

df = st.session_state.data

numeric_cols = get_numeric_columns(df)
categorical_cols = get_categorical_columns(df)

plot_type = st.selectbox(
    "Select Plot Type",
    ["3D Scatter Plot", "3D Surface Plot", "3D Line Plot", "Contour Plot", 
     "3D Bar Chart", "Wireframe Plot"]
)

st.sidebar.markdown("### 🎨 Plot Customization")
palette = st.sidebar.selectbox("Color Palette", ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                                                  'turbo', 'rainbow', 'jet'], index=0)
title = st.sidebar.text_input("Plot Title", "")
use_plotly = st.sidebar.checkbox("Use Interactive Plotly (where available)", value=True)

st.markdown("---")

if plot_type == "3D Scatter Plot":
    st.subheader("🔵 3D Scatter Plot")
    st.markdown("Visualize relationships between three variables")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 3:
            st.error("Need at least 3 numeric columns for 3D plots")
            st.stop()
        
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
        z_col = st.selectbox("Z-axis", [c for c in numeric_cols if c not in [x_col, y_col]])
        color_col = st.selectbox("Color by (optional)", [None] + categorical_cols + numeric_cols)
        size_col = st.selectbox("Size by (optional)", [None] + numeric_cols)
        
        if not use_plotly:
            marker_size = st.slider("Marker Size", 10, 200, 50)
            alpha = st.slider("Transparency", 0.1, 1.0, 0.7, 0.1)
            elev = st.slider("Elevation Angle", 0, 90, 30)
            azim = st.slider("Azimuth Angle", 0, 360, 45)
    
    with col1:
        if use_plotly:
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, 
                               color=color_col, size=size_col,
                               color_continuous_scale=palette,
                               title=title if title else None)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if color_col:
                if color_col in categorical_cols:
                    for cat in df[color_col].unique():
                        mask = df[color_col] == cat
                        ax.scatter(df[mask][x_col], df[mask][y_col], df[mask][z_col],
                                 label=cat, s=marker_size, alpha=alpha)
                    ax.legend()
                else:
                    scatter = ax.scatter(df[x_col], df[y_col], df[z_col], 
                                       c=df[color_col], cmap=palette, 
                                       s=marker_size, alpha=alpha)
                    plt.colorbar(scatter, ax=ax, label=color_col)
            else:
                ax.scatter(df[x_col], df[y_col], df[z_col], 
                         s=marker_size, alpha=alpha, c=sns.color_palette('deep')[0])
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(z_col)
            ax.view_init(elev=elev, azim=azim)
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()

elif plot_type == "3D Surface Plot":
    st.subheader("🏔️ 3D Surface Plot")
    st.markdown("Create a surface from gridded data")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        plot_mode = st.radio("Data Mode", ["From Columns", "Mathematical Function"])
        
        if plot_mode == "From Columns":
            if len(numeric_cols) < 3:
                st.error("Need at least 3 numeric columns")
                st.stop()
            
            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
            z_col = st.selectbox("Z-axis (values)", [c for c in numeric_cols if c not in [x_col, y_col]])
        else:
            func_type = st.selectbox("Function", 
                                    ["sin(x)*cos(y)", "x^2 + y^2", "sin(sqrt(x^2 + y^2))", 
                                     "exp(-(x^2 + y^2))", "x*y"])
            resolution = st.slider("Resolution", 20, 100, 50)
        
        cmap = st.selectbox("Color Map", MATPLOTLIB_COLORMAPS, index=0)
        
        if not use_plotly:
            elev = st.slider("Elevation Angle", 0, 90, 30)
            azim = st.slider("Azimuth Angle", 0, 360, 45)
    
    with col1:
        if plot_mode == "From Columns":
            pivot_data = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
            X = pivot_data.columns.values
            Y = pivot_data.index.values
            X, Y = np.meshgrid(X, Y)
            Z = pivot_data.values
        else:
            x = np.linspace(-5, 5, resolution)
            y = np.linspace(-5, 5, resolution)
            X, Y = np.meshgrid(x, y)
            
            if func_type == "sin(x)*cos(y)":
                Z = np.sin(X) * np.cos(Y)
            elif func_type == "x^2 + y^2":
                Z = X**2 + Y**2
            elif func_type == "sin(sqrt(x^2 + y^2))":
                Z = np.sin(np.sqrt(X**2 + Y**2))
            elif func_type == "exp(-(x^2 + y^2))":
                Z = np.exp(-(X**2 + Y**2))
            elif func_type == "x*y":
                Z = X * Y
        
        if use_plotly:
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale=cmap)])
            fig.update_layout(
                title=title if title else None,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9)
            plt.colorbar(surf, ax=ax, shrink=0.5)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=elev, azim=azim)
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()

elif plot_type == "3D Line Plot":
    st.subheader("📈 3D Line Plot")
    st.markdown("Draw lines in 3D space")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 3:
            st.error("Need at least 3 numeric columns")
            st.stop()
        
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
        z_col = st.selectbox("Z-axis", [c for c in numeric_cols if c not in [x_col, y_col]])
        color_col = st.selectbox("Color by (optional)", [None] + numeric_cols)
        
        if not use_plotly:
            linewidth = st.slider("Line Width", 1, 5, 2)
            elev = st.slider("Elevation Angle", 0, 90, 30)
            azim = st.slider("Azimuth Angle", 0, 360, 45)
    
    with col1:
        if use_plotly:
            fig = px.line_3d(df, x=x_col, y=y_col, z=z_col, 
                            color=color_col,
                            title=title if title else None)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if color_col:
                scatter = ax.plot(df[x_col], df[y_col], df[z_col], 
                                c=df[color_col], cmap=palette, linewidth=linewidth)
            else:
                ax.plot(df[x_col], df[y_col], df[z_col], linewidth=linewidth)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(z_col)
            ax.view_init(elev=elev, azim=azim)
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()

elif plot_type == "Contour Plot":
    st.subheader("🗺️ Contour Plot")
    st.markdown("Show contour lines of 3D data in 2D")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        plot_mode = st.radio("Data Mode", ["From Columns", "Mathematical Function"])
        
        if plot_mode == "From Columns":
            if len(numeric_cols) < 3:
                st.error("Need at least 3 numeric columns")
                st.stop()
            
            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
            z_col = st.selectbox("Z-axis (values)", [c for c in numeric_cols if c not in [x_col, y_col]])
        else:
            func_type = st.selectbox("Function", 
                                    ["sin(x)*cos(y)", "x^2 + y^2", "sin(sqrt(x^2 + y^2))", 
                                     "exp(-(x^2 + y^2))", "x*y"])
            resolution = st.slider("Resolution", 20, 100, 50)
        
        cmap = st.selectbox("Color Map", MATPLOTLIB_COLORMAPS, index=0)
        filled = st.checkbox("Filled Contours", value=True)
        levels = st.slider("Number of Levels", 5, 30, 10)
        show_labels = st.checkbox("Show Labels", value=True)
    
    with col1:
        if plot_mode == "From Columns":
            pivot_data = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
            X = pivot_data.columns.values
            Y = pivot_data.index.values
            X, Y = np.meshgrid(X, Y)
            Z = pivot_data.values
        else:
            x = np.linspace(-5, 5, resolution)
            y = np.linspace(-5, 5, resolution)
            X, Y = np.meshgrid(x, y)
            
            if func_type == "sin(x)*cos(y)":
                Z = np.sin(X) * np.cos(Y)
            elif func_type == "x^2 + y^2":
                Z = X**2 + Y**2
            elif func_type == "sin(sqrt(x^2 + y^2))":
                Z = np.sin(np.sqrt(X**2 + Y**2))
            elif func_type == "exp(-(x^2 + y^2))":
                Z = np.exp(-(X**2 + Y**2))
            elif func_type == "x*y":
                Z = X * Y
        
        if use_plotly:
            fig = go.Figure(data=go.Contour(x=X[0], y=Y[:, 0], z=Z, 
                                           colorscale=cmap, 
                                           contours_coloring='fill' if filled else 'lines',
                                           ncontours=levels))
            fig.update_layout(
                title=title if title else None,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if filled:
                cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
            else:
                cs = ax.contour(X, Y, Z, levels=levels, cmap=cmap)
            
            if show_labels:
                ax.clabel(cs, inline=True, fontsize=8)
            
            plt.colorbar(cs, ax=ax)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()

elif plot_type == "3D Bar Chart":
    st.subheader("📊 3D Bar Chart")
    st.markdown("Create bars in 3D space")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if len(numeric_cols) < 1:
            st.error("Need at least 1 numeric column")
            st.stop()
        
        z_col = st.selectbox("Values (Z-axis)", numeric_cols)
        max_bars = st.slider("Maximum bars to show", 10, 100, 50)
        cmap = st.selectbox("Color Map", MATPLOTLIB_COLORMAPS, index=0)
        elev = st.slider("Elevation Angle", 0, 90, 30)
        azim = st.slider("Azimuth Angle", 0, 360, 45)
    
    with col1:
        data_subset = df.head(max_bars)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.arange(len(data_subset))
        y = np.zeros(len(data_subset))
        z = np.zeros(len(data_subset))
        dx = np.ones(len(data_subset)) * 0.8
        dy = np.ones(len(data_subset)) * 0.8
        dz = data_subset[z_col].values
        
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(data_subset)))
        
        ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.8)
        
        ax.set_xlabel('Index')
        ax.set_ylabel('')
        ax.set_zlabel(z_col)
        ax.view_init(elev=elev, azim=azim)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        st.pyplot(fig)
        plt.close()

elif plot_type == "Wireframe Plot":
    st.subheader("🕸️ Wireframe Plot")
    st.markdown("Display a 3D wireframe surface")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        plot_mode = st.radio("Data Mode", ["From Columns", "Mathematical Function"])
        
        if plot_mode == "From Columns":
            if len(numeric_cols) < 3:
                st.error("Need at least 3 numeric columns")
                st.stop()
            
            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
            z_col = st.selectbox("Z-axis (values)", [c for c in numeric_cols if c not in [x_col, y_col]])
        else:
            func_type = st.selectbox("Function", 
                                    ["sin(x)*cos(y)", "x^2 + y^2", "sin(sqrt(x^2 + y^2))", 
                                     "exp(-(x^2 + y^2))", "x*y"])
            resolution = st.slider("Resolution", 20, 100, 30)
        
        cmap = st.selectbox("Color Map", MATPLOTLIB_COLORMAPS, index=0)
        elev = st.slider("Elevation Angle", 0, 90, 30)
        azim = st.slider("Azimuth Angle", 0, 360, 45)
    
    with col1:
        if plot_mode == "From Columns":
            pivot_data = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
            X = pivot_data.columns.values
            Y = pivot_data.index.values
            X, Y = np.meshgrid(X, Y)
            Z = pivot_data.values
        else:
            x = np.linspace(-5, 5, resolution)
            y = np.linspace(-5, 5, resolution)
            X, Y = np.meshgrid(x, y)
            
            if func_type == "sin(x)*cos(y)":
                Z = np.sin(X) * np.cos(Y)
            elif func_type == "x^2 + y^2":
                Z = X**2 + Y**2
            elif func_type == "sin(sqrt(x^2 + y^2))":
                Z = np.sin(np.sqrt(X**2 + Y**2))
            elif func_type == "exp(-(x^2 + y^2))":
                Z = np.exp(-(X**2 + Y**2))
            elif func_type == "x*y":
                Z = X * Y
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_wireframe(X, Y, Z, cmap=cmap, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
