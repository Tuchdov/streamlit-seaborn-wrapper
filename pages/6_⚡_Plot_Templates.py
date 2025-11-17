import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from utils import (
    check_data_loaded, get_numeric_columns, 
    get_categorical_columns, download_plot
)

st.title("⚡ Plot Templates & Presets")
st.markdown("Quick-start templates for common visualization patterns")

if not check_data_loaded():
    st.warning("⚠️ No data loaded. Please upload data or use sample data from the Home page to use templates.")
    st.stop()

df = st.session_state.data
numeric_cols = get_numeric_columns(df)
categorical_cols = get_categorical_columns(df)

st.sidebar.markdown("### 🎨 Template Categories")
template_category = st.sidebar.radio(
    "Select Category",
    ["Statistical Overview", "Comparison Templates", "Correlation Analysis", 
     "Distribution Study", "Time Series", "Multi-Panel Layouts"]
)

st.markdown("---")

if template_category == "Statistical Overview":
    st.header("📊 Statistical Overview Dashboard")
    st.markdown("Generate a comprehensive statistical overview of your data")
    
    if len(numeric_cols) < 1:
        st.error("Need at least 1 numeric column for statistical overview")
        st.stop()
    
    selected_col = st.selectbox("Select Variable to Analyze", numeric_cols)
    group_by = st.selectbox("Group by (optional)", [None] + categorical_cols)
    
    if st.button("Generate Overview", type="primary"):
        st.markdown("### 📈 Statistical Overview Results")
        
        if group_by:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            sns.histplot(data=df, x=selected_col, hue=group_by, kde=True, ax=axes[0, 0], palette='Set2')
            axes[0, 0].set_title(f'Distribution of {selected_col} by {group_by}', fontweight='bold')
            
            sns.boxplot(data=df, x=group_by, y=selected_col, palette='Set2', ax=axes[0, 1])
            axes[0, 1].set_title(f'{selected_col} by {group_by}', fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            sns.violinplot(data=df, x=group_by, y=selected_col, palette='Set2', ax=axes[1, 0])
            axes[1, 0].set_title(f'{selected_col} Distribution (Violin)', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            summary_stats = df.groupby(group_by)[selected_col].describe()
            axes[1, 1].axis('off')
            table_text = summary_stats.round(2).to_string()
            axes[1, 1].text(0.1, 0.9, f'Summary Statistics:\n\n{table_text}', 
                           verticalalignment='top', fontfamily='monospace', fontsize=9)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            sns.histplot(data=df, x=selected_col, kde=True, ax=axes[0, 0], color='steelblue')
            axes[0, 0].set_title(f'Distribution of {selected_col}', fontweight='bold')
            
            sns.boxplot(data=df, y=selected_col, ax=axes[0, 1], color='lightblue')
            axes[0, 1].set_title(f'Box Plot of {selected_col}', fontweight='bold')
            
            sns.kdeplot(data=df, x=selected_col, fill=True, ax=axes[1, 0], color='coral')
            axes[1, 0].set_title(f'Density Plot of {selected_col}', fontweight='bold')
            
            stats = df[selected_col].describe()
            axes[1, 1].axis('off')
            stats_text = f"""
Summary Statistics:
Mean: {stats['mean']:.2f}
Median: {stats['50%']:.2f}
Std Dev: {stats['std']:.2f}
Min: {stats['min']:.2f}
Max: {stats['max']:.2f}
Q1: {stats['25%']:.2f}
Q3: {stats['75%']:.2f}
            """
            axes[1, 1].text(0.1, 0.9, stats_text, verticalalignment='top', 
                           fontfamily='monospace', fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        download_plot(fig, f"statistical_overview_{selected_col}")

elif template_category == "Comparison Templates":
    st.header("⚖️ Comparison Analysis")
    st.markdown("Compare groups, categories, or time periods")
    
    comparison_type = st.selectbox(
        "Comparison Type",
        ["Group Comparison", "Before/After Comparison", "Multi-Group Analysis"]
    )
    
    if comparison_type == "Group Comparison":
        st.subheader("📊 Two-Group Comparison")
        
        if len(categorical_cols) < 1 or len(numeric_cols) < 1:
            st.error("Need at least 1 categorical and 1 numeric column")
            st.stop()
        
        cat_col = st.selectbox("Grouping Variable", categorical_cols)
        num_col = st.selectbox("Numeric Variable", numeric_cols)
        
        if st.button("Generate Comparison", type="primary"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            sns.barplot(data=df, x=cat_col, y=num_col, palette='viridis', ax=axes[0], errorbar='sd')
            axes[0].set_title(f'Mean {num_col} by {cat_col}', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            
            sns.boxplot(data=df, x=cat_col, y=num_col, palette='viridis', ax=axes[1])
            axes[1].set_title(f'Distribution by {cat_col}', fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            
            sns.violinplot(data=df, x=cat_col, y=num_col, palette='viridis', ax=axes[2])
            axes[2].set_title(f'Violin Plot by {cat_col}', fontweight='bold')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            download_plot(fig, f"group_comparison_{cat_col}_{num_col}")
    
    elif comparison_type == "Multi-Group Analysis":
        st.subheader("📊 Multi-Factor Comparison")
        
        if len(categorical_cols) < 2 or len(numeric_cols) < 1:
            st.error("Need at least 2 categorical and 1 numeric column")
            st.stop()
        
        cat_col1 = st.selectbox("Primary Grouping", categorical_cols)
        cat_col2 = st.selectbox("Secondary Grouping", [c for c in categorical_cols if c != cat_col1])
        num_col = st.selectbox("Numeric Variable", numeric_cols)
        
        if st.button("Generate Analysis", type="primary"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            sns.barplot(data=df, x=cat_col1, y=num_col, hue=cat_col2, palette='Set2', ax=axes[0])
            axes[0].set_title(f'{num_col} by {cat_col1} and {cat_col2}', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].legend(title=cat_col2)
            
            pivot_data = df.pivot_table(values=num_col, index=cat_col1, columns=cat_col2, aggfunc='mean')
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1])
            axes[1].set_title(f'Mean {num_col} Heatmap', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            download_plot(fig, f"multi_group_{cat_col1}_{cat_col2}")

elif template_category == "Correlation Analysis":
    st.header("🔗 Correlation Analysis")
    st.markdown("Explore relationships between variables")
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for correlation analysis")
        st.stop()
    
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Full Correlation Matrix", "Pairwise Relationships", "Target Correlation"]
    )
    
    if analysis_type == "Full Correlation Matrix":
        selected_cols = st.multiselect("Select Variables", numeric_cols, 
                                      default=numeric_cols[:min(8, len(numeric_cols))])
        method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
        
        if len(selected_cols) >= 2 and st.button("Generate Analysis", type="primary"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            corr = df[selected_cols].corr(method=method)
            
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                       vmin=-1, vmax=1, square=True, ax=axes[0])
            axes[0].set_title(f'{method.capitalize()} Correlation Matrix', fontweight='bold')
            
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, vmin=-1, vmax=1, square=True, ax=axes[1])
            axes[1].set_title('Lower Triangle Only', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("### 🔍 Strong Correlations")
            strong_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.5:
                        strong_corr.append({
                            'Variable 1': corr.columns[i],
                            'Variable 2': corr.columns[j],
                            'Correlation': f"{corr.iloc[i, j]:.3f}"
                        })
            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
            else:
                st.info("No strong correlations (|r| > 0.5) found")
            
            download_plot(fig, f"correlation_matrix_{method}")
    
    elif analysis_type == "Pairwise Relationships":
        selected_cols = st.multiselect("Select Variables (2-6 recommended)", numeric_cols, 
                                      default=numeric_cols[:min(4, len(numeric_cols))])
        hue_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
        
        if len(selected_cols) >= 2 and st.button("Generate Pair Plot", type="primary"):
            if hue_col:
                g = sns.pairplot(df[selected_cols + [hue_col]], hue=hue_col, 
                               palette='husl', diag_kind='kde', height=2.5)
            else:
                g = sns.pairplot(df[selected_cols], diag_kind='kde', height=2.5, 
                               palette='husl')
            
            g.fig.suptitle('Pairwise Relationships', y=1.01, fontweight='bold')
            st.pyplot(g.fig)
            plt.close()
            
            download_plot(g.fig, "pairplot")

elif template_category == "Distribution Study":
    st.header("📊 Distribution Analysis")
    st.markdown("Comprehensive distribution exploration")
    
    if len(numeric_cols) < 1:
        st.error("Need at least 1 numeric column")
        st.stop()
    
    study_type = st.selectbox(
        "Study Type",
        ["Single Variable Deep Dive", "Multi-Variable Comparison", "Distribution by Groups"]
    )
    
    if study_type == "Single Variable Deep Dive":
        var = st.selectbox("Select Variable", numeric_cols)
        
        if st.button("Generate Analysis", type="primary"):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            sns.histplot(data=df, x=var, kde=False, bins=30, ax=axes[0, 0], color='steelblue')
            axes[0, 0].set_title('Histogram', fontweight='bold')
            
            sns.histplot(data=df, x=var, kde=True, bins=30, ax=axes[0, 1], color='coral')
            axes[0, 1].set_title('Histogram + KDE', fontweight='bold')
            
            sns.kdeplot(data=df, x=var, fill=True, ax=axes[0, 2], color='green')
            axes[0, 2].set_title('KDE Plot', fontweight='bold')
            
            sns.boxplot(data=df, y=var, ax=axes[1, 0], color='lightblue')
            axes[1, 0].set_title('Box Plot', fontweight='bold')
            
            sns.violinplot(data=df, y=var, ax=axes[1, 1], color='plum')
            axes[1, 1].set_title('Violin Plot', fontweight='bold')
            
            sns.ecdfplot(data=df, x=var, ax=axes[1, 2], color='purple')
            axes[1, 2].set_title('ECDF Plot', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            download_plot(fig, f"distribution_study_{var}")
    
    elif study_type == "Distribution by Groups":
        var = st.selectbox("Numeric Variable", numeric_cols)
        group = st.selectbox("Group by", categorical_cols)
        
        if st.button("Generate Analysis", type="primary"):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            sns.histplot(data=df, x=var, hue=group, kde=True, ax=axes[0, 0], palette='Set2')
            axes[0, 0].set_title(f'{var} Distribution by {group}', fontweight='bold')
            
            sns.boxplot(data=df, x=group, y=var, palette='Set2', ax=axes[0, 1])
            axes[0, 1].set_title('Box Plot Comparison', fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            sns.violinplot(data=df, x=group, y=var, palette='Set2', ax=axes[1, 0])
            axes[1, 0].set_title('Violin Plot Comparison', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            sns.kdeplot(data=df, x=var, hue=group, fill=True, ax=axes[1, 1], palette='Set2')
            axes[1, 1].set_title('KDE Overlay', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            download_plot(fig, f"distribution_by_{group}")

elif template_category == "Time Series":
    st.header("📈 Time Series Visualization")
    st.markdown("Analyze temporal patterns and trends")
    
    st.info("💡 For best results, ensure you have a date/time column or sequential index")
    
    if len(numeric_cols) < 1:
        st.error("Need at least 1 numeric column")
        st.stop()
    
    value_col = st.selectbox("Value Column", numeric_cols)
    index_col = st.selectbox("Time/Index Column", ['Index (row number)'] + df.columns.tolist())
    group_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
    
    if st.button("Generate Time Series Analysis", type="primary"):
        if index_col == 'Index (row number)':
            x_data = df.index
            x_label = 'Index'
        else:
            x_data = df[index_col]
            x_label = index_col
        
        if group_col:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            for group in df[group_col].unique():
                mask = df[group_col] == group
                axes[0].plot(x_data[mask], df[value_col][mask], marker='o', 
                           label=group, alpha=0.7)
            axes[0].set_title(f'{value_col} Over Time by {group_col}', fontweight='bold')
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel(value_col)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            sns.boxplot(data=df, x=group_col, y=value_col, palette='Set3', ax=axes[1])
            axes[1].set_title(f'{value_col} Distribution by {group_col}', fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            axes[0].plot(x_data, df[value_col], marker='o', color='steelblue', linewidth=2)
            axes[0].set_title(f'{value_col} Over Time', fontweight='bold')
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel(value_col)
            axes[0].grid(True, alpha=0.3)
            
            window = max(3, len(df) // 20)
            rolling_mean = df[value_col].rolling(window=window).mean()
            axes[0].plot(x_data, rolling_mean, color='red', linewidth=2, 
                        label=f'{window}-point Moving Average')
            axes[0].legend()
            
            sns.histplot(data=df, x=value_col, kde=True, ax=axes[1], color='coral')
            axes[1].set_title(f'{value_col} Distribution', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        download_plot(fig, f"timeseries_{value_col}")

elif template_category == "Multi-Panel Layouts":
    st.header("🎨 Multi-Panel Dashboard Layouts")
    st.markdown("Create comprehensive multi-plot dashboards")
    
    layout_type = st.selectbox(
        "Layout Type",
        ["2x2 Grid", "3x2 Grid", "Custom Overview"]
    )
    
    if layout_type == "2x2 Grid":
        st.markdown("**Select 4 different visualizations**")
        
        col1, col2 = st.columns(2)
        with col1:
            plot1_var = st.selectbox("Top-Left: Numeric Variable", numeric_cols, key='p1')
            plot2_var = st.selectbox("Top-Right: Numeric Variable", 
                                    [c for c in numeric_cols if c != plot1_var], key='p2')
        with col2:
            if len(categorical_cols) > 0:
                plot3_cat = st.selectbox("Bottom-Left: Category", categorical_cols, key='p3')
                plot3_num = st.selectbox("Bottom-Left: Value", numeric_cols, key='p3n')
                plot4_cat = st.selectbox("Bottom-Right: Category", categorical_cols, key='p4')
                plot4_num = st.selectbox("Bottom-Right: Value", numeric_cols, key='p4n')
            else:
                st.warning("Need categorical columns for bottom panels")
        
        if len(categorical_cols) > 0 and st.button("Generate Dashboard", type="primary"):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            sns.histplot(data=df, x=plot1_var, kde=True, ax=axes[0, 0], color='steelblue')
            axes[0, 0].set_title(f'Distribution: {plot1_var}', fontweight='bold')
            
            sns.boxplot(data=df, y=plot2_var, ax=axes[0, 1], color='coral')
            axes[0, 1].set_title(f'Box Plot: {plot2_var}', fontweight='bold')
            
            sns.barplot(data=df, x=plot3_cat, y=plot3_num, palette='viridis', ax=axes[1, 0])
            axes[1, 0].set_title(f'{plot3_num} by {plot3_cat}', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            sns.violinplot(data=df, x=plot4_cat, y=plot4_num, palette='muted', ax=axes[1, 1])
            axes[1, 1].set_title(f'{plot4_num} by {plot4_cat}', fontweight='bold')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            download_plot(fig, "dashboard_2x2")

st.markdown("---")
st.info("💡 **Tip:** Templates provide quick insights. For detailed customization, visit the specific plot type pages (1D, 2D, 3D, Specialty).")
