import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

SEABORN_PALETTES = [
    'deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind',
    'Set1', 'Set2', 'Set3', 'Paired', 'husl', 'hls',
    'coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
]

SEABORN_STYLES = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

MATPLOTLIB_COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'twilight', 'turbo', 'jet', 'rainbow', 'coolwarm',
    'RdYlBu', 'RdYlGn', 'Spectral', 'seismic'
]

def get_plot_customization_sidebar():
    st.sidebar.markdown("### 🎨 Plot Customization")
    
    style = st.sidebar.selectbox("Plot Style", SEABORN_STYLES, index=0)
    sns.set_style(style)
    
    palette = st.sidebar.selectbox("Color Palette", SEABORN_PALETTES, index=0)
    
    fig_width = st.sidebar.slider("Figure Width", 6, 20, 10)
    fig_height = st.sidebar.slider("Figure Height", 4, 15, 6)
    
    title = st.sidebar.text_input("Plot Title", "")
    xlabel = st.sidebar.text_input("X-axis Label", "")
    ylabel = st.sidebar.text_input("Y-axis Label", "")
    
    return {
        'style': style,
        'palette': palette,
        'figsize': (fig_width, fig_height),
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel
    }

def apply_plot_formatting(fig, ax, config):
    if config.get('title'):
        ax.set_title(config['title'], fontsize=14, fontweight='bold')
    if config.get('xlabel'):
        ax.set_xlabel(config['xlabel'], fontsize=12)
    if config.get('ylabel'):
        ax.set_ylabel(config['ylabel'], fontsize=12)
    plt.tight_layout()

def download_plot(fig, filename="plot"):
    st.markdown("### 💾 Download Plot")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
        buf_png.seek(0)
        st.download_button(
            label="📥 Download PNG",
            data=buf_png,
            file_name=f"{filename}.png",
            mime="image/png"
        )
    
    with col2:
        buf_svg = io.BytesIO()
        fig.savefig(buf_svg, format='svg', bbox_inches='tight')
        buf_svg.seek(0)
        st.download_button(
            label="📥 Download SVG",
            data=buf_svg,
            file_name=f"{filename}.svg",
            mime="image/svg+xml"
        )
    
    with col3:
        buf_pdf = io.BytesIO()
        fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
        buf_pdf.seek(0)
        st.download_button(
            label="📥 Download PDF",
            data=buf_pdf,
            file_name=f"{filename}.pdf",
            mime="application/pdf"
        )

def check_data_loaded():
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload data or use sample data from the Home page.")
        return False
    return True

def get_numeric_columns(df):
    import numpy as np
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_all_columns(df):
    return df.columns.tolist()

def add_data_filters(df):
    """
    Add interactive data filtering UI and return filtered dataframe.
    Allows filtering by categorical and numeric columns.
    """
    import streamlit as st
    import numpy as np

    with st.expander("🔍 Filter Data", expanded=False):
        st.markdown("Filter the dataset before creating visualizations")

        numeric_cols = get_numeric_columns(df)
        categorical_cols = get_categorical_columns(df)

        filtered_df = df.copy()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Categorical Filters**")
            if len(categorical_cols) > 0:
                for cat_col in categorical_cols:
                    unique_values = df[cat_col].dropna().unique().tolist()
                    if len(unique_values) > 0 and len(unique_values) <= 50:
                        selected_values = st.multiselect(
                            f"{cat_col}",
                            options=unique_values,
                            default=unique_values,
                            key=f"filter_cat_{cat_col}"
                        )
                        if len(selected_values) > 0:
                            filtered_df = filtered_df[filtered_df[cat_col].isin(selected_values)]
                    elif len(unique_values) > 50:
                        st.info(f"{cat_col}: Too many unique values ({len(unique_values)}) to filter")
            else:
                st.info("No categorical columns available")

        with col2:
            st.markdown("**Numeric Filters**")
            if len(numeric_cols) > 0:
                for num_col in numeric_cols:
                    col_min = float(df[num_col].min())
                    col_max = float(df[num_col].max())

                    if col_min != col_max:
                        selected_range = st.slider(
                            f"{num_col}",
                            min_value=col_min,
                            max_value=col_max,
                            value=(col_min, col_max),
                            key=f"filter_num_{num_col}"
                        )
                        filtered_df = filtered_df[
                            (filtered_df[num_col] >= selected_range[0]) &
                            (filtered_df[num_col] <= selected_range[1])
                        ]
            else:
                st.info("No numeric columns available")

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Original Rows", len(df))
        with col_b:
            st.metric("Filtered Rows", len(filtered_df))
        with col_c:
            percentage = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
            st.metric("Percentage", f"{percentage:.1f}%")

        if len(filtered_df) == 0:
            st.warning("⚠️ No data matches the current filters. Please adjust your filter settings.")
            return df

    return filtered_df
