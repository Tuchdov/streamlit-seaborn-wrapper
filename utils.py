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
