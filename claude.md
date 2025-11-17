# Overview

This is a Streamlit-based visualization dashboard that enables users to create professional data visualizations without writing code. The application provides an interactive interface for uploading datasets (CSV, Excel, JSON) and generating various types of plots using Seaborn and Matplotlib libraries. Users can customize visualizations through an intuitive UI and explore data through 1D, 2D, 3D, and specialty plot types.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Technology Stack**: Streamlit (Python web framework)

The application uses Streamlit's multi-page architecture with a main entry point (`app.py`) and specialized visualization pages organized in a `/pages` directory. This design pattern enables:

- Clean separation of concerns with each plot category in its own page
- Streamlit's native navigation sidebar for page switching
- Session state management for persistent data across page transitions

**Rationale**: Streamlit was chosen for its rapid development capabilities and built-in UI components that eliminate the need for HTML/CSS/JavaScript. The multi-page structure keeps code organized and maintainable as new visualization types are added.

## State Management

**Solution**: Streamlit session state

The application maintains user data and filename in `st.session_state`:
- `st.session_state.data` - Stores the uploaded DataFrame
- `st.session_state.filename` - Stores the original filename

**Rationale**: Session state ensures uploaded data persists across page navigation without requiring re-uploads. This provides a seamless user experience when switching between different visualization pages.

## Visualization Architecture

**Dual Library Approach**: Seaborn + Matplotlib + Plotly

- **Seaborn**: Primary library for statistical visualizations (1D, 2D, specialty plots)
- **Matplotlib**: Low-level plotting for custom 3D visualizations and fine-grained control
- **Plotly**: Interactive 3D plots where user engagement is beneficial

**Rationale**: Seaborn provides high-level statistical graphics with minimal code, while Matplotlib offers flexibility for custom visualizations. Plotly adds interactivity for complex 3D plots where rotation and zoom enhance understanding.

## Code Organization Pattern

**Utility-Based Architecture**: Shared utilities in `utils.py`

Common functions are centralized including:
- `get_plot_customization_sidebar()` - Standardized plot styling controls
- `apply_plot_formatting()` - Consistent plot appearance across visualization types
- `check_data_loaded()` - Data validation guard
- Column type helpers (`get_numeric_columns`, `get_categorical_columns`)

**Rationale**: This DRY (Don't Repeat Yourself) approach ensures consistent UX across all plot types and simplifies maintenance. All pages import from utils, creating a shared component library.

## Page Structure

**Specialized Page Pattern**: Each visualization category gets its own page

1. **1D Plots** (`1_📈_1D_Plots.py`) - Histograms, KDE, box plots, violin plots, strip plots, swarm plots, count plots, ECDF plots
2. **2D Plots** (`2_📊_2D_Plots.py`) - Scatter, line, bar, heatmaps, regression plots, hexbin, joint plots, residual plots
3. **3D Plots** (`3_🎲_3D_Plots.py`) - 3D scatter, surface, contour plots, 3D line, 3D bar, wireframe (with Plotly interactivity)
4. **Specialty Plots** (`4_✨_Specialty_Plots.py`) - Pair plots, correlation matrices, cluster maps, cat plots, point plots, factor plots, Andrews curves
5. **Cheat Sheet** (`5_📚_Cheat_Sheet.py`) - Comprehensive documentation with visual examples, when-to-use guidance, best practices, and official documentation links for each plot type
6. **Templates** (`6_⚡_Plot_Templates.py`) - Quick-start templates including Statistical Overview, Comparison Analysis, Correlation Analysis, Distribution Study, Time Series, and Multi-Panel Layouts

**Rationale**: Categorical separation makes the interface less overwhelming and allows users to quickly find the right plot type. Numeric prefixes ensure predictable page ordering in Streamlit's sidebar. The Cheat Sheet and Templates pages provide non-technical users with guidance and quick-start options.

## Data Input Strategy

**Multi-Format Support**: CSV, Excel (xlsx/xls), JSON

File processing logic in `app.py` uses pandas for format detection and loading:
- Extension-based routing to appropriate pandas reader
- Error handling with user-friendly messages
- Sample data option for exploration without uploads

**Rationale**: Supporting multiple formats reduces friction for users working with diverse data sources. Pandas provides robust, battle-tested parsers for all supported formats.

## Customization System

**Sidebar-Based Configuration**: All plot customization through Streamlit sidebar

Standardized controls include:
- Seaborn style selection (darkgrid, whitegrid, etc.)
- Color palette choosers (18+ preset palettes)
- Figure dimension sliders
- Title and axis label inputs

**Rationale**: Sidebar placement keeps the main area focused on visualization output while providing easy access to styling options. Consistent placement across pages creates predictable UX.

## Plot Export Mechanism

**Download Functionality**: Plots can be exported as image files

Implementation appears to use a `download_plot()` utility function (referenced but not fully shown in utils.py).

**Rationale**: Users need to extract visualizations for reports and presentations. Providing download capability makes the tool practical for professional use.

# External Dependencies

## Python Libraries

- **streamlit**: Web application framework and UI components
- **pandas**: Data manipulation and file parsing (CSV/Excel/JSON)
- **numpy**: Numerical operations and sample data generation
- **matplotlib**: Core plotting library and 3D visualization support
- **seaborn**: Statistical visualization library built on matplotlib
- **plotly**: Interactive 3D plotting (graph_objects and express modules)
- **mpl_toolkits.mplot3d**: Matplotlib's 3D plotting toolkit

## Data Format Support

- **CSV files**: Via pandas.read_csv()
- **Excel files** (.xlsx, .xls): Via pandas.read_excel()
- **JSON files**: Via pandas.read_json()

## Styling Resources

- **Seaborn palettes**: 18 built-in color schemes (deep, muted, pastel, Set1, Set2, viridis, etc.)
- **Seaborn styles**: 5 plot styles (darkgrid, whitegrid, dark, white, ticks)
- **Matplotlib colormaps**: 14+ color gradients for heatmaps and 3D plots

No external APIs, databases, or authentication services are currently integrated. The application operates entirely client-side with uploaded data stored in session state.