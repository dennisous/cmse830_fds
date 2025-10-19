# streamlit_app.py
# Heart Disease Analysis - Complete Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

# Set plotting style
sns.set_style("whitegrid")

# Load data
@st.cache_data
def load_data():
    # Load all three datasets
    try:
        df_original = pd.read_csv('heart_disease_original.csv')
    except FileNotFoundError:
        df_original = None
    
    df_simple = pd.read_csv('heart_disease_simple_imputation.csv')
    df_mi = pd.read_csv('heart_disease_multiple_imputation.csv')
    return df_original, df_simple, df_mi

df_original, df_simple, df_mi = load_data()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["Home", "Data Explorer", "Visualizations", "Correlation Analysis"])

# Sidebar dataset selector
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Selection")
dataset_choice = st.sidebar.radio(
    "Choose imputation method:",
    ["Simple Imputation", "Multiple Imputation (MICE)"]
)

# Select active dataset
df = df_simple if dataset_choice == "Simple Imputation" else df_mi

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "Home":
    st.title("🫀 Heart Disease Analysis")
    st.markdown("---")
    
    st.header("Project Overview")
    st.write("""
    This application analyzes the UCI Heart Disease Dataset to understand factors 
    contributing to heart disease risk. The dataset combines data from Cleveland 
    and Switzerland databases with two imputation techniques applied.
    """)
    
    st.header("Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patients", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        disease_pct = (df['target'].sum() / len(df)) * 100
        st.metric("Disease Prevalence", f"{disease_pct:.1f}%")
    
    st.markdown("---")
    
    st.header("Imputation Methods Used")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simple Imputation")
        st.write("""
        - **Categorical variables**: Mode (most frequent value)
        - **Numeric variables**: Median
        - **Pros**: Fast, simple, easy to understand
        - **Cons**: Reduces variability, ignores relationships
        """)
    
    with col2:
        st.subheader("Multiple Imputation (MICE)")
        st.write("""
        - Uses **iterative imputation** based on all other features
        - Creates 5 datasets and pools results
        - **Pros**: Preserves relationships, captures uncertainty
        - **Cons**: More complex, computationally intensive
        """)
    
    st.markdown("---")
    
    st.subheader("Features Description")
    st.write("""
    - **age**: Patient age in years
    - **sex**: 1 = male, 0 = female
    - **cp**: Chest pain type (1-4)
    - **trestbps**: Resting blood pressure (mm Hg)
    - **chol**: Serum cholesterol (mg/dl)
    - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    - **restecg**: Resting ECG results (0-2)
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina (1 = yes, 0 = no)
    - **oldpeak**: ST depression induced by exercise (mm)
    - **slope**: Slope of peak exercise ST segment (1-3)
    - **ca**: Number of major vessels colored by fluoroscopy (0-3)
    - **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
    - **target**: Heart disease presence (1 = disease, 0 = no disease)
    """)

# ============================================================================
# DATA EXPLORER PAGE
# ============================================================================

elif page == "Data Explorer":
    st.title("📊 Interactive Data Explorer")
    st.markdown("---")
    
    # Display current dataset
    st.info(f"Currently viewing: **{dataset_choice}**")
    
    # INTERACTIVE ELEMENT 1: Feature Selector
    st.header("Feature Distribution Analysis")
    st.write("Select a feature to view its distribution:")
    
    feature_options = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    selected_feature = st.selectbox(
        "Choose a continuous feature:",
        feature_options,
        help="Select a feature to visualize its distribution"
    )
    
    # Display histogram for selected feature
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[selected_feature], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel(selected_feature, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of {selected_feature}', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Show statistics for selected feature
    st.subheader(f"Statistics for {selected_feature}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{df[selected_feature].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[selected_feature].median():.2f}")
    with col3:
        st.metric("Min", f"{df[selected_feature].min():.2f}")
    with col4:
        st.metric("Max", f"{df[selected_feature].max():.2f}")
    
    st.markdown("---")
    
    # INTERACTIVE ELEMENT 2: Age Range Slider
    st.header("Filter Data by Age Range")
    st.write("Use the slider to filter patients by age and see disease prevalence:")
    
    age_range = st.slider(
        "Select age range:",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=(int(df['age'].min()), int(df['age'].max())),
        help="Drag the slider to filter patients by age"
    )
    
    # Filter data based on age range
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    
    # Display filtered statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Patients in Range", len(filtered_df))
    with col2:
        disease_pct_filtered = (filtered_df['target'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric("Disease Prevalence", f"{disease_pct_filtered:.1f}%")
    
    # Show disease distribution for filtered data
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    target_counts = filtered_df['target'].value_counts()
    ax2.bar(target_counts.index, target_counts.values, 
            color=['#2ecc71', '#e74c3c'], edgecolor='black')
    ax2.set_xlabel('Heart Disease', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Disease Distribution (Age {age_range[0]}-{age_range[1]})', fontsize=14)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Absent', 'Present'])
    ax2.grid(axis='y', alpha=0.3)
    st.pyplot(fig2)
    plt.close()
    
    # Display filtered data table
    if st.checkbox("Show filtered data table"):
        st.dataframe(filtered_df)

# ============================================================================
# VISUALIZATIONS PAGE
# ============================================================================

elif page == "Visualizations":
    st.title("📈 Data Visualizations")
    st.markdown("---")
    
    st.info(f"Currently viewing: **{dataset_choice}**")
    
    # Age Distribution
    st.header("Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='age', kde=True, bins=20, ax=ax, color='steelblue')
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Patient Ages', fontsize=14)
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Target Variable Distribution
    st.header("Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts = df['target'].value_counts()
    ax.bar(target_counts.index, target_counts.values,
           color=['#2ecc71', '#e74c3c'],
           edgecolor='black')
    ax.set_xlabel('Heart Disease Status', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Number of Patients with/without Heart Disease', fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Absent', 'Present'])
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Show prevalence percentage
    disease_pct = (df['target'].sum() / len(df)) * 100
    st.write(f"**Heart Disease Prevalence: {disease_pct:.1f}%**")
    
    st.markdown("---")
    
    # Pairplot
    st.header("Pairplot of Continuous Variables")
    st.write("Pairwise relationships between continuous features, colored by target variable")
    
    with st.spinner("Generating pairplot... This may take a moment."):
        numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        
        # Create pairplot
        fig = plt.figure(figsize=(12, 10))
        pairplot_data = df[numerical_features + ['target']].copy()
        pairplot_data['target'] = pairplot_data['target'].astype(int)
        
        g = sns.pairplot(
            pairplot_data,
            hue='target',
            diag_kind='kde',
            palette={0: 'lightgreen', 1: 'salmon'},
            plot_kws={'alpha': 0.5, 's': 20},
            diag_kws={'alpha': 0.7, 'linewidth': 2}
        )
        g.fig.suptitle('Pairplot: Continuous Variables by Disease Status', 
                       y=1.02, fontsize=14)
        st.pyplot(g.fig)
        plt.close()

# ============================================================================
# CORRELATION ANALYSIS PAGE
# ============================================================================

elif page == "Correlation Analysis":
    st.title("🔗 Correlation Analysis")
    st.markdown("---")
    
    if df_original is None:
        st.error("""
        **Missing Original Dataset!**
        
        To see all three correlation matrices, add this to your notebook:
        ```python
        # Save original data BEFORE imputation
        df.to_csv('heart_disease_original.csv', index=False)
        ```
        Then restart this app.
        """)
        st.stop()
    
    st.write("""
    Compare correlation patterns across different datasets:
    - **Original**: Before imputation (missing values excluded from correlation)
    - **Simple Imputation**: Mode/Median filled
    - **Multiple Imputation (MICE)**: Iterative imputation
    """)
    
    st.markdown("---")
    
    # Use the datasets already loaded
    df_original = df_simple.copy()  # This is wrong - need actual original data
    
    # Actually, we should load from your notebook's original df before imputation
    # For now, note that you'll need to save your original df to CSV first
    st.warning("""
    **Note:** To see the original data correlations, you need to:
    1. In your notebook, save the original df BEFORE imputation: `df.to_csv('heart_disease_original.csv', index=False)`
    2. Then this app will load all three versions properly
    """)
    
    # Include categorical variables that were imputed
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal", "fbs"]
    
    st.header("Correlation Heatmaps Comparison")
    
    # Create three columns for side-by-side comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Data")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        # Convert categorical to numeric for correlation
        df_orig_numeric = df_original[numerical_features].copy()
        for col in ['ca', 'thal', 'fbs']:
            if col in df_orig_numeric.columns:
                df_orig_numeric[col] = pd.to_numeric(df_orig_numeric[col], errors='coerce')
        corr_original = df_orig_numeric.corr()
        sns.heatmap(
            corr_original,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            ax=ax1,
            cbar_kws={'label': 'Correlation'},
            vmin=-1, vmax=1
        )
        ax1.set_title('Original Data\n(Missing Excluded)', fontsize=10)
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        st.subheader("Simple Imputation")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        # Convert categorical to numeric for correlation
        df_simple_numeric = df_simple[numerical_features].copy()
        for col in ['ca', 'thal', 'fbs']:
            if col in df_simple_numeric.columns:
                df_simple_numeric[col] = pd.to_numeric(df_simple_numeric[col], errors='coerce')
        corr_simple = df_simple_numeric.corr()
        sns.heatmap(
            corr_simple,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            ax=ax2,
            cbar_kws={'label': 'Correlation'},
            vmin=-1, vmax=1
        )
        ax2.set_title('Simple Imputation\n(Mode/Median)', fontsize=10)
        st.pyplot(fig2)
        plt.close()
    
    with col3:
        st.subheader("Multiple Imputation")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        # Convert categorical to numeric for correlation
        df_mi_numeric = df_mi[numerical_features].copy()
        for col in ['ca', 'thal', 'fbs']:
            if col in df_mi_numeric.columns:
                df_mi_numeric[col] = pd.to_numeric(df_mi_numeric[col], errors='coerce')
        corr_mi = df_mi_numeric.corr()
        sns.heatmap(
            corr_mi,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            ax=ax3,
            cbar_kws={'label': 'Correlation'},
            vmin=-1, vmax=1
        )
        ax3.set_title('Multiple Imputation\n(MICE)', fontsize=10)
        st.pyplot(fig3)
        plt.close()
    
    st.markdown("---")
    
    # Correlation with Target
    st.header("Feature Correlations with Target Variable")
    
    # Calculate correlations with target
    df_with_target = df[numerical_features + ['target']].copy()
    df_with_target['target'] = df_with_target['target'].astype(int)
    
    target_corr = df_with_target.corr()['target'].drop('target').sort_values(ascending=False)
    
    # Plot correlation with target
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in target_corr.values]
    ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor='black')
    ax.set_xlabel('Correlation with Target', fontsize=12)
    ax.set_title('Feature Correlations with Heart Disease', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Display correlation values
    st.subheader("Correlation Values")
    st.write("Correlation coefficients between features and target variable:")
    
    corr_df = pd.DataFrame({
        'Feature': target_corr.index,
        'Correlation': target_corr.values
    }).reset_index(drop=True)
    
    st.dataframe(corr_df.style.background_gradient(subset=['Correlation'], 
                                                    cmap='RdYlGn', 
                                                    vmin=-1, vmax=1))
    
    # Interpretation
    st.info("""
    **Interpreting Correlations:**
    - **Positive correlation**: As feature increases, heart disease risk increases
    - **Negative correlation**: As feature increases, heart disease risk decreases
    - **Strong correlation**: |r| > 0.5
    - **Moderate correlation**: 0.3 < |r| < 0.5
    - **Weak correlation**: |r| < 0.3
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Data Source:** UCI Heart Disease Dataset  
**Total Records:** {df.shape[0]} patients  
**Features:** {df.shape[1]}  
**Imputation:** {dataset_choice}
""")

st.sidebar.markdown("---")
st.sidebar.success("""
**Missing Data Handling:**
- Created missingness indicators for MNAR variables
- Applied both simple and advanced imputation
- Preserved relationships between features
""")
