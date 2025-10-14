# streamlit_app.py
# Heart Disease Analysis - Minimum Requirements Streamlit App

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('heart.csv')
    df = df.drop_duplicates()
    return df

df = load_data()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer"])

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "Home":
    st.title("🫀 Heart Disease Analysis")
    st.markdown("---")
    
    st.header("Project Overview")
    st.write("""
    This application analyzes the UCI Heart Disease Dataset to understand factors 
    contributing to heart disease risk. The dataset combines data from Cleveland, 
    Hungary, Switzerland, and VA Long Beach databases.
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
    
    st.subheader("Features Description")
    st.write("""
    - **age**: Patient age in years
    - **sex**: 1 = male, 0 = female
    - **cp**: Chest pain type (0-3)
    - **trestbps**: Resting blood pressure (mm Hg)
    - **chol**: Serum cholesterol (mg/dl)
    - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    - **restecg**: Resting ECG results (0-2)
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina (1 = yes, 0 = no)
    - **oldpeak**: ST depression induced by exercise
    - **slope**: Slope of peak exercise ST segment (0-2)
    - **ca**: Number of major vessels colored by fluoroscopy (0-3)
    - **thal**: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)
    - **target**: Heart disease presence (1 = disease, 0 = no disease)
    """)

# ============================================================================
# DATA EXPLORER PAGE (WITH 2 INTERACTIVE ELEMENTS)
# ============================================================================

elif page == "Data Explorer":
    st.title("📊 Interactive Data Explorer")
    st.markdown("---")
    
    # INTERACTIVE ELEMENT 1: Feature Selector (Dropdown)
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
    
    # INTERACTIVE ELEMENT 2: Slider for Age Filtering
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
    
    # Display filtered data table
    if st.checkbox("Show filtered data table"):
        st.dataframe(filtered_df)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.info("""
**Data Source:** UCI Heart Disease Dataset  
**Total Records:** 1025 patients  
**Purpose:** Educational analysis for midterm project
""")
