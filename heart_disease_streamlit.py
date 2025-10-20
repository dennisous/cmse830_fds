# heart_disease_streamlit.py
# Heart Disease Analysis streamlit app
#Several of these sections were generated with assistance of Claude Sonnet 4.5 10/15/2025

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Factor Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

# Load data
@st.cache_data
def load_data():
    try:
        df_original = pd.read_csv('data/heart_disease_original.csv')
    except FileNotFoundError:
        st.error("Please ensure 'heart_disease_original.csv' exists in the same directory!")
        df_original = None
    
    try:
        df_simple = pd.read_csv('data/heart_disease_simple_imputation.csv')
    except FileNotFoundError:
        st.error("Please ensure 'heart_disease_simple_imputation.csv' exists!")
        df_simple = None
        
    try:
        df_knn = pd.read_csv('data/heart_disease_knn_imputation.csv')
    except FileNotFoundError:
        st.error("Please ensure 'heart_disease_knn_imputation.csv' exists!")
        df_knn = None
    
    return df_original, df_simple, df_knn

df_original, df_simple, df_knn = load_data()

# Define column groups
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]

# SIDEBAR NAVIGATION

st.sidebar.title("🫀 Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["Home", "Data Explorer", "EDA - Univariate Analysis", "Pair-Plot Analysis", 
     "Missingness Analysis", "Correlation Analysis"]
)

# HOME PAGE

if page == "Home":
    st.title("🫀 Heart Disease Risk Factor Analysis")
    st.markdown("---")
    
    # Audience and Goal section
    st.header("🎯 Project Purpose")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Target Audience")
        st.info("""
        **Primary Audience:**
        - People ages 25-70 concerned about heart disease risk
        - Healthcare providers seeking to understand key risk factors
        - Medical researchers analyzing cardiovascular health patterns
        """)
    
    with col2:
        st.subheader("🎯 Project Goal")
        st.success("""
        **Main Objective:**
        To inform the audience of risk factors that contribute to heart disease 
        so that they can make informed decisions to reduce their risk and improve 
        cardiovascular health outcomes.
        """)
    
    st.markdown("---")
    
    # Dataset Overview
    st.header("📋 Dataset Overview")
    st.write("""
    This application analyzes the **UCI Heart Disease Dataset**, combining data from 
    Cleveland and Switzerland patient databases. The analysis employs two imputation 
    techniques to handle missing data while preserving data integrity.
    """)
    
    if df_original is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", df_original.shape[0])
        with col2:
            st.metric("Total Features", df_original.shape[1])
        with col3:
            disease_pct = (df_original['target'].astype(int).sum() / len(df_original)) * 100
            st.metric("Disease Prevalence", f"{disease_pct:.1f}%")
    
    st.markdown("---")
    
    # Data Source
    st.header("🔗 Data Source")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Cleveland and Switzerland Database:**")
        st.write("https://archive.ics.uci.edu/dataset/45/heart+disease")
    
    
    st.markdown("---")
    
    # Imputation Methods
    st.header("🔧 Imputation Methods")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simple Imputation")
        st.write("""
        **Approach:**
        - **Categorical variables**: Mode (most frequent value)
        - **Numeric variables**: Median
        """)
    
    with col2:
        st.subheader("KNN Imputation")
        st.write("""
        **Approach:**
        - Uses K-Nearest Neighbors algorithm (K=5)
        - Imputes based on similar patients' values
        """)
    
    st.markdown("---")
    
    # Feature Descriptions
    st.header("📊 Feature Descriptions")
    st.write("### There are 14 variables total in the dataset:")
    
    feature_descriptions = {
        "age": "Quantitative integer variable measured in years",
        "sex": "Categorical variable (0 = Female, 1 = Male)",
        "cp": "Categorical variable with 4 levels, signifying chest pain type:\n" +
               "  - Value 1: typical angina\n" +
               "  - Value 2: atypical angina\n" +
               "  - Value 3: non-anginal pain\n" +
               "  - Value 4: asymptomatic",
        "trestbps": "Quantitative integer variable denoting resting blood pressure when " +
                    "admitted to hospital (mm Hg)",
        "chol": "Quantitative integer variable denoting serum cholesterol (mg/dl)",
        "fbs": "Binary categorical variable (1 if fasting blood sugar > 120 mg/dl, 0 otherwise)",
        "restecg": "Categorical variable depicting resting electrocardiographic results:\n" +
                   "  - Value 0: normal\n" +
                   "  - Value 1: having ST-T wave abnormality (T wave inversions and/or ST " +
                   "elevation or depression > 0.05 mV)\n" +
                   "  - Value 2: showing probable or definite left ventricular hypertrophy " +
                   "by Estes' criteria",
        "thalach": "Integer quantitative variable depicting maximum heart rate achieved",
        "exang": "Binary categorical variable denoting exercise induced angina (1 = yes, 0 = no)",
        "oldpeak": "Quantitative float variable denoting ST depression induced by exercise " +
                   "relative to rest (mm). Measures how much your heart's electrical signal " +
                   "'drops' during exercise compared to rest. Higher values are more concerning " +
                   "as it suggests the heart is having trouble getting oxygen when exercising.",
        "slope": "Integer categorical variable depicting slope of peak exercise ST segment:\n" +
                 "  - Value 1: upsloping\n" +
                 "  - Value 2: flat\n" +
                 "  - Value 3: downsloping",
        "ca": "Integer categorical variable representing number of major vessels covered by " +
              "fluoroscopy (0-3)",
        "thal": "Categorical integer variable:\n" +
                "  - 3 = normal\n" +
                "  - 6 = fixed defect\n" +
                "  - 7 = reversible defect\n" +
                "  (Values 6 and 7 show blockage of blood flow during exertion)",
        "target": "Integer categorical variable of interest (1 = diagnosis of heart disease, " +
                  "0 = no diagnosis)"
    }
    
    for feature, description in feature_descriptions.items():
        with st.expander(f"**{feature}**"):
            st.write(description)

# DATA EXPLORER PAGE

elif page == "Data Explorer":
    st.title("🔍 Interactive Data Explorer")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset not found!")
        st.stop()
    
    st.info("Viewing: **Original Dataset** (before imputation)")
    
    # Age Distribution with Slider
    st.header("📊 Age Distribution Analysis")
    st.write("Use the slider to filter patients by age range and explore disease prevalence:")
    
    age_range = st.slider(
        "Select age range:",
        min_value=int(df_original['age'].min()),
        max_value=int(df_original['age'].max()),
        value=(int(df_original['age'].min()), int(df_original['age'].max())),
        help="Drag the slider to filter patients by age"
    )
    
    # Filter data based on age range
    filtered_df = df_original[(df_original['age'] >= age_range[0]) & (df_original['age'] <= age_range[1])]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patients in Range", len(filtered_df))
    with col2:
        disease_count = filtered_df['target'].astype(int).sum()
        st.metric("Patients with Disease", disease_count)
    with col3:
        disease_pct_filtered = (disease_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric("Disease Prevalence", f"{disease_pct_filtered:.1f}%")
    
    # Disease distribution in filtered range
    st.subheader("Disease Distribution in Selected Age Range")
    
    if len(filtered_df) > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        target_counts = filtered_df['target'].astype(int).value_counts().sort_index()
        
        bars = ax2.bar(target_counts.index, target_counts.values,
                       color=['#3498db', '#e74c3c'],
                       edgecolor='black', width=0.6)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Heart Disease Status', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'Disease Distribution (Age {age_range[0]}-{age_range[1]})', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['No Disease', 'Disease'])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        st.pyplot(fig2)
        plt.close()
    else:
        st.warning("No patients in selected age range!")
    
    st.markdown("---")
    
    # Feature selector
    st.header("📈 Individual Feature Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_feature = st.selectbox(
            "Choose a feature to explore:",
            numeric_cols,
            help="Select a continuous feature to visualize"
        )
    
    with col2:
        st.write("")  # Spacing
        show_by_target = st.checkbox("Show distribution by disease status", value=True)
    
    # Plot selected feature
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    if show_by_target:
        for target_val, color, label in [(0, '#3498db', 'No Disease'), 
                                          (1, '#e74c3c', 'Disease')]:
            data_subset = filtered_df[filtered_df['target'].astype(int) == target_val][selected_feature]
            sns.kdeplot(data=data_subset, fill=True, alpha=0.5, linewidth=2,
                       label=label, ax=ax3, color=color)
    else:
        sns.histplot(data=filtered_df, x=selected_feature, kde=True, bins=20,
                    color='steelblue', edgecolor='black', alpha=0.7, ax=ax3,
                    line_kws={"linewidth": 2.5})
    
    ax3.set_xlabel(selected_feature.capitalize(), fontsize=12)
    ax3.set_ylabel('Density' if show_by_target else 'Frequency', fontsize=12)
    ax3.set_title(f'Distribution of {selected_feature.capitalize()}', 
                 fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    if show_by_target:
        ax3.legend(fontsize=10)
    
    st.pyplot(fig3)
    plt.close()
    
    # Show statistics
    st.subheader(f"Statistics for {selected_feature}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{filtered_df[selected_feature].mean():.2f}")
    with col2:
        st.metric("Median", f"{filtered_df[selected_feature].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{filtered_df[selected_feature].std():.2f}")
    with col4:
        st.metric("Range", f"{filtered_df[selected_feature].max() - filtered_df[selected_feature].min():.2f}")

# EDA - UNIVARIATE ANALYSIS PAGE

elif page == "EDA - Univariate Analysis":
    st.title("📊 Exploratory Data Analysis - Univariate")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset not found!")
        st.stop()
    
    st.info("Viewing: **Original Dataset** (before imputation)")
    
    # Numeric Features
    st.header("Distribution of Quantitative Variables")
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, var in enumerate(numeric_cols):
        sns.histplot(x=var, data=df_original, 
                    kde=True, 
                    line_kws={"linewidth": 2.5}, 
                    color="steelblue",
                    edgecolor="black",
                    alpha=0.7,
                    ax=axs[i])
        axs[i].set_title(f"Distribution of {var.capitalize()}", 
                        fontsize=12, fontweight="bold")
        axs[i].set_xlabel(f"{var.capitalize()}", fontsize=11)
        axs[i].set_ylabel("Frequency", fontsize=11)
        axs[i].grid(axis="y", alpha=0.3, linestyle="--")
    
    fig.delaxes(axs[5])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Key insights
    st.subheader("📝 Key Insights - Numeric Variables")
    st.write("""
    - **Cholesterol** is approximately normally distributed in this dataset.
    - **Oldpeak** has a large peak around value 0, suggesting that a majority of patients 
      in the dataset do not have trouble with blood flow to the heart after exercising.
    """)
    
    st.markdown("---")
    
    # Categorical Features
    st.header("Frequency of Categorical Variables by Disease Status")
    st.write("Stacked histograms showing the distribution of categorical features, stratified by heart disease presence.")
    
    categorical_cols_minusTarget = [col for col in categorical_cols if col != "target"]
    
    fig2, axs2 = plt.subplots(3, 3, figsize=(16, 12))
    axs2 = axs2.flatten()
    
    for i, var in enumerate(categorical_cols_minusTarget):
        sns.histplot(x=var, 
                    data=df_original, 
                    hue="target",
                    multiple="stack",
                    discrete=True,
                    edgecolor="black", 
                    palette={0: '#3498db', 1: '#e74c3c'},
                    ax=axs2[i])
        
        # Add count labels
        for container in axs2[i].containers:
            axs2[i].bar_label(container, fmt="%d", label_type="edge", fontsize=7)
        
        # Set x-ticks to only show existing values
        unique_vals = sorted(df_original[var].dropna().unique())
        axs2[i].set_xticks(unique_vals)
        axs2[i].set_xticklabels(unique_vals)
        
        axs2[i].set_title(f"Distribution of {var.capitalize()} by Target", 
                         fontsize=12, fontweight="bold")
        axs2[i].set_xlabel(f"{var.capitalize()}", fontsize=11)
        axs2[i].set_ylabel(f"Count", fontsize=11)
        axs2[i].legend(labels=["No Disease", "Disease"], 
                      title="Target", 
                      fontsize=8,
                      title_fontsize=9)
    
    fig2.delaxes(axs2[8])
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    
    # Key insights for categorical variables
    st.subheader("📝 Key Insights - Categorical Variables")
    
    st.write("""
    **Major Risk Factors Identified:**
    
    - **Sex**: Over 2/3 of males in the dataset have heart disease, whereas less than half 
      of females have disease, suggesting sex is an important risk factor.
    
    - **Chest Pain (cp = 4)**: Patients with asymptomatic chest pain are overwhelmingly 
      in the disease class. This suggests that asymptomatic chest pain appears to be a 
      strong indicator of heart disease risk.
    
    - **Resting ECG (restecg = 1)**: Patients with ST-T wave abnormality (T wave inversions 
      and/or ST elevation or depression > 0.05 mV) are predominantly in the disease class, 
      indicating that ST-T wave abnormalities are a strong predictor of heart disease.
    
    - **Exercise Induced Angina (exang = 1)**: The proportion of patients with heart disease 
      is substantially higher among those who experience exercise-induced angina compared to 
      those who do not.
    
    - **ST Segment Slope**: Patients whose ST segment slope is flat or downsloping during 
      peak exercise show a much higher disease rate compared to those with an upsloping pattern.
    
    - **Number of Vessels (ca > 0)**: The proportion of disease cases increases sharply when 
      ca (number of major vessels colored by fluoroscopy) is greater than 0, suggesting that 
      any vessel blockage visible on fluoroscopy indicates elevated heart disease risk.
    
    - **Thalassemia (thal = 6 or 7)**: Patients with fixed defect or reversible defect 
      predominantly have heart disease, unlike those with normal results, indicating that 
      abnormal thallium stress test results are strongly associated with disease presence.
    """)

# PAIR-PLOT ANALYSIS PAGE

elif page == "Pair-Plot Analysis":
    st.title("🔗 Pair-Plot Analysis")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset not found!")
        st.stop()
    
    st.info("Viewing: **Original Dataset** (before imputation)")
    
    st.write("""
    Pairwise relationships between continuous features, showing both scatter plots 
    and distribution curves (KDE), colored by disease status.
    """)
    
    with st.spinner("Generating pairplot... This may take a moment."):
        # Create pairplot
        g = sns.pairplot(
            data=df_original[numeric_cols + ["target"]],
            hue="target",
            diag_kind="kde",
            palette={0: "#3498db", 1: "#e74c3c"},  
            plot_kws={
                "alpha": 0.7,
                "s": 30,
                "edgecolor": "white",
                "linewidth": 0.5
            },
            diag_kws={
                "fill": True,
                "alpha": 0.7,
                "linewidth": 2.5
            },
            corner=True,  
            height=2.5, 
            aspect=1.2   
        )
        
        g._legend.set_title("Heart Disease", prop={'size': 11, 'weight': 'bold'})
        for text, label in zip(g._legend.texts, ["No Disease", "Disease"]):
            text.set_text(label)
            text.set_fontsize(10)
        
        g.fig.suptitle("Pairwise Relationships of Numerical Features by Disease Status", 
                       fontsize=16, fontweight='bold', y=1.02)
        
        st.pyplot(g.fig)
        plt.close()
    
    st.markdown("---")
    
    # Key insights
    st.header("📝 Key Insights from Pair-Plots")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.subheader("🔍 Correlation Patterns")
        st.write("""
        - **Age and Thalach (Maximum heart rate)** show a moderate negative correlation. 
          Older patients tend to have a lower maximum heart rate.
        
        - **Oldpeak and Disease**: Patients with heart disease show higher oldpeak values 
          (ST depression) compared to those without heart disease, suggesting that greater 
          oldpeak values increase risk of heart disease.
        """)
    
    with insights_col2:
        st.subheader("📊 Distribution Differences")
        st.write("""
        - **Maximum Heart Rate (thalach)**: The kernel density estimate for the no disease 
          group reveals that patients without the disease achieve a greater maximum heart 
          rate on average, compared to those with disease.
        
        - **Age Distribution**: The median age is higher for patients with disease compared 
          to those without. The no disease group shows greater age variability, with a wider 
          distribution and lower peak density.
        """)
    
    st.markdown("---")
    
    st.info("""
    **Interpretation Note:** While individual features show overlap between groups, 
    combinations of features (visible in the scatter plots) may provide better discrimination 
    for classification models. The pair-plots reveal that multivariate patterns are more 
    informative than univariate distributions alone.
    """)

# MISSINGNESS ANALYSIS PAGE

elif page == "Missingness Analysis":
    st.title("🔍 Missing Data Analysis")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset with missing values not found! Please ensure 'heart_disease_original.csv' exists.")
        st.stop()
    
    st.write("""
    Analysis of missing data patterns in the original dataset before imputation. 
    Understanding missingness mechanisms is crucial for appropriate handling strategies.
    """)
    
    # Missing value counts
    st.header("📊 Missing Value Summary")
    
    missing_counts = df_original.isna().sum()
    missing_pct = (missing_counts / len(df_original) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Feature': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Missing Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(missing_df.style.background_gradient(subset=['Missing Percentage'], 
                                                              cmap='Reds'))
        
        with col2:
            # Bar plot of missing percentages
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(missing_df['Feature'], missing_df['Missing Percentage'], 
                   color='coral', edgecolor='black')
            ax.set_xlabel('Missing Percentage (%)', fontsize=12)
            ax.set_title('Missing Data by Feature', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
            plt.close()
    else:
        st.success("No missing values found in the dataset!")
    
    st.markdown("---")
    
    # Missing data heatmap
    st.header("🗺️ Missing Data Pattern Visualization")
    st.write("Yellow indicates missing values, purple indicates observed values.")
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    missing_mask = df_original.isna()
    sns.heatmap(missing_mask, cbar=False, cmap="viridis", 
               yticklabels=False, ax=ax2)
    ax2.set_title("Missing Value Patterns in the Dataset", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Features", fontsize=13)
    ax2.set_ylabel("Row Index", fontsize=13)
    st.pyplot(fig2)
    plt.close()
    
    st.markdown("---")
    
    # Analysis and interpretation
    st.header("📝 Missingness Mechanism Analysis")
    
    st.subheader("🔬 Missing Not At Random (MNAR) Assessment")
    
    st.write("""
    The majority of missing data comes from the Switzerland dataset and is very likely 
    **Missing Not At Random (MNAR)**. This has important implications for imputation strategies:
    """)
    
    st.write("**Key Variables with MNAR patterns:**")
    
    with st.expander("**ca (Number of vessels via fluoroscopy)**"):
        st.write("""
        - **Why it might be missing**: Cardiac catheterization requires X-ray procedures to test 
          arterial health and blood flow. The procedure is costly and cannot be performed on every patient.
        - **MNAR possible reasoning**: The decision to perform cardiac catheterization is 
          related to the patient's suspected heart disease severity
        - **Implication**: Missing values are actually informative, they may indicate a lower risk.
        """)
    
    with st.expander("**slope, thal (Exercise test results)**"):
        st.write("""
        - These variables require specialized stress testing
        - Missing values likely indicate patients who didn't undergo complete cardiac workup
        - Missingness may be related to disease severity or hospital protocols
        """)
    
    with st.expander("**fbs (Fasting blood sugar)**"):
        st.write("""
        - Requires patient to be fasted before measurement
        - Cleveland patients have fbs values collected, but Switzerland data doesn't
        - **MNAR possible reasoning**: Differences in hospital procedures
        """)
    
    with st.expander("**chol (Cholesterol)**"):
        st.write("""
        - All missing cholesterol data comes from the Switzerland dataset
        - Originally labeled as value 0 before conversion to NaN
        - Represents data collection differences between Cleveland and Switzerland database
        """)
    
    st.markdown("---")
    
    st.subheader("💡 Imputation Strategy")
    
    st.write("""
    Given the MNAR nature of the missing data, this analysis employs two strategies:
    
    1. **Missing Indicators**: Created binary indicator variables (`ca_missing`, `fbs_missing`, 
       `slope_missing`, `thal_missing`, `chol_missing`) to capture the information in the 
       missingness itself
    
    2. **Advanced Imputation**: Applied both simple (median/mode) and KNN imputation methods 
       to fill missing values
    
    3. **Comparative Analysis**: Correlation patterns are compared across imputation methods 
       to assess the impact of different strategies
    """)
    
    st.info("""
    **Important**: Because data is likely MNAR, any imputation method may introduce bias. 
    The missing indicators I created help capture whether a test was performed, which can add predictive power
    for the classification models I will use during the final project.
    """)

# CORRELATION ANALYSIS PAGE

elif page == "Correlation Analysis":
    st.title("🔗 Correlation Analysis")
    st.markdown("---")
    
    if df_original is None or df_simple is None or df_knn is None:
        st.error("One or more required datasets not found! Please ensure all CSV files exist.")
        st.stop()
    
    st.write("""
    Compare correlation patterns across different imputation methods to understand how 
    missing data handling affects feature relationships.
    """)
    
    # Toggle for variable type
    st.header("📊 Select Variable Type")
    var_type = st.radio(
        "Choose which variables to analyze:",
        ["Numeric Variables", "Categorical Variables"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if var_type == "Numeric Variables":
        st.subheader("Correlation Heatmaps - Numeric Features")

        # Create three correlation matrices
        # Convert to numeric for correlation
        cols_to_use = numeric_cols + ["target"]
        
        # Original
        df_orig_num = df_original[cols_to_use].copy()
        for col in df_orig_num.columns:
            df_orig_num[col] = pd.to_numeric(df_orig_num[col], errors='coerce')
        corr_original = df_orig_num.corr()
        
        # Simple
        df_simple_num = df_simple[cols_to_use].copy()
        for col in df_simple_num.columns:
            df_simple_num[col] = pd.to_numeric(df_simple_num[col], errors='coerce')
        corr_simple = df_simple_num.corr()
        
        # KNN
        df_knn_num = df_knn[cols_to_use].copy()
        for col in df_knn_num.columns:
            df_knn_num[col] = pd.to_numeric(df_knn_num[col], errors='coerce')
        corr_knn = df_knn_num.corr()
        
        # Display side by side
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Data**")
            fig1, ax1 = plt.subplots(figsize=(7, 6))
            sns.heatmap(
                corr_original,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                ax=ax1,
                cbar_kws={'label': 'Correlation (r)'},
                vmin=-1, vmax=1,
                square=True
            )
            ax1.set_title('Original\n(Missing Excluded)', fontsize=11, fontweight='bold')
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            st.markdown("**Simple Imputation**")
            fig2, ax2 = plt.subplots(figsize=(7, 6))
            sns.heatmap(
                corr_simple,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                ax=ax2,
                cbar_kws={'label': 'Correlation (r)'},
                vmin=-1, vmax=1,
                square=True
            )
            ax2.set_title('Simple Imputation\n(Mode/Median)', fontsize=11, fontweight='bold')
            st.pyplot(fig2)
            plt.close()
        
        with col3:
            st.markdown("**KNN Imputation**")
            fig3, ax3 = plt.subplots(figsize=(7, 6))
            sns.heatmap(
                corr_knn,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                ax=ax3,
                cbar_kws={'label': 'Correlation (r)'},
                vmin=-1, vmax=1,
                square=True
            )
            ax3.set_title('KNN Imputation\n(K=5 Neighbors)', fontsize=11, fontweight='bold')
            st.pyplot(fig3)
            plt.close()
        
    else:  # Categorical Variables
        st.subheader("Correlation Heatmaps - Categorical Features")
        st.write("**Note:** Categorical variables in correlation calculations are ordinal "
                "(have meaningful numeric order), making Pearson correlation appropriate.")
        
        
        # Create three correlation matrices for categorical
        # Original
        df_orig_cat = df_original[categorical_cols].copy()
        for col in df_orig_cat.columns:
            df_orig_cat[col] = pd.to_numeric(df_orig_cat[col], errors='coerce')
        corr_original_cat = df_orig_cat.corr()
        
        # Simple
        df_simple_cat = df_simple[categorical_cols].copy()
        for col in df_simple_cat.columns:
            df_simple_cat[col] = pd.to_numeric(df_simple_cat[col], errors='coerce')
        corr_simple_cat = df_simple_cat.corr()
        
        # KNN
        df_knn_cat = df_knn[categorical_cols].copy()
        for col in df_knn_cat.columns:
            df_knn_cat[col] = pd.to_numeric(df_knn_cat[col], errors='coerce')
        corr_knn_cat = df_knn_cat.corr()
        
        # Display side by side
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Data**")
            fig1, ax1 = plt.subplots(figsize=(8, 7))
            sns.heatmap(
                corr_original_cat,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                ax=ax1,
                cbar_kws={'label': 'Correlation (r)'},
                vmin=-1, vmax=1,
                square=True
            )
            ax1.set_title('Original\n(Missing Excluded)', fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            st.markdown("**Simple Imputation**")
            fig2, ax2 = plt.subplots(figsize=(8, 7))
            sns.heatmap(
                corr_simple_cat,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                ax=ax2,
                cbar_kws={'label': 'Correlation (r)'},
                vmin=-1, vmax=1,
                square=True
            )
            ax2.set_title('Simple Imputation\n(Mode)', fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig2)
            plt.close()
        
        with col3:
            st.markdown("**KNN Imputation**")
            fig3, ax3 = plt.subplots(figsize=(8, 7))
            sns.heatmap(
                corr_knn_cat,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                ax=ax3,
                cbar_kws={'label': 'Correlation (r)'},
                vmin=-1, vmax=1,
                square=True
            )
            ax3.set_title('KNN Imputation\n(K=5 Neighbors)', fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig3)
            plt.close()
    
    st.markdown("---")
    
    # Analysis and interpretation
    st.header("📝 Correlation Analysis Insights")
    
    if var_type == "Categorical Variables":
        st.subheader("🔍 Simple Imputation Effects")
        st.write("""
        Some correlations are changing directions from positive to negative (and vice versa) 
        after simple imputation, such as:
        - COR(exang, thal)
        - COR(sex, ca)
        
        Some correlation coefficient magnitudes are changing drastically, such as:
        - Variable pair (ca, target): from r = 0.46 to r = 0.22 (a sizeable jump)
        
        **Interpretation**: The spikes in correlation coefficient magnitude along with direction 
        changes support reasoning that data is MNAR. Simple imputation (mode) does not account 
        for the relationships between variables, leading to biased estimates.
        """)
        
        st.markdown("---")
        
        st.subheader("🔍 KNN Imputation Effects")
        st.write("""
        Similar to simple imputation, some correlations change direction completely after KNN 
        imputation. Notable examples:
        
        - **(thal, target)**: Original r = 0.51 changes dramatically in both direction and 
          magnitude to r = -0.04
        - **(slope, target)**: Changes direction completely from r = 0.31 to r = -0.31
        
        **Interpretation**: The many severe cases of direction reversal indicate that KNN is 
        likely imputing poorly - learning patterns from incorrect neighbors. This strengthens 
        the argument that the data is MNAR, and that careful analysis and model fitting techniques 
        will need to be examined during later project work.
        """)
    
    st.markdown("---")
    
    # Interpretation guide
    st.info("""
    **Interpreting Correlations:**
    - **Positive correlation (red)**: As feature increases, heart disease risk increases
    - **Negative correlation (blue)**: As feature increases, heart disease risk decreases
    - **Strong correlation**: |r| > 0.5
    - **Moderate correlation**: 0.3 < |r| < 0.5
    - **Weak correlation**: |r| < 0.3
    """)

# FOOTER / SIDEBAR INFO

st.sidebar.markdown("---")
if df_original is not None:
    st.sidebar.info(f"""
    **📊 Dataset Info:**  
    **Patients:** {df_original.shape[0]}  
    **Features:** {df_original.shape[1]}  
    **Disease Prevalence:** {(df_original['target'].astype(int).sum() / len(df_original) * 100):.1f}%
    """)

st.sidebar.markdown("---")
st.sidebar.success("""
**🔧 Data Processing:**
- ✅ MNAR indicators created
- ✅ Two imputation methods applied
- ✅ Comparative analysis enabled
""")

st.sidebar.markdown("---")
st.sidebar.write("**📚 Data Sources:**")
st.sidebar.write("[UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)")
st.sidebar.write("[Switzerland Data](https://archive.ics.uci.edu/dataset/45/heart+disease)")