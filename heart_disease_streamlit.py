# heart_disease_streamlit.py
# Heart Disease Analysis Streamlit App
# Several sections generated with assistance of Claude Sonnet 4.5 10/15/2025
# Updated with third dataset, Random Forest, and model comparison

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Factor Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================

@st.cache_data
def load_data():
    """Load all three datasets with proper error handling."""
    try:
        df_original = pd.read_csv('data/heart_disease_original.csv')
    except FileNotFoundError:
        st.error("Please ensure 'heart_disease_original.csv' exists in the data directory!")
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

# Define column groups (updated to match new naming convention)
numeric_cols = ["age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak"]
categorical_cols = ["gender", "chestpain", "fastingbloodsugar", "restingrelectro", 
                    "exerciseangia", "slope", "noofmajorvessels", "target"]

# ============================================================================
# SESSION STATE FOR MODEL RESULTS (Advanced Streamlit feature)
# ============================================================================

if 'logistic_model_fitted' not in st.session_state:
    st.session_state.logistic_model_fitted = False
if 'rf_model_fitted' not in st.session_state:
    st.session_state.rf_model_fitted = False

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🫀 Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["Home", "Data Explorer", "EDA - Univariate Analysis", "Pair-Plot Analysis", 
     "Missingness Analysis", "Correlation Analysis", "Logistic Regression Analysis",
     "Random Forest Analysis", "Model Comparison & Recommendations"]
)

# ============================================================================
# HOME PAGE
# ============================================================================

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
    This application analyzes a **combined heart disease dataset** integrating data from 
    **three distinct sources**:
    
    1. **Cleveland Clinic Database** (UCI Repository)
    2. **Switzerland Hospital Database** (UCI Repository)  
    3. **Multispeciality Indian Hospital Database** (Mendeley Data)
    
    The analysis employs two imputation techniques to handle missing data while 
    preserving data integrity and accounts for the MNAR (Missing Not At Random) 
    nature of the missing values.
    """)
    
    if df_original is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", df_original.shape[0])
        with col2:
            st.metric("Total Features", df_original.shape[1])
        with col3:
            target_col = 'target' if 'target' in df_original.columns else df_original.columns[-1]
            disease_pct = (df_original[target_col].astype(int).sum() / len(df_original)) * 100
            st.metric("Disease Prevalence", f"{disease_pct:.1f}%")
        with col4:
            st.metric("Data Sources", "3")
    
    st.markdown("---")
    
    # Data Sources
    st.header("🔗 Data Sources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏥 Cleveland & Switzerland")
        st.write("**UCI Machine Learning Repository**")
        st.write("[Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)")
        st.caption("Classic heart disease dataset with clinical measurements")
    
    with col2:
        st.subheader("🏥 Indian Hospital")
        st.write("**Mendeley Data Repository**")
        st.write("[Cardiovascular Disease Dataset](https://data.mendeley.com/datasets/dzz48mvjht/1)")
        st.caption("Multispeciality hospital cardiovascular data")
    
    with col3:
        st.subheader("📚 Citation")
        st.write("""
        Doppala, Bhanu Prakash; Bhattacharyya, Debnath (2021), 
        "Cardiovascular_Disease_Dataset", Mendeley Data, V1, 
        doi: 10.17632/dzz48mvjht.1
        """)
    
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
        
        **Limitation**: Does not account for relationships between variables
        """)
    
    with col2:
        st.subheader("KNN Imputation")
        st.write("""
        **Approach:**
        - Uses K-Nearest Neighbors algorithm (K=5)
        - Imputes based on similar patients' values
        
        **Limitation**: May learn incorrect patterns for MNAR data
        """)
    
    st.markdown("---")
    
    # Feature Descriptions
    st.header("📊 Feature Descriptions")
    st.write("### There are 13 variables total in the combined dataset:")
    
    feature_descriptions = {
        "age": "Quantitative integer variable measured in years",
        "gender": "Categorical variable (0 = Female, 1 = Male)",
        "chestpain": "Categorical variable with 4 levels, signifying chest pain type:\n" +
               "  - Value 1: typical angina\n" +
               "  - Value 2: atypical angina\n" +
               "  - Value 3: non-anginal pain\n" +
               "  - Value 4: asymptomatic",
        "restingBP": "Quantitative integer variable denoting resting blood pressure when " +
                    "admitted to hospital (mm Hg)",
        "serumcholestrol": "Quantitative integer variable denoting serum cholesterol (mg/dl)",
        "fastingbloodsugar": "Binary categorical variable (1 if fasting blood sugar > 120 mg/dl, 0 otherwise)",
        "restingrelectro": "Categorical variable depicting resting electrocardiographic results:\n" +
                   "  - Value 0: normal\n" +
                   "  - Value 1: having ST-T wave abnormality (T wave inversions and/or ST " +
                   "elevation or depression > 0.05 mV)\n" +
                   "  - Value 2: showing probable or definite left ventricular hypertrophy " +
                   "by Estes' criteria",
        "maxheartrate": "Integer quantitative variable depicting maximum heart rate achieved",
        "exerciseangia": "Binary categorical variable denoting exercise induced angina (1 = yes, 0 = no)",
        "oldpeak": "Quantitative float variable denoting ST depression induced by exercise " +
                   "relative to rest (mm). Measures how much your heart's electrical signal " +
                   "'drops' during exercise compared to rest. Higher values are more concerning " +
                   "as it suggests the heart is having trouble getting oxygen when exercising.",
        "slope": "Integer categorical variable depicting slope of peak exercise ST segment:\n" +
                 "  - Value 1: upsloping\n" +
                 "  - Value 2: flat\n" +
                 "  - Value 3: downsloping",
        "noofmajorvessels": "Integer categorical variable representing number of major vessels covered by " +
              "fluoroscopy (0-3)",
        "target": "Integer categorical variable of interest (1 = diagnosis of heart disease, " +
                  "0 = no diagnosis)"
    }
    
    for feature, description in feature_descriptions.items():
        with st.expander(f"**{feature}**"):
            st.write(description)

# ============================================================================
# DATA EXPLORER PAGE
# ============================================================================

elif page == "Data Explorer":
    st.title("🔍 Interactive Data Explorer")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset not found!")
        st.stop()
    
    st.info("Viewing: **Original Dataset** (combined from Cleveland, Switzerland, and Indian Hospital data)")
    
    # Target Variable Encoding Explanation
    st.header("🎯 Target Variable Encoding")
    st.write("""
    The target variable represents heart disease diagnosis. The original dataset contained a 
    multi-level variable (`num`/`target`) with values 0-4, which was encoded into a binary classification:
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Original Variable: `target`")
        st.markdown("""
        **Description of target variable:**
        
        - **Value 0**: No vessels with >50% diameter narrowing (absence of significant disease)
        - **Value 1**: 1 major vessel with >50% diameter narrowing
        - **Value 2**: 2 major vessels with >50% diameter narrowing
        - **Value 3**: 3 major vessels with >50% diameter narrowing
        - **Value 4**: 4 major vessels with >50% diameter narrowing
        """)
    
    with col2:
        st.markdown("### Encoding Logic")
        st.code("""
# Encoding values 1-4 as "1" (heart disease present)
# All these values signify heart risk as at least 
# 1 major vessel has >50% diameter narrowing

df["target"] = np.where(
    df["target"].isin([1, 2, 3, 4]), 
    1,  # Heart disease present
    0   # No heart disease
)
        """, language="python")
        
        st.info("""
        **Binary Target Variable:**
        - **0**: No heart disease (target = 0)
        - **1**: Heart disease present (target = 1, 2, 3, or 4)
        """)
    
    st.markdown("---")
    
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
            sns.kdeplot(data=data_subset.dropna(), fill=True, alpha=0.5, linewidth=2,
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

# ============================================================================
# EDA - UNIVARIATE ANALYSIS PAGE
# ============================================================================

elif page == "EDA - Univariate Analysis":
    st.title("📊 Exploratory Data Analysis - Univariate")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset not found!")
        st.stop()
    
    st.info("Viewing: **Original Dataset** (combined from Cleveland, Switzerland, and Indian Hospital)")
    
    # Summary Statistics Section
    st.header("📋 Summary Statistics")
    st.write("Overview of the dataset's quantitative and qualitative variables:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quantitative Variables")
        quant_summary = df_original[numeric_cols].describe()
        st.dataframe(quant_summary.style.format("{:.2f}"), use_container_width=True)
    
    with col2:
        st.subheader("Qualitative Variables")
        # Convert to string type to get categorical statistics
        qual_data = df_original[categorical_cols].astype(str)
        qual_summary = qual_data.describe()
        st.dataframe(qual_summary, use_container_width=True)
    
    st.markdown("---")
    
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
    - **Serum Cholesterol** is approximately normally distributed with a slight right skew.
    - **Oldpeak** has a large peak around value 0, suggesting that a majority of patients 
      in the dataset do not have trouble with blood flow to the heart after exercising.
    - The **age** distribution spans from late 20s to late 70s, capturing a wide range of 
      cardiovascular risk profiles across all three hospital datasets.
    - **Maximum Heart Rate** shows a roughly normal distribution centered around 140-150 bpm.
    """)
    
    st.markdown("---")
    
    # Categorical Features
    st.header("Frequency of Categorical Variables by Disease Status")
    st.write("Stacked histograms showing the distribution of categorical features, stratified by heart disease presence.")
    
    categorical_cols_minusTarget = [col for col in categorical_cols if col != "target"]
    
    fig2, axs2 = plt.subplots(3, 3, figsize=(16, 12))
    axs2 = axs2.flatten()
    
    for i, var in enumerate(categorical_cols_minusTarget):
        if var in df_original.columns:
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
            axs2[i].set_xticklabels([str(int(v)) if not pd.isna(v) else '' for v in unique_vals])
            
            axs2[i].set_title(f"Distribution of {var.capitalize()} by Target", 
                             fontsize=12, fontweight="bold")
            axs2[i].set_xlabel(f"{var.capitalize()}", fontsize=11)
            axs2[i].set_ylabel(f"Count", fontsize=11)
            axs2[i].legend(labels=["No Disease", "Disease"], 
                          title="Target", 
                          fontsize=8,
                          title_fontsize=9)
    
    # Remove extra subplots
    for j in range(len(categorical_cols_minusTarget), len(axs2)):
        fig2.delaxes(axs2[j])
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    
    # Key insights for categorical variables
    st.subheader("📝 Key Insights - Categorical Variables")
    
    st.write("""
    **Major Risk Factors Identified Across All Three Hospital Datasets:**
    
    - **Gender**: Over 2/3 of males in the dataset have heart disease, whereas less than half 
      of females have disease, suggesting gender is an important risk factor. This pattern is 
      consistent across Cleveland, Switzerland, and Indian hospital data.
    
    - **Chest Pain (chestpain = 4)**: Patients with asymptomatic chest pain are overwhelmingly 
      in the disease class. This suggests that asymptomatic chest pain appears to be a 
      strong indicator of heart disease risk.
    
    - **Resting ECG (restingrelectro = 1)**: Patients with ST-T wave abnormality (T wave inversions 
      and/or ST elevation or depression > 0.05 mV) are predominantly in the disease class, 
      indicating that ST-T wave abnormalities are a strong predictor of heart disease.
    
    - **Exercise Induced Angina (exerciseangia = 1)**: The proportion of patients with heart disease 
      is substantially higher among those who experience exercise-induced angina compared to 
      those who do not.
    
    - **ST Segment Slope**: Patients whose ST segment slope is flat or downsloping during 
      peak exercise show a much higher disease rate compared to those with an upsloping pattern.
    
    - **Number of Vessels (noofmajorvessels > 0)**: The proportion of disease cases increases sharply when 
      the number of major vessels colored by fluoroscopy is greater than 0, suggesting that 
      any vessel blockage visible on fluoroscopy indicates elevated heart disease risk.
    """)

# ============================================================================
# PAIR-PLOT ANALYSIS PAGE
# ============================================================================

elif page == "Pair-Plot Analysis":
    st.title("🔗 Pair-Plot Analysis")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset not found!")
        st.stop()
    
    st.info("Viewing: **Original Dataset** (combined from all three hospital sources)")
    
    st.write("""
    Pairwise relationships between continuous features, showing both scatter plots 
    and distribution curves (KDE), colored by disease status.
    """)
    
    with st.spinner("Generating pairplot... This may take a moment."):
        # Create pairplot
        plot_data = df_original[numeric_cols + ["target"]].dropna()
        
        g = sns.pairplot(
            data=plot_data,
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
        - **Age and Max Heart Rate** show a moderate negative correlation. 
          Older patients tend to have a lower maximum heart rate.
        
        - **Oldpeak and Disease**: Patients with heart disease show higher oldpeak values 
          (ST depression) compared to those without heart disease, suggesting that greater 
          oldpeak values increase risk of heart disease.
        
        - The pairwise correlations between predictors are generally weak (|r| < 0.3), 
          indicating that multicollinearity should not be a major concern in modeling.
        """)
    
    with insights_col2:
        st.subheader("📊 Distribution Differences")
        st.write("""
        - **Maximum Heart Rate**: The kernel density estimate for the no disease 
          group reveals that patients without the disease achieve a greater maximum heart 
          rate on average, compared to those with disease.
        
        - **Age Distribution**: The median age is higher for patients with disease compared 
          to those without. The no disease group shows greater age variability, with a wider 
          distribution and lower peak density.
        
        - **Serum Cholesterol**: Higher density regions for elevated cholesterol values 
          are observed in the at-risk class compared to the healthy class.
        """)
    
    st.markdown("---")
    
    st.info("""
    **Interpretation Note:** While individual features show overlap between groups, 
    combinations of features (visible in the scatter plots) may provide better discrimination 
    for classification models. The pair-plots reveal that multivariate patterns are more 
    informative than univariate distributions alone.
    """)

# ============================================================================
# MISSINGNESS ANALYSIS PAGE
# ============================================================================

elif page == "Missingness Analysis":
    st.title("🔍 Missing Data Analysis")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset with missing values not found!")
        st.stop()
    
    st.write("""
    Analysis of missing data patterns in the original combined dataset before imputation. 
    Understanding missingness mechanisms is crucial for appropriate handling strategies.
    """)
    
    # Missing value counts
    st.header("📊 Missing Value Summary")
    
    # Exclude the missing indicator columns from the analysis
    analysis_cols = [col for col in df_original.columns if not col.endswith('_missing')]
    df_analysis = df_original[analysis_cols]
    
    missing_counts = df_analysis.isna().sum()
    missing_pct = (missing_counts / len(df_analysis) * 100).round(2)
    
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
    missing_mask = df_analysis.isna()
    sns.heatmap(missing_mask, cbar=False, cmap="viridis", 
               yticklabels=False, ax=ax2)
    ax2.set_title("Missing Value Patterns in the Combined Dataset", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Features", fontsize=13)
    ax2.set_ylabel("Row Index", fontsize=13)
    st.pyplot(fig2)
    plt.close()
    
    st.markdown("---")
    
    # Analysis and interpretation
    st.header("📝 Missingness Mechanism Analysis")
    
    st.subheader("🔬 Missing Not At Random (MNAR) Assessment")
    
    st.write("""
    The majority of missing data comes from the **Switzerland dataset** and is very likely 
    **Missing Not At Random (MNAR)**. The **Indian Hospital dataset** also contributes missing 
    values, particularly for serum cholesterol. This has important implications for imputation strategies:
    """)
    
    st.write("**Key Variables with MNAR patterns:**")
    
    with st.expander("**noofmajorvessels (Number of vessels via fluoroscopy)**"):
        st.write("""
        - **Why it might be missing**: Cardiac catheterization requires X-ray procedures to test 
          arterial health and blood flow. The procedure is costly and cannot be performed on every patient.
        - **MNAR reasoning**: The decision to perform cardiac catheterization is 
          related to the patient's suspected heart disease severity
        - **Implication**: Missing values are actually informative - they may indicate a lower risk profile.
        """)
    
    with st.expander("**slope (Exercise test results)**"):
        st.write("""
        - These variables require specialized stress testing
        - Missing values likely indicate patients who didn't undergo complete cardiac workup
        - Missingness may be related to disease severity or hospital protocols
        - Different hospitals (Cleveland, Switzerland, India) have varying testing procedures
        """)
    
    with st.expander("**fastingbloodsugar (Fasting blood sugar)**"):
        st.write("""
        - Requires patient to be fasted before measurement
        - Cleveland patients have fbs values collected, but Switzerland data doesn't
        - **MNAR reasoning**: Differences in hospital procedures/policies across countries
        """)
    
    with st.expander("**serumcholestrol (Cholesterol)**"):
        st.write("""
        - Missing cholesterol data comes from both Switzerland and Indian Hospital datasets
        - Originally labeled as value 0 before conversion to NaN
        - Represents data collection differences between hospital databases
        - Indian Hospital data shows cholesterol measurement wasn't standard for all patients
        """)
    
    st.markdown("---")
    
    st.subheader("💡 Imputation Strategy")
    
    st.write("""
    Given the MNAR nature of the missing data, this analysis employs multiple strategies:
    
    1. **Missing Indicators**: Created binary indicator variables (`serumcholestrol_missing`, 
       `noofmajorvessels_missing`, `fastingbloodsugar_missing`, `slope_missing`) to capture 
       the information in the missingness itself
    
    2. **Simple Imputation**: Applied median/mode imputation as a baseline approach
    
    3. **KNN Imputation**: Applied K-Nearest Neighbors imputation (K=5) as an advanced method
    
    4. **Comparative Analysis**: Correlation patterns are compared across imputation methods 
       to assess the impact of different strategies
    """)
    
    st.info("""
    **Important**: Because data is likely MNAR, any imputation method may introduce bias. 
    The missing indicators help capture whether a test was performed, which can add predictive power
    for the classification models. For final modeling, we use **complete case analysis** (dropping 
    rows with missing values) to avoid imputation bias.
    """)

# ============================================================================
# CORRELATION ANALYSIS PAGE
# ============================================================================

elif page == "Correlation Analysis":
    st.title("🔗 Correlation Analysis")
    st.markdown("---")
    
    if df_original is None or df_simple is None or df_knn is None:
        st.error("One or more required datasets not found! Please ensure all CSV files exist.")
        st.stop()
    
    st.write("""
    Compare correlation patterns across different imputation methods to understand how 
    missing data handling affects feature relationships. Data is combined from Cleveland, 
    Switzerland, and Indian Hospital sources.
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
        Some correlations change directions from positive to negative (and vice versa) 
        after simple imputation, such as certain variable pairs with noofmajorvessels and slope.
        
        Some correlation coefficient magnitudes change drastically as well.
        
        **Interpretation**: The spikes in correlation coefficient magnitude along with direction 
        changes support reasoning that data is MNAR. Simple imputation (mode) does not account 
        for the relationships between variables, leading to biased estimates.
        """)
        
        st.markdown("---")
        
        st.subheader("🔍 KNN Imputation Effects")
        st.write("""
        Similar to simple imputation, some correlations change direction completely after KNN 
        imputation. Notable patterns include dramatic changes in correlations involving 
        slope and noofmajorvessels with the target variable.
        
        **Interpretation**: The many severe cases of direction reversal indicate that KNN is 
        likely imputing poorly - learning patterns from incorrect neighbors. This strengthens 
        the argument that the data is MNAR, and justifies our decision to use complete case 
        analysis (dropping missing values) for final model fitting.
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

# ============================================================================
# LOGISTIC REGRESSION ANALYSIS PAGE
# ============================================================================

elif page == "Logistic Regression Analysis":
    st.title("📊 Logistic Regression Analysis")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset not found!")
        st.stop()
    
    st.info("""
    This page fits a logistic regression model using **all available predictors** from the 
    combined dataset (Cleveland, Switzerland, and Indian Hospital data).
    
    **Note:** This analysis uses **complete case analysis** - rows with missing values are 
    excluded to avoid imputation bias given the MNAR nature of the data. Missing data indicator 
    variables are also excluded from the model.
    """)
    
    # =========================================================================
    # FEATURE ENGINEERING SECTION
    # =========================================================================
    
    st.header("🔧 Feature Engineering Steps")
    
    st.write("""
    Before fitting the logistic regression model, several feature engineering steps were 
    performed during the data preparation phase to ensure data quality and compatibility 
    across all three hospital datasets:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1️⃣ Column Standardization")
        st.write("""
        The three datasets (Cleveland, Switzerland, Indian Hospital) used different column 
        naming conventions. All columns were renamed to a consistent format:
        """)
        st.code("""
# Renaming for consistency across datasets
rename_mapping = {
    "sex" : "gender",
    "cp" : "chestpain",
    "trestbps" : "restingBP",
    "chol" : "serumcholestrol",
    "fbs" : "fastingbloodsugar",
    "restecg" : "restingrelectro",
    "thalach" : "maxheartrate",
    "exang" : "exerciseangia",
    "ca" : "noofmajorvessels",
    "num" : "target"
}
        """, language="python")
        
        st.subheader("2️⃣ Categorical Encoding Alignment")
        st.write("""
        The Indian Hospital dataset used different encoding for chest pain (0-3 instead of 1-4). 
        This was re-mapped to match Cleveland and Switzerland:
        """)
        st.code("""
# Chest pain encoding alignment
cp_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
df3["chestpain"] = df3["chestpain"].replace(cp_mapping)
        """, language="python")
    
    with col2:
        st.subheader("3️⃣ Missing Value Indicators (MNAR)")
        st.write("""
        Since data is likely **Missing Not At Random (MNAR)**, binary indicator variables 
        were created to capture the informativeness of missingness:
        """)
        st.code("""
# Create MNAR indicator columns
df["serumcholestrol_missing"] = df["serumcholestrol"].isna().astype(int)
df["noofmajorvessels_missing"] = df["noofmajorvessels"].isna().astype(int)
df["fastingbloodsugar_missing"] = df["fastingbloodsugar"].isna().astype(int)
df["slope_missing"] = df["slope"].isna().astype(int)
        """, language="python")
        
        st.subheader("4️⃣ Data Type Conversions")
        st.write("""
        Variables were converted to appropriate data types for modeling:
        """)
        st.code("""
# Numeric columns
numeric_cols = ["age", "restingBP", "serumcholestrol", 
                "maxheartrate", "oldpeak"]
for var in numeric_cols:
    df[var] = pd.to_numeric(df[var], errors="coerce")

# Categorical columns  
categorical_cols = ["gender", "chestpain", "fastingbloodsugar",
                    "restingrelectro", "exerciseangia", "slope",
                    "noofmajorvessels", "target"]
for var in categorical_cols:
    df[var] = df[var].astype("category")
        """, language="python")
    
    st.subheader("5️⃣ Target Variable Encoding")
    st.write("""
    The original target variable had values 0-4 representing severity of vessel narrowing. 
    This was converted to binary classification:
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.code("""
# Binary encoding: 0 = No disease, 1 = Disease present
df["target"] = np.where(df["target"].isin([1, 2, 3, 4]), 1, 0)
        """, language="python")
    with col2:
        st.write("""
        - **0**: No heart disease
        - **1**: Heart disease (1+ vessels with >50% narrowing)
        """)
    
    st.subheader("6️⃣ One-Hot Encoding for Model Fitting")
    st.write("""
    Categorical variables are converted to dummy variables using one-hot encoding with 
    `drop_first=True` to avoid multicollinearity (the dummy variable trap):
    """)
    st.code("""
# One-hot encoding for logistic regression
X = pd.get_dummies(X, drop_first=True).astype(float)
X = sm.add_constant(X)  # Add intercept term
    """, language="python")
    
    st.markdown("---")
    
    # Import required libraries
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
    
    # Prepare data
    df_analysis = df_original.copy()
    
    # Drop rows with missing values (complete case analysis)
    df_clean = df_analysis.dropna()
    
    # Exclude missing indicator columns
    exclude_cols = [col for col in df_clean.columns if col.endswith('_missing')]
    feature_cols = [col for col in df_clean.columns if col != 'target' and col not in exclude_cols]
    
    # Display data info
    st.subheader("📋 Data Preparation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Observations", len(df_analysis))
    with col2:
        st.metric("Complete Cases Used", len(df_clean))
    with col3:
        st.metric("Excluded (Missing)", len(df_analysis) - len(df_clean))
    
    st.write(f"**Features used in model:** {', '.join(feature_cols)}")
    
    st.markdown("---")
    
    # Prepare X and y
    X = df_clean[feature_cols].copy()
    y = df_clean['target'].astype(int)
    
    # Convert categorical to dummies
    X = pd.get_dummies(X, drop_first=True).astype(float)
    X = sm.add_constant(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Fit the model
    with st.spinner("Fitting logistic regression model..."):
        logit_model = sm.Logit(y_train, X_train)
        result = logit_model.fit(disp=0)
    
    st.session_state.logistic_model_fitted = True
    
    # Model Summary Statistics
    st.subheader("📋 Model Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pseudo R²", f"{result.prsquared:.4f}")
    with col2:
        st.metric("Log-Likelihood", f"{result.llf:.2f}")
    with col3:
        st.metric("AIC", f"{result.aic:.2f}")
    with col4:
        st.metric("BIC", f"{result.bic:.2f}")
    
    st.markdown("---")
    
    # Coefficient Estimates Table
    st.subheader("📊 Coefficient Estimates and Statistical Tests")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Variable': result.params.index,
        'Coefficient (β)': result.params.values,
        'Std Error': result.bse.values,
        'z-value': result.tvalues.values,
        'P-value': result.pvalues.values,
        'CI Lower (95%)': result.conf_int()[0].values,
        'CI Upper (95%)': result.conf_int()[1].values
    })
    
    # Add significance stars
    def add_significance_stars(p_val):
        if p_val < 0.001:
            return '***'
        elif p_val < 0.01:
            return '**'
        elif p_val < 0.05:
            return '*'
        elif p_val < 0.1:
            return '.'
        else:
            return ''
    
    results_df['Significance'] = results_df['P-value'].apply(add_significance_stars)
    
    # Style the dataframe
    def color_pvalues(val):
        if val < 0.001:
            return 'background-color: #d4edda; font-weight: bold'
        elif val < 0.01:
            return 'background-color: #d1ecf1; font-weight: bold'
        elif val < 0.05:
            return 'background-color: #fff3cd'
        elif val < 0.1:
            return 'background-color: #f8f9fa'
        else:
            return ''
    
    styled_results = results_df.style.format({
        'Coefficient (β)': '{:.4f}',
        'Std Error': '{:.4f}',
        'z-value': '{:.4f}',
        'P-value': '{:.6f}',
        'CI Lower (95%)': '{:.4f}',
        'CI Upper (95%)': '{:.4f}'
    }).applymap(color_pvalues, subset=['P-value'])
    
    st.dataframe(styled_results, hide_index=True, use_container_width=True)
    
    st.caption("""
    **Significance codes:** *** p < 0.001, ** p < 0.01, * p < 0.05, . p < 0.1
    """)
    
    st.markdown("---")
    
    # Odds Ratios
    st.subheader("🎯 Odds Ratios (Exponentiated Coefficients)")
    
    st.info("""
    **What are Odds Ratios?**
    
    Odds ratios are the exponentiated coefficients (e^β) and provide a more interpretable measure:
    - **OR = 1**: No effect on odds of heart disease
    - **OR > 1**: Increased odds of heart disease
    - **OR < 1**: Decreased odds of heart disease
    """)
    
    # Calculate odds ratios
    odds_ratios_df = pd.DataFrame({
        'Variable': result.params.index,
        'Coefficient (β)': result.params.values,
        'Odds Ratio (e^β)': np.exp(result.params.values),
        'OR CI Lower (95%)': np.exp(result.conf_int()[0].values),
        'OR CI Upper (95%)': np.exp(result.conf_int()[1].values),
        'P-value': result.pvalues.values,
        'Significance': results_df['Significance']
    })
    
    st.dataframe(odds_ratios_df.style.format({
        'Coefficient (β)': '{:.4f}',
        'Odds Ratio (e^β)': '{:.4f}',
        'OR CI Lower (95%)': '{:.4f}',
        'OR CI Upper (95%)': '{:.4f}',
        'P-value': '{:.6f}'
    }), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Model Performance on Test Set
    st.subheader("📈 Model Performance on Test Set")
    
    # Predictions
    y_pred_prob = result.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.4f}")
    with col2:
        from sklearn.metrics import precision_score
        prec = precision_score(y_test, y_pred)
        st.metric("Precision", f"{prec:.4f}")
    with col3:
        from sklearn.metrics import recall_score
        rec = recall_score(y_test, y_pred)
        st.metric("Recall", f"{rec:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix and ROC Curve side by side
    st.subheader("📊 Confusion Matrix & ROC Curve")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        ax_cm.set_xlabel('Predicted', fontsize=12)
        ax_cm.set_ylabel('Actual', fontsize=12)
        ax_cm.set_title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
        st.pyplot(fig_cm)
        plt.close()
    
    with col2:
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, alpha=0.3)
        st.pyplot(fig_roc)
        plt.close()
    
    st.markdown("---")
    
    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Key Findings
    st.subheader("📝 Key Findings - Significant Predictors")
    
    sig_predictors = results_df[
        (results_df['P-value'] < 0.05) & (results_df['Variable'] != 'const')
    ].sort_values('P-value')
    
    if len(sig_predictors) > 0:
        st.success(f"**Statistically Significant Predictors (p < 0.05):** {len(sig_predictors)}")
        
        for idx, row in sig_predictors.iterrows():
            var_name = row['Variable']
            coef = row['Coefficient (β)']
            p_val = row['P-value']
            or_val = np.exp(coef)
            
            if or_val > 1:
                direction = "increases"
                pct_change = (or_val - 1) * 100
            else:
                direction = "decreases"
                pct_change = (1 - or_val) * 100
            
            st.write(f"• **{var_name}**: A one-unit increase {direction} the odds of heart disease by {pct_change:.1f}% (OR = {or_val:.3f}, p = {p_val:.6f})")
    else:
        st.warning("No statistically significant predictors at the α = 0.05 level.")
    
    # Additional model diagnostics
    with st.expander("📊 Additional Model Diagnostics"):
        st.write("### Full Model Summary")
        st.text(result.summary())
        
        st.write("### Likelihood Ratio Test")
        st.write(f"**LR statistic:** {result.llr:.4f}")
        st.write(f"**LR p-value:** {result.llr_pvalue:.6f}")
        
        if result.llr_pvalue < 0.05:
            st.success("The model is significantly better than the null model (intercept only).")
        else:
            st.warning("The model is not significantly better than the null model.")

# ============================================================================
# RANDOM FOREST ANALYSIS PAGE
# ============================================================================

elif page == "Random Forest Analysis":
    st.title("🌲 Random Forest Analysis with Grid Search")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset not found!")
        st.stop()
    
    st.info("""
    This page fits a **Random Forest classifier** using Grid Search cross-validation to find 
    optimal hyperparameters. Random Forest is an ensemble method that builds multiple decision 
    trees and aggregates their predictions for more robust results.
    """)
    
    # Import required libraries
    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
    import statsmodels.api as sm
    
    # Prepare data
    df_analysis = df_original.copy()
    df_clean = df_analysis.dropna()
    
    # Exclude missing indicator columns
    exclude_cols = [col for col in df_clean.columns if col.endswith('_missing')]
    feature_cols = [col for col in df_clean.columns if col != 'target' and col not in exclude_cols]
    
    st.markdown("---")
    
    # Grid Search Explanation
    st.header("🔍 Grid Search Cross-Validation Approach")
    
    st.write("""
    **Grid Search** is a hyperparameter tuning technique that systematically searches through 
    a specified parameter grid to find the optimal combination of hyperparameters. Combined 
    with **5-fold cross-validation**, it provides robust estimates of model performance.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Hyperparameter Grid")
        st.code("""
param_grid = {
    'n_estimators': [100, 200, 300],     # Number of trees
    'max_features': ['sqrt', 0.5],       # Features per split
    'max_depth': [10, 20, None],         # Tree depth
    'min_samples_split': [2, 5]          # Min samples to split
}
        """, language="python")
    
    with col2:
        st.subheader("🎯 Why These Parameters?")
        st.write("""
        - **n_estimators**: More trees generally improve performance but increase computation
        - **max_features**: Controls randomness - 'sqrt' is common for classification
        - **max_depth**: Prevents overfitting; None allows full tree growth
        - **min_samples_split**: Higher values prevent overfitting on noisy data
        """)
    
    st.info("**Total models fitted:** 3 × 2 × 3 × 2 × 5 folds = **180 models**")
    
    st.markdown("---")
    
    # Data preparation
    st.subheader("📋 Data Preparation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Observations", len(df_analysis))
    with col2:
        st.metric("Complete Cases Used", len(df_clean))
    with col3:
        st.metric("Excluded (Missing)", len(df_analysis) - len(df_clean))
    
    # Prepare X and y
    X = df_clean[feature_cols].copy()
    y = df_clean['target'].astype(int)
    
    # Convert categorical to dummies (without constant for RF)
    X = pd.get_dummies(X, drop_first=True).astype(float)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    st.markdown("---")
    
    # Fit Random Forest with Grid Search
    st.subheader("🚀 Model Training")
    
    with st.spinner("Running Grid Search... This may take a moment."):
        # Define the model and parameter grid
        rf = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 0.5],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        # 5-fold CV
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='accuracy',
            verbose=0,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
    
    st.session_state.rf_model_fitted = True
    best_rf_model = grid_search.best_estimator_
    
    # Best parameters
    st.success("✅ Grid Search Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Best Hyperparameters")
        best_params_df = pd.DataFrame({
            'Parameter': list(grid_search.best_params_.keys()),
            'Best Value': list(grid_search.best_params_.values())
        })
        st.dataframe(best_params_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("📊 Cross-Validation Score")
        st.metric("Best CV Accuracy", f"{grid_search.best_score_:.4f}")
        st.caption("Average accuracy across 5 folds with best parameters")
    
    st.markdown("---")
    
    # Model Performance
    st.subheader("📈 Model Performance on Test Set")
    
    # Predictions
    y_pred_rf = best_rf_model.predict(X_test)
    y_pred_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        acc_rf = accuracy_score(y_test, y_pred_rf)
        st.metric("Accuracy", f"{acc_rf:.4f}")
    with col2:
        from sklearn.metrics import precision_score
        prec_rf = precision_score(y_test, y_pred_rf)
        st.metric("Precision", f"{prec_rf:.4f}")
    with col3:
        from sklearn.metrics import recall_score
        rec_rf = recall_score(y_test, y_pred_rf)
        st.metric("Recall", f"{rec_rf:.4f}")
    with col4:
        from sklearn.metrics import f1_score
        f1_rf = f1_score(y_test, y_pred_rf)
        st.metric("F1 Score", f"{f1_rf:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix and ROC Curve
    st.subheader("📊 Confusion Matrix & ROC Curve")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax_cm,
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        ax_cm.set_xlabel('Predicted', fontsize=12)
        ax_cm.set_ylabel('Actual', fontsize=12)
        ax_cm.set_title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
        st.pyplot(fig_cm)
        plt.close()
    
    with col2:
        # ROC Curve
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr_rf, tpr_rf, color='forestgreen', lw=2, label=f'ROC Curve (AUC = {roc_auc_rf:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, alpha=0.3)
        st.pyplot(fig_roc)
        plt.close()
    
    st.markdown("---")
    
    # Classification Report
    st.subheader("📋 Classification Report")
    report_rf = classification_report(y_test, y_pred_rf, target_names=['No Disease', 'Disease'], output_dict=True)
    report_rf_df = pd.DataFrame(report_rf).transpose()
    st.dataframe(report_rf_df.style.format("{:.4f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance Analysis
    st.header("🎯 Feature Importance Analysis")
    
    st.write("""
    Feature importance in Random Forest is computed based on how much each feature 
    contributes to reducing impurity (Gini impurity) across all trees. Higher values 
    indicate more important features for predicting heart disease.
    """)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance (Gini)': best_rf_model.feature_importances_
    }).sort_values('Importance (Gini)', ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Importance Scores")
        st.dataframe(
            feature_importance.style.format({'Importance (Gini)': '{:.4f}'}).background_gradient(
                subset=['Importance (Gini)'], cmap='YlOrRd'
            ),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.subheader("📈 Feature Importance Plot")
        fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        top_n = min(15, len(feature_importance))
        top_features = feature_importance.head(top_n)
        
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_n))[::-1]
        
        bars = ax_imp.barh(range(top_n), top_features['Importance (Gini)'].values[::-1], 
                          color=colors, edgecolor='black')
        ax_imp.set_yticks(range(top_n))
        ax_imp.set_yticklabels(top_features['Feature'].values[::-1])
        ax_imp.set_xlabel('Gini Importance', fontsize=12)
        ax_imp.set_title('Top Features by Gini Importance\n(Random Forest)', fontsize=14, fontweight='bold')
        ax_imp.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, top_features['Importance (Gini)'].values[::-1]):
            ax_imp.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close()
    
    st.markdown("---")
    
    # Key Insights on Feature Importance
    st.subheader("📝 Key Insights - Important Predictors")
    
    top_3 = feature_importance.head(3)
    
    st.success("**Top 3 Most Important Features for Heart Disease Prediction:**")
    
    for i, (idx, row) in enumerate(top_3.iterrows(), 1):
        st.write(f"""
        **{i}. {row['Feature']}** (Importance: {row['Importance (Gini)']:.4f})
        """)
    
    st.write("""
    **Interpretation:**
    
    The Random Forest model identifies the most powerful splitting conditions for predicting 
    heart disease. Features with higher Gini importance are more effective at separating 
    patients with and without heart disease. These findings align with our exploratory 
    analysis, confirming that:
    
    - Chest pain type, exercise-induced symptoms, and ST segment characteristics are 
      critical clinical indicators
    - The number of major vessels colored by fluoroscopy provides significant predictive value
    - Age and maximum heart rate contribute meaningfully to risk stratification
    """)

# ============================================================================
# MODEL COMPARISON & RECOMMENDATIONS PAGE
# ============================================================================

elif page == "Model Comparison & Recommendations":
    st.title("⚖️ Model Comparison & Recommendations")
    st.markdown("---")
    
    if df_original is None:
        st.error("Original dataset not found!")
        st.stop()
    
    st.write("""
    This page compares the performance of **Logistic Regression** and **Random Forest** 
    models, discusses their trade-offs, and provides recommendations based on the findings 
    from analyzing heart disease data from Cleveland, Switzerland, and Indian Hospital sources.
    """)
    
    # Re-run both models to get comparison metrics
    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import statsmodels.api as sm
    
    # Prepare data
    df_analysis = df_original.copy()
    df_clean = df_analysis.dropna()
    exclude_cols = [col for col in df_clean.columns if col.endswith('_missing')]
    feature_cols = [col for col in df_clean.columns if col != 'target' and col not in exclude_cols]
    
    X = df_clean[feature_cols].copy()
    y = df_clean['target'].astype(int)
    X = pd.get_dummies(X, drop_first=True).astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Fit Logistic Regression
    X_train_lr = sm.add_constant(X_train)
    X_test_lr = sm.add_constant(X_test)
    logit_model = sm.Logit(y_train, X_train_lr)
    result_lr = logit_model.fit(disp=0)
    y_pred_prob_lr = result_lr.predict(X_test_lr)
    y_pred_lr = (y_pred_prob_lr >= 0.5).astype(int)
    
    # Fit Random Forest
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 0.5],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    y_pred_prob_rf = best_rf.predict_proba(X_test)[:, 1]
    
    st.markdown("---")
    
    # Performance Comparison Table
    st.header("📊 Performance Comparison")
    
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
        'Logistic Regression': [
            accuracy_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_lr),
            recall_score(y_test, y_pred_lr),
            f1_score(y_test, y_pred_lr),
            roc_auc_score(y_test, y_pred_prob_lr)
        ],
        'Random Forest': [
            accuracy_score(y_test, y_pred_rf),
            precision_score(y_test, y_pred_rf),
            recall_score(y_test, y_pred_rf),
            f1_score(y_test, y_pred_rf),
            roc_auc_score(y_test, y_pred_prob_rf)
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Difference'] = comparison_df['Random Forest'] - comparison_df['Logistic Regression']
    comparison_df['Better Model'] = comparison_df['Difference'].apply(
        lambda x: '🌲 Random Forest' if x > 0.005 else ('📊 Logistic Regression' if x < -0.005 else '🤝 Similar')
    )
    
    st.dataframe(
        comparison_df.style.format({
            'Logistic Regression': '{:.4f}',
            'Random Forest': '{:.4f}',
            'Difference': '{:+.4f}'
        }).background_gradient(subset=['Logistic Regression', 'Random Forest'], cmap='YlGn'),
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown("---")
    
    # ROC Curves Comparison
    st.header("📈 ROC Curve Comparison")
    
    from sklearn.metrics import roc_curve, auc
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
    auc_lr = auc(fpr_lr, tpr_lr)
    auc_rf = auc(fpr_rf, tpr_rf)
    
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
    ax_roc.plot(fpr_rf, tpr_rf, color='forestgreen', lw=2, label=f'Random Forest (AUC = {auc_rf:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    st.pyplot(fig_roc)
    plt.close()
    
    st.markdown("---")
    
    # Trade-offs Discussion
    st.header("⚖️ Model Trade-offs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Logistic Regression")
        st.write("**Advantages:**")
        st.write("""
        - ✅ **Interpretability**: Coefficients directly show feature effects (odds ratios)
        - ✅ **Statistical inference**: P-values and confidence intervals for hypothesis testing
        - ✅ **Simplicity**: Easy to explain to stakeholders and clinicians
        - ✅ **Fast training**: Computationally efficient
        - ✅ **Probabilistic output**: Well-calibrated probability estimates
        """)
        
        st.write("**Disadvantages:**")
        st.write("""
        - ❌ **Linear assumption**: Assumes linear relationship between log-odds and features
        - ❌ **Feature interactions**: Doesn't automatically capture interactions
        - ❌ **Complex patterns**: May miss non-linear relationships
        """)
    
    with col2:
        st.subheader("🌲 Random Forest")
        st.write("**Advantages:**")
        st.write("""
        - ✅ **Non-linear patterns**: Captures complex relationships automatically
        - ✅ **Feature interactions**: Naturally handles interactions between variables
        - ✅ **Robust**: Less sensitive to outliers and noise
        - ✅ **Feature importance**: Built-in variable importance measure
        - ✅ **No assumptions**: Non-parametric, flexible approach
        """)
        
        st.write("**Disadvantages:**")
        st.write("""
        - ❌ **Black box**: Harder to interpret individual predictions
        - ❌ **Overfitting risk**: Can overfit with too many trees or depth
        - ❌ **Computational cost**: Slower training than logistic regression
        - ❌ **Probability calibration**: May need calibration for accurate probabilities
        """)
    
    st.markdown("---")
    
    # Recommendation
    st.header("🎯 Recommendation")
    
    # Determine which model performed better
    lr_avg = np.mean([accuracy_score(y_test, y_pred_lr), f1_score(y_test, y_pred_lr), roc_auc_score(y_test, y_pred_prob_lr)])
    rf_avg = np.mean([accuracy_score(y_test, y_pred_rf), f1_score(y_test, y_pred_rf), roc_auc_score(y_test, y_pred_prob_rf)])
    
    if rf_avg > lr_avg + 0.02:
        recommended = "Random Forest"
        color = "success"
    elif lr_avg > rf_avg + 0.02:
        recommended = "Logistic Regression"
        color = "info"
    else:
        recommended = "Either model (similar performance)"
        color = "warning"
    
    if color == "success":
        st.success(f"**Recommended Model: {recommended}**")
    elif color == "info":
        st.info(f"**Recommended Model: {recommended}**")
    else:
        st.warning(f"**Recommended Model: {recommended}**")
    
    st.write("""
    **Rationale:**
    
    Based on the analysis of combined heart disease data from three hospital sources:
    
    1. **For Clinical Decision Support**: Logistic Regression is preferred when interpretability 
       is crucial. Clinicians can understand exactly how each risk factor (chest pain type, 
       ST segment slope, number of vessels, etc.) contributes to the predicted risk.
    
    2. **For Screening/Triage**: Random Forest may be preferred when maximizing predictive 
       accuracy is the priority, especially if the model will be used as a preliminary 
       screening tool before clinical evaluation.
    
    3. **For Research**: Logistic Regression provides statistical inference (p-values, 
       confidence intervals) that support hypothesis testing about risk factors.
    """)
    
    st.markdown("---")
    
    # Clinical Recommendations
    st.header("💊 Clinical Recommendations Based on Findings")
    
    st.write("""
    Based on the analysis of heart disease risk factors across Cleveland, Switzerland, and 
    Indian Hospital datasets, the following recommendations emerge:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Key Risk Factors Identified")
        st.write("""
        1. **Chest Pain Type**: Asymptomatic chest pain (type 4) is strongly associated with 
           heart disease - patients should not dismiss lack of typical symptoms
        
        2. **Exercise Response**: ST segment depression during exercise (high oldpeak) and 
           flat/downsloping ST segments indicate higher risk
        
        3. **Exercise-Induced Angina**: Presence of angina during exercise is a significant 
           warning sign
        
        4. **Vessel Involvement**: Any major vessel blockage visible on fluoroscopy indicates 
           substantially elevated risk
        
        5. **Maximum Heart Rate**: Lower maximum heart rate during exercise correlates with 
           higher disease risk
        """)
    
    with col2:
        st.subheader("📋 Recommendations for Risk Reduction")
        st.write("""
        1. **Regular Cardiac Screening**: Especially for males over 50 and individuals with 
           known risk factors
        
        2. **Exercise Stress Testing**: Important diagnostic tool - abnormal results warrant 
           further investigation
        
        3. **Lifestyle Modifications**:
           - Regular physical activity to improve cardiac function
           - Heart-healthy diet to manage cholesterol
           - Blood pressure management
        
        4. **Symptom Awareness**: Don't ignore atypical symptoms - asymptomatic presentation 
           can still indicate disease
        
        5. **Follow-up Care**: Patients with any risk factors should maintain regular 
           cardiovascular monitoring
        """)
    
    st.markdown("---")
    
    # Limitations
    st.header("⚠️ Limitations and Future Directions")
    
    st.write("""
    **Study Limitations:**
    
    1. **Missing Data**: The MNAR nature of missing data (especially for noofmajorvessels and 
       serumcholestrol) required complete case analysis, which may introduce selection bias
    
    2. **Data Heterogeneity**: Combining data from three different hospitals/countries 
       introduces variability in measurement protocols and patient populations
    
    3. **Temporal Considerations**: The datasets span different time periods, and clinical 
       practices may have evolved
    
    4. **External Validation**: Models should be validated on independent datasets before 
       clinical deployment
    
    **Future Directions:**
    
    - Implement more sophisticated missing data handling (e.g., multiple imputation with 
      sensitivity analysis)
    - Explore ensemble methods combining both models
    - Incorporate additional features (lifestyle factors, genetic markers)
    - Develop risk stratification thresholds for clinical use
    """)

# ============================================================================
# FOOTER / SIDEBAR INFO
# ============================================================================

st.sidebar.markdown("---")
if df_original is not None:
    st.sidebar.info(f"""
    **📊 Dataset Info:**  
    **Total Patients:** {df_original.shape[0]}  
    **Features:** {df_original.shape[1]}  
    **Data Sources:** 3 hospitals  
    **Disease Prevalence:** {(df_original['target'].astype(int).sum() / len(df_original) * 100):.1f}%
    """)

st.sidebar.markdown("---")
st.sidebar.success("""
**🔧 Data Processing:**
- ✅ Three datasets integrated
- ✅ MNAR indicators created
- ✅ Two imputation methods applied
- ✅ Complete case analysis for modeling
""")

st.sidebar.markdown("---")
st.sidebar.write("**📚 Data Sources:**")
st.sidebar.write("[UCI Heart Disease (Cleveland & Switzerland)](https://archive.ics.uci.edu/dataset/45/heart+disease)")
st.sidebar.write("[Mendeley Data (Indian Hospital)](https://data.mendeley.com/datasets/dzz48mvjht/1)")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit | CMSE 830 Final Project")
