# Heart Disease Risk Factor Analysis

Interactive data analysis and machine learning application exploring risk factors for heart disease using combined datasets from three international hospital sources.

## Overview

This project analyzes cardiovascular health data from **Cleveland Clinic**, **Switzerland Hospital**, and **Multispeciality Indian Hospital** databases to identify key risk factors contributing to heart disease. The application provides comprehensive exploratory data analysis, missing data handling strategies, multiple imputation approaches, correlation analysis, and **predictive modeling with Logistic Regression and Random Forest classifiers**.

## Target Audience

- Individuals ages 25-70 concerned about heart disease risk
- Healthcare providers analyzing cardiovascular risk factors
- Medical researchers studying heart disease patterns
- Data science students learning applied machine learning techniques

## Features

### Data Exploration
- Interactive data exploration with age-based filtering
- Dynamic feature distribution visualizations
- Summary statistics for quantitative and qualitative variables

### Exploratory Data Analysis
- Univariate analysis with distribution plots for all features
- Bivariate analysis with pair-plots colored by disease status
- Stacked histograms showing categorical features by heart disease outcome

### Missing Data Analysis
- Missing data pattern visualization (heatmaps)
- MNAR (Missing Not At Random) assessment and discussion
- Comparison of Simple (median/mode) and KNN imputation methods
- Correlation heatmaps comparing imputation effects

### Machine Learning Models
- **Logistic Regression Analysis**
  - Feature engineering documentation
  - Coefficient estimates with statistical significance testing
  - Odds ratios with confidence intervals
  - ROC curve and confusion matrix
  - Model diagnostics and interpretation

- **Random Forest Analysis**
  - Grid Search cross-validation for hyperparameter tuning
  - Feature importance analysis (Gini importance scores)
  - ROC curve and confusion matrix
  - Identification of most predictive features

### Model Comparison & Recommendations
- Side-by-side performance metrics comparison
- ROC curve overlay for both models
- Discussion of model trade-offs (interpretability vs. accuracy)
- Clinical recommendations based on findings
- Study limitations and future directions

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dennisous/cmse830_fds.git
cd cmse830_fds
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook to generate required CSV files:
```bash
jupyter notebook heart_disease_analysis.ipynb
```

This will create the needed files in data folder:
- `heart_disease_original.csv`
- `heart_disease_simple_imputation.csv`
- `heart_disease_knn_imputation.csv`

4. Launch the Streamlit app:
```bash
streamlit run heart_disease_streamlit.py
```

## Data Sources

| Source | Description | Link |
|--------|-------------|------|
| UCI Heart Disease | Cleveland & Switzerland patient databases | [UCI Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Mendeley Data | Multispeciality Indian Hospital cardiovascular data | [Mendeley Data](https://data.mendeley.com/datasets/dzz48mvjht/1) |

### Citation
Doppala, Bhanu Prakash; Bhattacharyya, Debnath (2021), "Cardiovascular_Disease_Dataset", Mendeley Data, V1, doi: 10.17632/dzz48mvjht.1

## Project Structure

```
cmse830_fds/
├── heart_disease_analysis.ipynb       # Data preprocessing, EDA, and model development
├── heart_disease_streamlit.py         # Interactive Streamlit application
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── data/
    ├── heart_disease_original.csv              # Combined dataset with missing values
    ├── heart_disease_simple_imputation.csv     # Simple (median/mode) imputed dataset
    ├── heart_disease_knn_imputation.csv        # KNN imputed dataset
    ├── switzerland_dataset.csv                 # Raw Switzerland data
    └── multispeciality_hospital_india.csv      # Raw Indian hospital data
```

## Key Findings

### Most Important Risk Factors (from Random Forest Feature Importance)
1. **Chest Pain Type** - Asymptomatic chest pain strongly associated with heart disease
2. **ST Segment Characteristics** - Oldpeak and slope during exercise
3. **Number of Major Vessels** - Fluoroscopy results indicating blockages
4. **Maximum Heart Rate** - Lower max HR correlates with higher risk
5. **Exercise-Induced Angina** - Presence indicates elevated risk

### Model Performance
- Both Logistic Regression and Random Forest achieve strong predictive performance
- Logistic Regression preferred for interpretability and clinical decision support
- Random Forest captures non-linear patterns and feature interactions

## Technologies Used

- **Streamlit** - Interactive web application framework
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning (Random Forest, GridSearchCV, metrics)
- **Statsmodels** - Statistical modeling (Logistic Regression with inference)
- **SciPy** - Statistical computations

## Streamlit Cloud Deployment

The app is deployed on Streamlit Cloud. To deploy your own version:

1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Select your forked repository
4. Set main file path to `heart_disease_streamlit.py`
5. Deploy!

## Author

**Dennis Ous**  
Michigan State University  
CMSE 830 - Foundations of Data Science  
Fall 2024

## License

This project is for educational purposes as part of the MSU CMSE 830 course.
