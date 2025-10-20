# Heart Disease Risk Factor Analysis

Interactive data analysis application exploring risk factors for heart disease using the UCI Heart Disease Dataset.

## Overview
This project analyzes cardiovascular health data from Cleveland and Switzerland patient databases to identify key risk factors contributing to heart disease. The application hosts exploratory data analysis, missing data handling strategies, imputation strategies and limitations of them, and correlation analysis across different imputation methods.

## Target Audience
- Individuals ages 25-70 concerned about heart disease risk
- Healthcare providers analyzing cardiovascular risk factors
- Medical researchers studying heart disease patterns

## Features
- Interactive data exploration with age-based filtering
- Univariate and bivariate analysis visualizations
- Missing data pattern analysis (MNAR assessment)
- Comparison of Simple and KNN imputation methods
- Correlation heatmaps for numeric and categorical features

## Setup Instructions

### Prerequisites
- Python 3.7+
- Required packages: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `ucimlrepo`

### Installation
1. Clone the repository:
```bash
git clone https://github.com/dennisous/cmse830_fds.git
cd cmse830_fds
```

2. Install dependencies:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn ucimlrepo
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
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

## Project Structure
```
cmse830_fds/
├── heart_disease_analysis.ipynb       # Data preprocessing and analysis
├── heart_disease_streamlit.py         # Interactive Streamlit application
├── README.md
├──Data
    ├──heart_disease_original.csv         # Original dataset with missing values
    ├──heart_disease_simple_imputation.csv
    ├──heart_disease_knn_imputation.csv       
    
```

## Author
Dennis Ous - MSU CMSE 830 Project
