# Heart Disease Analysis - Technical Documentation

## Data Dictionary

### Dataset Overview
- **Sources**: Cleveland Clinic (USA), Switzerland Hospital, Multispeciality Indian Hospital
- **Total Observations**: ~700 patients (after combining all sources)
- **Target Variable**: Binary classification (0 = No heart disease, 1 = Heart disease present)

### Feature Definitions

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `age` | Numeric | Patient age in years | 29-77 |
| `gender` | Categorical | Biological sex | 0 = Female, 1 = Male |
| `chestpain` | Categorical | Chest pain type | 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal, 4 = Asymptomatic |
| `restingBP` | Numeric | Resting blood pressure (mm Hg) | 94-200 |
| `serumcholestrol` | Numeric | Serum cholesterol (mg/dl) | 126-564 |
| `fastingbloodsugar` | Binary | Fasting blood sugar > 120 mg/dl | 0 = No, 1 = Yes |
| `restingrelectro` | Categorical | Resting ECG results | 0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy |
| `maxheartrate` | Numeric | Maximum heart rate achieved | 71-202 |
| `exerciseangia` | Binary | Exercise-induced angina | 0 = No, 1 = Yes |
| `oldpeak` | Numeric | ST depression induced by exercise (mm) | 0-6.2 |
| `slope` | Categorical | Slope of peak exercise ST segment | 1 = Upsloping, 2 = Flat, 3 = Downsloping |
| `noofmajorvessels` | Categorical | Major vessels colored by fluoroscopy | 0-3 |
| `target` | Binary | Heart disease diagnosis | 0 = Absent, 1 = Present |

### Engineered Features (MNAR Indicators)
| Variable | Description |
|----------|-------------|
| `serumcholestrol_missing` | 1 if cholesterol was not measured |
| `noofmajorvessels_missing` | 1 if fluoroscopy not performed |
| `fastingbloodsugar_missing` | 1 if fasting blood sugar not recorded |
| `slope_missing` | 1 if stress test slope not recorded |

---

## Modeling Approach

### Data Preprocessing
1. **Dataset Integration**: Combined three hospital datasets with column name standardization
2. **Encoding Alignment**: Remapped Indian Hospital categorical encodings to match UCI format
3. **Missing Data Handling**: Created MNAR indicator variables; used complete case analysis for modeling
4. **Target Encoding**: Converted multi-class (0-4) to binary (0 vs 1-4)
5. **One-Hot Encoding**: Applied to categorical variables with `drop_first=True`

### Model 1: Logistic Regression
- **Library**: `statsmodels.api.Logit`
- **Purpose**: Interpretable model with statistical inference (p-values, confidence intervals)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Output**: Odds ratios for clinical interpretation

### Model 2: Random Forest Classifier
- **Library**: `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Parameter Grid**:
  - `n_estimators`: [100, 200, 300]
  - `max_features`: ['sqrt', 0.5]
  - `max_depth`: [10, 20, None]
  - `min_samples_split`: [2, 5]
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Output**: Gini-based feature importance scores

### Train-Test Split
- **Split Ratio**: 80% training, 20% testing
- **Random State**: 42 (reproducibility)
- **Stratification**: Shuffle enabled

### Model Selection Criteria
- **Interpretability**: Logistic Regression preferred for clinical settings
- **Predictive Power**: Random Forest captures non-linear relationships
- **Final Recommendation**: Context-dependent; both models provide complementary insights

---

## File Descriptions

| File | Purpose |
|------|---------|
| `heart_disease_analysis.ipynb` | Full analysis pipeline: data loading, EDA, preprocessing, modeling |
| `heart_disease_streamlit.py` | Interactive web application with all visualizations and models |
| `heart_disease_original.csv` | Combined raw data with missing values intact |
| `heart_disease_simple_imputation.csv` | Data after median/mode imputation |
| `heart_disease_knn_imputation.csv` | Data after KNN (K=5) imputation |

---

*Documentation prepared for CMSE 830 - Foundations of Data Science, Michigan State University*
