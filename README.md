# Credit Scoring Custom Model App

## 📌 Overview
The **Credit Scoring Custom Model App** is a **Streamlit-based interactive application** designed to build a **custom credit scoring model** using **logistic regression with regularization techniques**. The app employs **Palencia-based binning**, **feature selection**, and **automated model tuning** to ensure the best predictive performance.

This project automates the **data preprocessing, feature engineering, model training, and scoring process**, allowing users to:
- Upload their dataset (CSV/Excel)
- Perform **data cleaning & preprocessing**
- Generate **WOE binning & feature selection**
- Encode categorical variables for modeling
- Optimize hyperparameters via **grid search**
- Train a **logistic regression model**
- Generate **scorecards & performance metrics**
- Export results to **Excel** for further analysis

---

## 🛠️ Project Modules
The project consists of multiple **Python scripts**, each handling different aspects of the **credit scoring pipeline**.

### **1️⃣ `main.py` - Streamlit App Interface**
This is the **main entry point** for the **interactive UI**, built using **Streamlit**. It allows users to:

- **Upload dataset** and set key parameters
- **Perform feature selection** using IV values
- **Apply optimal binning techniques**
- **Encode categorical variables**
- **Build and tune a logistic regression model**
- **Generate credit scorecards & performance projections**
- **Download final scorecards & reports**

🔹 **User Inputs:** Project Name, Target Variable, Dataset Upload, Model Hyperparameters
🔹 **Outputs:** Model Summary, Score Distribution, Evaluation Metrics, Downloadable Excel Report

---

### **2️⃣ `preprocessing.py` - Data Cleaning & Preparation**
This module handles **initial data preprocessing**:
- Filters out **irrelevant columns** (e.g., IDs, personal details, application numbers)
- Removes **sparse features** (above a defined missing rate threshold)
- Splits **numerical and categorical variables**
- Computes **initial feature importance using Information Value (IV)**
- Generates **predictor logic** for defining monotonic trends (ascending/descending)

🔹 **Key Functions:**
- `initial_filtering()` - Removes unnecessary & sparse features
- `num_cat_split()` - Splits numerical & categorical variables
- `calc_iv()` - Computes Information Value (IV) for each predictor

---

### **3️⃣ `binning.py` - Feature Binning & Selection**
This module applies **Palencia-based binning** to optimize predictor selection:
- Uses **OptimalBinning** to find the best cut-offs for numerical variables
- Selects **categorical features** based on their IV scores
- Defines **monotonic binning trends** (ascending/descending)
- Converts numerical variables into **binned categories**

🔹 **Key Functions:**
- `feature_selection_palencia()` - Selects features based on IV
- `merging_for_model()` - Bins features into optimal categories

---

### **4️⃣ `encoder.py` - Categorical Feature Encoding**
This module **converts categorical features into numerical representations**:
- Uses **one-hot encoding** to create dummy variables
- Ensures the dataset remains **structured & formatted correctly** for modeling

🔹 **Key Function:**
- `encoder()` - Transforms categorical variables into dummy variables

---

### **5️⃣ `correlation.py` - Feature Correlation Analysis**
This module **removes highly correlated predictors** to prevent multicollinearity:
- Generates a **correlation matrix**
- Eliminates variables with **high mutual correlation (> threshold)**
- Displays a **heatmap visualization** of correlations

🔹 **Key Function:**
- `filtering()` - Filters out highly correlated features

---

### **6️⃣ `model.py` - Model Training & Hyperparameter Tuning**
This module builds a **logistic regression model with L1/L2 regularization**:
- Splits data into **training & test sets**
- Performs **Grid Search** to find the best hyperparameters
- Selects the model with the **best KS-score**
- Plots **ROC curves** for model evaluation

🔹 **Key Function:**
- `build()` - Trains & tunes a logistic regression model

---

### **7️⃣ `scoring.py` - Scorecard Generation**
This module **applies the trained model** to compute credit scores:
- Calculates **logit, odds, and probability scores**
- Computes **score distributions & cut-off points**
- Generates **Kolmogorov-Smirnov (KS) plots**
- Evaluates model performance using **ROC & AUC scores**
- Outputs **final scorecard & performance projection table (PPT)**

🔹 **Key Function:**
- `scoring()` - Computes credit scores & KS/AUC metrics

---

### **8️⃣ `scoring_custom_model.py` - Streamlit App for Custom Scoring**
This script extends `main.py` with a **customizable model scoring pipeline**:
- Allows **custom feature selection & tuning**
- Applies **custom binning & predictor selection logic**
- Provides **real-time performance tracking** via `stqdm`

🔹 **Key Function:**
- `scoring()` - Computes credit scores & scorecard performance

---

### **9️⃣ `eva.py` - Model Evaluation Metrics**
This module provides **performance evaluation functions**:
- Computes **Kolmogorov-Smirnov (KS) metrics**
- Plots **Lift & ROC curves**

🔹 **Key Functions:**
- `eva_dfkslift()` - Computes KS & Lift metrics
- `eva_pks()` - Plots KS Test results

---

### **🔟 `scorecard_ppt.py` - Excel Report Generation**
This module exports **model results** into an Excel file with:
- **Scorecard** (feature scores)
- **Performance Projection Table (PPT)** (approval rates, odds, etc.)
- **Missing Rate Summary**
- **IV Analysis**
- **Feature Binning Statistics**

🔹 **Key Functions:**
- `create()` - Generates Excel file with scorecard & PPT
- `download()` - Enables file download via Streamlit

---

## 📈 Model Evaluation Metrics
The app provides the following **model performance metrics**:
- **Kolmogorov-Smirnov (KS) Score** - Measures separation between good & bad applicants.
- **AUC-ROC (Area Under the Curve - Receiver Operating Characteristic)** - Evaluates predictive power.
- **Gini Coefficient** - Measures inequality of score distributions.
- **Lift Charts** - Shows effectiveness of credit risk segmentation.

---

## 📥 Output & Reports
- 📊 **Scorecard & Model Report** (`.xlsx` file)
- 📈 **ROC Curve & Score Distribution Plots**
- ✅ **Optimal Cutoff Scores & Performance Metrics**

---

## 🚀 How to Run the App
1️⃣ **Install dependencies:**  
```sh
pip install -r requirements.txt
```

2️⃣ **Run the Streamlit app:**  
```sh
streamlit run main.py
```

3️⃣ **Upload a dataset** & adjust model parameters via the **UI**.
4️⃣ **Train the model, view evaluation metrics, and download results**.

---

## 🏆 Conclusion
This project provides a **powerful end-to-end solution** for **building, evaluating, and exporting credit scoring models**. It enables **customizable feature selection, model tuning, and scorecard generation** while maintaining **model interpretability** and **business relevance**.

💡 **Next Steps:**
- Enhance with **XGBoost/Random Forest models** for better prediction.
- Implement **automated feature selection** for improved interpretability.
- Add **real-time API integration** for credit risk assessment.

---

🎯 **Built with:** Streamlit, Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib, OptBinning

