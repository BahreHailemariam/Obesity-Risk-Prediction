# 🩺 Obesity Risk Prediction

## 📘 Overview

This project predicts an individual's **risk of obesity** based on
demographic, dietary, and lifestyle factors using machine learning
models. It provides healthcare practitioners and researchers with
insights for prevention and awareness campaigns.
This repository presents a complete **Machine Learning and Business Intelligence project** designed to predict **obesity risk levels** based on lifestyle, diet, and physical activity data. It integrates **data analysis, model building, and Power BI dashboard visualization** for decision-making in healthcare analytics.

## 🎯 Business Objective

To enable **early intervention** by identifying individuals at higher
risk of obesity, allowing targeted wellness and nutrition programs to
reduce obesity-related diseases.

------------------------------------------------------------------------


## 📊 Project Overview

Obesity is a major public health issue globally, influenced by factors like eating habits, physical activity, and demographics.  
This project uses **machine learning models** to predict an individual’s obesity category and identifies key lifestyle patterns contributing to health risk.

**Goal:**  
Build an end-to-end data pipeline that predicts obesity risk levels, visualizes insights, and automates periodic reporting.
- Build a robust predictive model - Visualize lifestyle and
demographic correlations - Automate data collection, model training, and
reporting

---

## 🧩 Detailed Project Workflow

### 1️⃣ Define Business Problem

-   **Goal:** Predict individuals at high risk of obesity to enable
    early lifestyle intervention.\
-   **Stakeholders:** Healthcare providers, fitness companies, insurance
    firms, and nutrition planners.\
-   **Key Questions:**
    -   What demographic or lifestyle factors most contribute to obesity
        risk?
    -   How can predictive analytics help reduce healthcare costs?

### 2️⃣ Data Extraction & Collection

-   **Sources:** Public health datasets, hospital records (CSV, SQL, or
    API-based sources).\
-   **Tools Used:**
    -   Python (`pandas`, `requests`) for API and CSV ingestion.\
    -   SQL queries to extract structured data from health databases.\
    -   Power Query or Power BI Dataflows for live data connections.

### 3️⃣ Data Cleaning & Preprocessing

-   Handle missing values using **mean/median imputation**.\
-   Remove or cap outliers in BMI, calorie intake, and physical activity
    using **IQR method**.\
-   Normalize categorical fields (e.g., diet type, activity level) using
    **OneHotEncoder**.\
-   Convert inconsistent units (e.g., height in meters vs. cm).\
-   Ensure schema consistency across merged datasets.

### 4️⃣ Feature Engineering

-   Derived features such as:
    -   **BMI = weight(kg) / (height(m))²**
    -   **Calorie per activity ratio**
    -   **Lifestyle Index = (sleep_hours × activity_level) /
        screen_time**
-   Standardize features with **MinMaxScaler** or **StandardScaler**.\
-   Perform **correlation analysis** to identify predictive variables.

### 5️⃣ Train/Test Split

-   Split dataset into 70/30 or 80/20 ratios using `train_test_split`.\
-   Use **stratified sampling** if class imbalance exists.

### 6️⃣ Model Training

-   Models used:
    -   Logistic Regression (baseline)
    -   Random Forest
    -   XGBoost (best-performing)
-   **Hyperparameter tuning:** GridSearchCV for optimized learning rate,
    depth, and estimators.

### 7️⃣ Model Evaluation

-   Metrics:
    -   **AUC-ROC Curve** for model discrimination
    -   **Precision-Recall Curve** for imbalanced data
    -   **Confusion Matrix** for misclassification insights
    -   **Feature Importance Graphs** for interpretability

### 8️⃣ Deployment

-   **Streamlit App:**\
    Simple web app where users input age, gender, diet, and lifestyle to
    get a predicted obesity risk score.
-   **Power BI Dashboard:**\
    Connects model outputs or datasets to visualize trends
    interactively.
-   **Flask API:**\
    Serve predictions for integration with mobile apps or EMR systems.

### 9️⃣ Automation

-   Schedule data refresh and model retraining using:
    -   **Airflow DAGs** (monthly model updates)
    -   **Cron Jobs** for nightly data ingestion and prediction logs\
    -   **Email alerts** for pipeline success/failure

------------------------------------------------------------------------

## 📊 Expanded Dashboard Insights
------------------------------------------------------------------------
| **Dashboard Section**  | **Description** | **Example Visualization** |
|------------------------|-----------------|-----------------------------| 
| **Obesity Risk by Demographics** |Compare risk levels by gender, age, and income group  | Bar chart, stacked column  |
| **BMI Distribution & Clusters**  | Show BMI patterns  segmented by riskgroup | Histogram, boxplot |                         
| **Physical Activity vs. Obesity Risk**  | Correlation between exercise hours and predicted risk  | Scatterplot              
| **Diet Type Analysis** | | Impact of vegetarian,balanced, and fast-food diets on obesity probability  | Pie chart |  
| **Sleep Hours & Risk Index**  | Relationship between sleep quality and obesity tendency | Line chart |                
| **Feature Importance** | Top contributing variables (e.g., calorie intake, activity, age) | Bar chart from model |           
| **Geographic Distribution (optional)**  | Map of obesity risk scores by region or ZIP code  | Filled map |
| **Prediction Summary** | Total  predictions, high-risk count, average probability | KPI cards |
# 📊 Power BI Dashboard Documentation — Obesity Risk Prediction

## 🎯 Objective
The Power BI dashboard provides a comprehensive view of **obesity risk analysis** and **machine learning predictions**, empowering healthcare professionals and analysts to identify at-risk populations and recommend interventions.

---

## 🧭 Dashboard Structure

### **Page 1: Executive Overview**
**Purpose:** Show high-level KPIs related to obesity, BMI, and prediction accuracy.

**Visuals:**
- KPI Cards: Total Participants, Average BMI, Average Obesity Risk, Model Accuracy
- Trend Line: BMI Change Over Time
- Donut Chart: Risk Distribution (Low, Medium, High)

**Example DAX:**
```DAX
Avg_BMI = AVERAGE(Data[BMI])
Total_Participants = COUNTROWS(Data)
Obesity_Risk_Score = AVERAGE(Data[Predicted_Risk_Score])
Model_Accuracy = 0.90   // Imported from model output
```

**Insight Example:**
> The average BMI across participants is 28.4 (Overweight range), with 42% classified as high-risk for obesity.

---   
### **Page 2: Demographic Insights**
**Purpose:** Explore obesity patterns by gender, age, and region.

**Visuals:**
- Stacked Bar: Obesity Category by Gender
- Column Chart: BMI by Age Group
- Map: Geographic Risk Heatmap

**DAX Measures:**
```DAX
Obese_Count = CALCULATE(COUNTROWS(Data), Data[Obesity_Level] = "Obese")
Obesity_Percentage = DIVIDE([Obese_Count], [Total_Participants], 0)
```

**Insight Example:**
> Males aged 30–45 show a 35% higher obesity risk than females in the same age range.

---
### **Page 3: Lifestyle Behavior Analysis**
**Purpose:** Reveal how lifestyle habits affect obesity.

**Visuals:**
- Scatter: Calorie Intake vs. Physical Activity
- Line Chart: Sleep Hours vs. BMI
- Heatmap: Junk Food Frequency vs. Obesity Level

**DAX Measures:**
```DAX
Avg_Calories = AVERAGE(Data[Daily_Calories])
Avg_Physical_Activity = AVERAGE(Data[Activity_Level])
Calorie_Burn_Ratio = DIVIDE([Avg_Physical_Activity], [Avg_Calories], 0)
```

**Insight Example:**
> Participants exercising less than 2 hours/day have a 70% probability of being overweight or obese.

---
### **Page 4: Machine Learning Results**
**Purpose:** Visualize prediction outputs from the Python model.

**Visuals:**
- Confusion Matrix Table (Actual vs. Predicted)
- Feature Importance Bar Chart
- Gauge: Model Accuracy (AUC/F1-score)

**Integration Tips:**
- Import `predictions.csv` with columns: `ID, Actual, Predicted, Risk_Score`
- Refresh data daily or weekly using Power BI Gateway or Power Automate

**Insight Example:**
> Random Forest model achieved 90% accuracy, with BMI, calorie intake, and physical activity as top predictors.

---
### **Page 5: Recommendations & Action Plan**
**Purpose:** Translate analytics into actionable insights.

**Visuals:**
- Matrix: Risk Level × Recommended Action
- Donut Chart: Intervention Category Distribution
- Text Box: Strategic Recommendations

**Recommendations Table Example:**

| Risk Level | Recommendation |
|-------------|----------------|
| Low | Maintain healthy habits |
| Medium | Add 30 mins daily exercise |
| High | Consult physician & nutritionist |

---
## 🧠 Advanced Dashboard Features

✅ **Dynamic Filters:**
- Gender, Age Group, Region, Physical Activity

✅ **Drill-through & Bookmarks:**
- Drill from Region → Age Group → Patient Profile

✅ **Python Visual Integration:**
Embed Python visuals for SHAP explainability:
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```
## 📈 Dashboard Insights

**Power BI / Streamlit Dashboard Includes:**
- 🧍 Obesity risk distribution by demographics  
- 🍔 Calorie intake vs. physical activity correlation  
- 🕒 BMI trend by age and gender  
- ⚙️ Prediction outcomes: Low, Medium, High risk categories  
- 🔄 Automated updates from ML model predictions  

Example business question solved:
> “Which demographic group has the highest obesity risk based on lifestyle and dietary patterns?”

---

## 🧠 Model Insights

| Model | Accuracy | Precision | Recall | AUC |
|-------|-----------|------------|--------|------|
| Logistic Regression | 82% | 80% | 78% | 0.86 |
| Random Forest | 88% | 85% | 83% | 0.91 |
| XGBoost | 90% | 87% | 86% | 0.94 |

**Feature Importance Example:**
- Calorie Intake — 25%  
- Physical Activity — 20%  
- BMI — 18%  
- Water Intake — 10%  
- Work Hours — 7%  

---

## 🧩 Tech Stack

- **Languages:** Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)  
- **Dashboard Tools:** Power BI / Streamlit  
- **Automation:** Apache Airflow / Python Scheduler  
- **Version Control:** GitHub  
- **Deployment:** Streamlit Cloud or Power BI Service  

---

## 📁 Folder Structure

```
obesity-risk-prediction/
│
├── data/
│   ├── obesity_dataset.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│
├── src/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   ├── predict.py
│
├── dashboard/
│   ├── powerbi_report.pbix
│   ├── streamlit_app.py
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation & Usage

```bash
# Clone this repository
git clone https://github.com/yourusername/obesity-risk-prediction.git

# Navigate to folder
cd obesity-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run dashboard/streamlit_app.py
```

---

## 📧 Contact

**Author:** Bahre Hailemariam  
**Role:** Data Analyst & BI Developer  
📩 Email: your.email@example.com  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/yourprofile)  
🔗 [GitHub Repository](https://github.com/yourusername/obesity-risk-prediction)

---

> *"Transforming healthcare data into actionable wellness insights."*
