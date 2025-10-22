# 🩺 Obesity Risk Prediction

## 📘 Overview

This project predicts an individual's **risk of obesity** based on
demographic, dietary, and lifestyle factors using machine learning
models. It provides healthcare practitioners and researchers with
insights for prevention and awareness campaigns.

## 🎯 Business Objective

To enable **early intervention** by identifying individuals at higher
risk of obesity, allowing targeted wellness and nutrition programs to
reduce obesity-related diseases.

------------------------------------------------------------------------
# 🩺 Obesity Risk Prediction

This repository presents a complete **Machine Learning and Business Intelligence project** designed to predict **obesity risk levels** based on lifestyle, diet, and physical activity data. It integrates **data analysis, model building, and Power BI dashboard visualization** for decision-making in healthcare analytics.

---

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
  **Dashboard Section**    **Description**   **Example Visualization**
  ------------------------ ----------------- -----------------------------
  **Obesity Risk by        Compare risk      Bar chart, stacked column
  Demographics**           levels by gender, 
                           age, and income   
                           group             

  **BMI Distribution &     Show BMI patterns Histogram, boxplot
  Clusters**               segmented by risk 
                           group             

  **Physical Activity      Correlation       Scatterplot
  vs. Obesity Risk**       between exercise  
                           hours and         
                           predicted risk    

  **Diet Type Analysis**   Impact of         Pie chart
                           vegetarian,       
                           balanced, and     
                           fast-food diets   
                           on obesity        
                           probability       

  **Sleep Hours & Risk     Relationship      Line chart
  Index**                  between sleep     
                           quality and       
                           obesity tendency  

  **Feature Importance**   Top contributing  Bar chart from model
                           variables (e.g.,  
                           calorie intake,   
                           activity, age)    

  **Geographic             Map of obesity    Filled map
  Distribution             risk scores by    
  (optional)**             region or ZIP     
                           code              

  **Prediction Summary**   Total             KPI cards
                           predictions,      
                           high-risk count,  
                           average           
                           probability       
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Tech Stack

**Languages & Tools:** Python, SQL, Scikit-learn, Pandas, Power BI,
Streamlit, Airflow\
**Libraries:** NumPy, Matplotlib, Seaborn, XGBoost, Flask\
**Data Sources:** Public health surveys, fitness tracker data, nutrition
databases

------------------------------------------------------------------------

## 👤 Author

**Bahre Hailemariam**\
Data Analyst & BI Developer\
🔗 [LinkedIn](https://www.linkedin.com/) \|
[GitHub](https://github.com/)
