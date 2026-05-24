# 🏃‍♂️Predict User Churn on Digital Health Platforms

Capstone Project - Professional Certificate in Data Analytics (Imperial College Business School)
Author: Lilian Gomez
🚀 Live Project Site: [Link to your GitHub repository or pages]

## 🧠 Problem Statement
Given physical workout logs and user metadata from the Endomondo ecosystem (behavioral, biological & spatial attributes), predict whether a platform user will abandon the app (**Churn** = 1) or remain actively retained (**Retained** = 0). 

Accurate, predictive classifications allow digital health networks to:
* Deploy automated push notifications and adaptive training milestones.
* Target marketing resources to users showing structural habit decay before they delete their accounts.
* Maximize platform subscription customer lifetime value (LTV).

The project evaluates submissions using the **Unified F1-Score** and **Classification Accuracy** to safely navigate background class imbalances.

## 📂 Dataset

| File | Description |
| :--- | :--- |
| `data/endomondo_user_features_2015.csv` | Processed, user-level feature matrix consisting of 1,059 unique user behavior profiles. |
| `notebooks/1.0_data_cleaning_and_audit.ipynb` | Pipeline auditing, handling missing sensor data, dropping corrupted features. |
| `notebooks/2.0_user_feature_engineering.ipynb` | Compressing raw time-series workout logs into unique user profiles. |
| `notebooks/3.0_model_training_and_evaluation.ipynb` | 3-way data partitioning, hyperparameter tuning, and final evaluation. |

*Note: The primary dataset is derived from sequential historical wearable tracking profiles containing exercise frequencies, cardiovascular markers, and GPS spatial features.*

## 🔍 Approach Overview

1. **Exploration & Profiling:** Investigated data missingness, target class balance (~61.7% churn vs 38.3% retained), and feature cardinality.
2. **Feature Handling:**
   * Audited and dropped the `avg_speed` column entirely due to massive satellite tracking gaps.
   * Executed median imputation across remaining continuous biological and temporal attributes.
   * Compiled user historical logs into 7 engineered summary features (`total_workouts`, `total_active_minutes`, `avg_workout_duration`, `overall_avg_hr`, `sport_diversity`, `home_lat`, `home_lon`).
3. **Data Partitioning (3-Way Split):** Enforced a strict 70% Train / 15% Validation / 15% Test stratified partition split to establish complete data isolation and eliminate leakage.
4. **Baseline Anchor:** Deployed a structural Dummy Classifier (predicting the majority class) to anchor prediction floors.
5. **Models Evaluated & Tuned:**
   * Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, and XGBoost.
   * Automated hyperparameter optimization executed via `GridSearchCV` on the validation split.
6. **Model Selection:** The **Tuned Random Forest Classifier** achieved the optimal balance of Precision and Recall on the validation data.
7. **Final Exam:** Evaluated the fully optimized Random Forest champion against the locked, untouched Test Set.

## 📊 Key Insights

* **Volume Over Variety:** Consistency completely dominates predictive importance. `total_workouts` and `total_active_minutes` provide the loudest signal (~45% predictive weight).
* **Geographic Clusters:** Spatial features (`home_lat` and `home_lon`) emerged as robust secondary signals, indicating regional patterns in app abandonment.
* **The Dummy Trap:** While a naive baseline scores high Recall due to the background class imbalance, it provides zero predictive intelligence. The Tuned Random Forest successfully cuts out wasted outreach by lifting classification accuracy to **72.0%** and reaching an **80.0% Churn F1-Score** on the unseen final test group.

### Final Performance Leaderboard (Validation Set)
| Model Architecture | Validation Accuracy | Validation Precision | Validation Recall | Validation F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **0.704403** | **0.725664** | **0.836735** | **0.777251** |
| XGBoost (Tuned) | 0.666667 | 0.677165 | 0.877551 | 0.764444 |
| Dummy Baseline | 0.616352 | 0.616352 | 1.000000 | 0.762646 |
| Decision Tree (Tuned) | 0.691824 | 0.733333 | 0.785714 | 0.758621 |
| Support Vector Machine (Tuned) | 0.666667 | 0.699115 | 0.806122 | 0.748815 |
| Logistic Regression (Tuned) | 0.666667 | 0.702703 | 0.795918 | 0.746411 |
| K-Nearest Neighbors (Tuned) | 0.672956 | 0.734694 | 0.734694 | 0.734694 |

### Visual Insights
![Confusion Matrix Heatmap](output/confusion_matrix.png)
![Feature Importance Chart](output/feature_importance.png)

## 🛠️ Tech Stack / Dependencies

* Core Python (≥3.9 recommended)
* `pandas` & `numpy` (Data consolidation & wrangling)
* `scikit-learn` (Scaling, 3-way partition splitting, GridSearchCV, and model suite evaluation)
* `matplotlib` & `seaborn` (Visual performance matrices & bar charts)
* `xgboost` (Advanced gradient tree boosting)

