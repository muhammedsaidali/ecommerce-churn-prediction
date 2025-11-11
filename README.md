# 🛒 Ecommerce Customer Churn Prediction

> A Machine Learning–powered Streamlit web app that predicts whether an e-commerce customer is likely to churn or stay — based on behavioral, transactional, and engagement data.

![App Screenshot](images/app_preview.png)

---

## 🌟 **Overview**

Customer churn directly impacts business revenue and growth.  
This project builds an intelligent system that:
- Predicts the probability of customer churn
- Helps businesses **retain valuable customers**
- Offers actionable insights into **key churn-driving factors**

---

## 🧠 **Project Workflow**

1. **Data Collection & Understanding**
   - Realistic e-commerce dataset with customer-level details:
     `customer_id`, `signup_date`, `last_purchase_date`, `total_orders`,  
     `sessions`, `complaints`, `satisfaction_score`, `preferred_category`, `membership_type`, `region`, etc.
   - Target variable: `churn` (1 = customer churned, 0 = active)

2. **Data Cleaning**
   - Removed duplicates and handled missing values (numeric imputed with medians)
   - Encoded categorical fields (gender, membership type, preferred category, region)
   - Parsed date columns and created **tenure** and **recency** features

3. **Feature Engineering (RFM-inspired)**
   - Derived key behavioral metrics:
     - `spend_per_order = total_spent / total_orders`
     - `activity_ratio = sessions / tenure_days`
     - `complaints_per_order = complaints / total_orders`
     - `avg_satisfaction_recent = satisfaction_score / (recency_days + 1)`
   - Scaled and aligned features using training medians (stored in `train_stats.pkl`)

4. **Outlier Handling**
   - Winsorized heavy-tailed variables (like `sessions` and `total_spent`) at 99th percentile  
   - Ensured robust learning and stable feature distributions

5. **Model Training**
   - Algorithm: **Random Forest Classifier** (selected for its balance of performance and interpretability)
   - Framework: Scikit-learn pipeline with SMOTE for class imbalance
   - Evaluation metrics:
     - **Precision**, **Recall**, **F1-score**
     - **ROC-AUC** and **PR-AUC** (for imbalanced data)

6. **Cross Validation**
   - Used **StratifiedKFold (5-fold)** cross-validation  
   - Ensured balanced class representation across folds  
   - Tracked variance in recall and precision for model robustness

7. **Threshold Optimization**
   - Computed optimal **decision threshold (0.375)** for best F1-score  
   - Stored threshold in metadata (`train_stats.pkl`) for consistent inference  
   - Example:
     ```
     Best-F1 threshold: 0.375
     Precision=0.47, Recall=0.85, F1=0.60
     ```

8. **Deployment (Streamlit App)**
   - Real-time interactive UI built with Streamlit  
   - Accepts manual customer input (transactional & demographic data)  
   - Displays:
     - Churn Probability
     - Classification (“High Risk” / “Likely to Stay”)
     - Top feature importances  
     - Actionable recommendations for retention

---

## 🧾 **Model Artifacts**

| File | Description |
|------|--------------|
| `rf_final_model.pkl` | Trained Random Forest pipeline (SMOTE + RF) |
| `train_stats.pkl` | Stores feature list, training medians, and best threshold |
| `app.py` | Streamlit web app for real-time predictions |
| `requirements.txt` | Python dependencies for deployment |
| `README.md` | Project documentation |
| `venv/` | Virtual environment (not required for GitHub upload) |

---

## ⚙️ **How to Run Locally**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/muhammedsaidali/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction
