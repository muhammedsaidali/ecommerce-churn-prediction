# 🛒 Ecommerce Churn Prediction

## 📘 Project Overview
This project predicts whether a customer is likely to **churn** (stop purchasing) based on their engagement, satisfaction, and spending behavior.

Built using:
- **Python**
- **Scikit-Learn**
- **Streamlit**
- **Imbalanced-Learn (SMOTE)**

It provides a **real-time churn prediction app** with explainable feature importance and project methodology.

---

## ⚙️ Tech Stack
- **Language:** Python 3.x  
- **Libraries:** pandas, numpy, scikit-learn, imbalanced-learn, streamlit, matplotlib  
- **Model:** Random Forest Classifier  
- **Deployment:** Streamlit App  

---

## 📊 Model Details
| Metric | Score |
|---------|--------|
| Accuracy | 84% |
| F1-Score | 0.61 |
| ROC-AUC | 0.82 |
| Best Threshold | 0.375 |
| Cross-Validation (5-Fold) | F1 = 0.609 ± 0.006 |

✅ **Chosen Model:** Random Forest (with SMOTE)

---

## 🧩 Key Features Used
| Feature | Description |
|----------|-------------|
| `tenure_days` | Days since customer signup |
| `recency_days` | Days since last purchase |
| `total_spent` | Lifetime amount spent |
| `sessions` | Total visits or interactions |
| `satisfaction_score` | Average satisfaction rating |
| `complaints` | Total complaints filed |
| `activity_ratio` | Sessions per active day |
| `avg_satisfaction_recent` | Recent satisfaction weighted by recency |

---

## 🚀 Run Locally
To run this project on your system:

1️⃣ Clone the repository:
```bash
git clone https://github.com/<your-username>/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction
