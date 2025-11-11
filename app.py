import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Ecommerce Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Sleek Dark Teal Gradient Aesthetic Theme (Unchanged - it's great!)
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* --- Primary App Background and Font --- */
.stApp {
    /* Sleek Dark Teal Gradient: Deep, professional look */
    background: linear-gradient(145deg, #102A43 0%, #243B55 70%, #0E2938 100%);
    color: #E0E7E9; /* Light text for dark background */
    font-family: 'Poppins', 'Segoe UI', sans-serif;
    font-weight: 400;
}

/* --- Refined Card Style --- */
.card {
    /* Slightly lighter dark background for cards with subtle teal border */
    background: #2D4158; 
    border-radius: 24px; 
    /* Professional blue/teal shadow */
    box-shadow: 0 10px 30px rgba(47, 65, 84, 0.4); 
    border: 1px solid #486581; 
    padding: 36px 30px;
    margin-bottom: 30px;
    transition: transform 0.3s ease-in-out;
}
.card:hover {
    transform: translateY(-3px); 
    box-shadow: 0 12px 35px rgba(47, 65, 84, 0.6); 
}

/* --- Typography --- */
h1, h2, h3, .section-title {
    color: #62B8D2; /* Bright Teal/Cyan accent */
    font-family: 'Poppins', sans-serif;
    font-weight: 700; 
    letter-spacing: 0.03em;
}

h1 {
    font-size: 3.2rem; 
    margin-bottom: 8px;
    color: #92CDDD; /* Lighter accent for the main title */
}

h2 {
    font-size: 2.2rem;
    margin-top: 0;
    margin-bottom: 15px;
}

h3 {
    font-size: 1.5rem;
    color: #A0DDE6;
    font-weight: 600;
}

/* Section Title Pillbox */
.section-title {
    font-size: 1.3rem;
    margin-bottom: 20px;
    background: #486581; /* Muted background pill */
    padding: 12px 25px;
    border-radius: 30px; 
    color: #E0E7E9;
    box-shadow: 0 2px 10px rgba(16, 42, 67, 0.6);
    display: inline-block;
    font-weight: 600;
    border: 1px solid #62B8D2;
}

/* Subtitle and Muted Text */
.subtitle {
    color: #CCD6E0; /* Slightly lighter body text */
    font-size: 1.15rem;
    font-weight: 400;
    margin-bottom: 25px;
}

.muted {
    color: #A9BCCF;
    font-size: 1rem;
    font-weight: 300; 
}

/* Data box style */
.data-box {
    background: #1E3144; /* Darker background for data tables/sections */
    border-radius: 18px;
    padding: 20px 25px;
    border: 1px solid #486581;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

/* --- Inputs and Buttons --- */

/* Buttons with vibrant coral/orange accent (high contrast) */
div.stButton > button {
    background-color: #FF7043; /* Vibrant Orange/Coral */
    color: white;
    border-radius: 16px;
    border: none;
    padding: 1em 2em;
    font-weight: 700;
    font-size: 1.15rem;
    transition: background 0.3s ease, transform 0.1s ease;
    box-shadow: 0 4px 15px rgba(255, 112, 67, 0.4);
}
div.stButton > button:hover {
    background-color: #E65100;
    transform: translateY(-1px);
}

/* Input labels */
[role="listitem"] label, .stNumberInput > label, .stSelectbox > label, .stSlider > label {
    color: #A0DDE6 !important; /* Bright Teal labels */
    font-weight: 600;
    font-family: 'Poppins', sans-serif;
    font-size: 1.05rem;
}

/* Input text fields */
input, select, .stNumberInput input {
    color: #E0E7E9 !important;
    font-weight: 600;
    background-color: #1E3144; /* Dark input background */
    border-radius: 8px;
    border: 1px solid #486581;
    padding: 8px 12px;
}

/* Prediction Result Box */
.prediction-box {
    text-align: center;
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}

/* Dark theme specific result colors */
.prediction-churn {
    background-color: #4D0000; /* Dark Red background */
    border: 2px solid #FF5252; /* Bright Red border */
}
.prediction-stay {
    background-color: #004D40; /* Dark Green background */
    border: 2px solid #69F0AE; /* Bright Green border */
}

/* Table styling for dark theme */
.stDataFrame {
    color: #E0E7E9;
    background-color: #1E3144;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<div class='card'>
    <h1>🛒 Ecommerce Churn Risk Analyzer</h1>
    <div class='subtitle'>
        Uncover potential customer churn before it happens. Input customer data below to get a real-time churn prediction.
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model + Metadata
# ----------------------------
MODEL_PATH = Path("rf_final_model.pkl")
STATS_PATH = Path("train_stats.pkl")

with st.container():
    if not MODEL_PATH.exists() or not STATS_PATH.exists():
        st.error("Model or metadata not found. Place `rf_final_model.pkl` and `train_stats.pkl` in the same folder.")
        st.stop()

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(STATS_PATH, "rb") as f:
            stats = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    expected_features = [c for c in stats.get("expected_features", []) if c != "review_sentiment"]
    train_medians = stats.get("train_medians", {})
    best_threshold = float(stats.get("best_threshold", 0.5))

    if not expected_features:
        st.error("Missing feature list in metadata. Ensure correct `train_stats.pkl` file.")
        st.stop()


# ----------------------------
# Helper functions
# ----------------------------
def collapse_duplicate_cols(df):
    if not df.columns.duplicated().any():
        return df
    cols = df.columns
    new = []
    for n in pd.unique(cols):
        group = df.loc[:, cols == n]
        merged = group.bfill(axis=1).iloc[:,0].rename(n)
        new.append(merged)
    return pd.concat(new, axis=1)


def prepare_input(inputs):
    row = {}
    num_fields = [
        "age","total_orders","avg_order_value","total_spent","sessions","complaints","satisfaction_score"
    ]
    for f in num_fields:
        row[f] = float(inputs.get(f, train_medians.get(f, 0.0)))
    row["total_spent_calc"] = row["total_spent"]
    row["tenure_days"] = float(inputs.get("tenure_days", train_medians.get("tenure_days", 0.0)))
    row["recency_days"] = float(inputs.get("recency_days", train_medians.get("recency_days", 0.0)))

    tot = max(row["total_orders"], 1e-6)
    sess = max(row["sessions"], 1e-6)
    row["spend_per_order"] = row["total_spent"] / tot
    row["orders_per_session"] = row["total_orders"] / sess
    row["complaints_per_order"] = row["complaints"] / tot
    row["spend_per_day"] = row["total_spent"] / (row["tenure_days"] + 1)
    row["activity_ratio"] = row["sessions"] / (row["tenure_days"] + 1)
    row["avg_satisfaction_recent"] = row["satisfaction_score"] / (row["recency_days"] + 1)

    dummies = [
        "gender_Male","membership_type_Gold","membership_type_Silver",
        "preferred_category_Beauty","preferred_category_Books","preferred_category_Electronics",
        "preferred_category_Fashion","preferred_category_Grocery","preferred_category_Health",
        "preferred_category_Home","preferred_category_Sports","preferred_category_Toys"
    ]
    for d in dummies: row[d] = 0
    if inputs.get("gender") == "Male": row["gender_Male"] = 1
    mem = inputs.get("membership_type")
    if mem == "Gold": row["membership_type_Gold"] = 1
    if mem == "Silver": row["membership_type_Silver"] = 1
    pref = f"preferred_category_{inputs.get('preferred_category')}"
    if pref in row: row[pref] = 1
    row["region_encoded"] = float(inputs.get("region_encoded", train_medians.get("region_encoded", 0.0)))

    df = pd.DataFrame([row])
    df = collapse_duplicate_cols(df)

    for f in expected_features:
        if f not in df.columns: df[f] = train_medians.get(f, 0.0)

    extras = [c for c in df.columns if c not in expected_features]
    if extras: df = df.drop(columns=extras)

    return df[expected_features].astype(float).fillna(0.0)


# ----------------------------
# Input Form
# ----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>👤 Customer Input Parameters</div>", unsafe_allow_html=True)

with st.form("predict"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3>Transaction Metrics</h3>", unsafe_allow_html=True)
        total_orders = st.number_input("Total Orders (Lifetime)", 0, 10000, 6, help="Number of completed purchases.")
        total_spent = st.number_input("Total Spent (Lifetime, $)", 0.0, 1e6, 350.0, help="Total monetary value of all purchases.")
        avg_order_value = st.number_input("Average Order Value ($)", 0.0, 1e4, 60.0, help="Total spent divided by total orders.")
        sessions = st.number_input("Total Sessions/Visits", 0, 10000, 11, help="Total number of visits to the e-commerce site.")
        
        st.markdown("<br><h3>Account Details</h3>", unsafe_allow_html=True)
        tenure_days = st.number_input("Tenure (days)", 0, 5000, 900, help="How long the customer has been registered.")
        age = st.number_input("Age", 18, 100, 35)
        gender = st.selectbox("Gender", ["Female", "Male"]) 
        
    with col2:
        st.markdown("<h3>Behavioral Metrics</h3>", unsafe_allow_html=True)
        recency_days = st.number_input("Recency (days)", 0, 5000, 30, help="Days since the last order. **Lower is better.**")
        satisfaction_score = st.slider("Satisfaction Score (1-10)", 0.0, 10.0, 7.5, 0.1, help="Overall customer satisfaction on a scale of 0 to 10.")
        complaints = st.number_input("Complaints Filed", 0, 1000, 0, help="Number of customer service complaints.")
        
        st.markdown("<br><h3>Categorical Preferences</h3>", unsafe_allow_html=True)
        membership_type = st.selectbox("Membership Tier", ["Free","Silver","Gold","Platinum"])
        preferred_category = st.selectbox("Preferred Category", [
             "Fashion","Electronics","Beauty","Books","Grocery","Home","Health","Sports","Toys"
        ])
        region_encoded = st.number_input("Region Index", value=float(train_medians.get("region_encoded", 0.0)), help="Encoded regional feature from the training data.")

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Run Churn Prediction")
st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Prediction Result
# ----------------------------
if submitted:
    inputs = dict(
        age=age,total_orders=total_orders,avg_order_value=avg_order_value,
        total_spent=total_spent,sessions=sessions,complaints=complaints,
        satisfaction_score=satisfaction_score,tenure_days=tenure_days,recency_days=recency_days,
        gender=gender,membership_type=membership_type,preferred_category=preferred_category,
        region_encoded=region_encoded
    )
    
    try:
        X = prepare_input(inputs)
        proba = model.predict_proba(X)[:,1][0]
        pred = int(proba >= best_threshold)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()
        
    st.markdown("---")
    
    color_churn = "#FF5252" # Bright Red accent for dark theme
    color_stay = "#69F0AE"  # Bright Green accent for dark theme
    
    if pred == 1:
        color = color_churn
        text = "HIGH CHURN RISK"
        emoji = "⚠️"
        box_class = "prediction-churn"
        action_text = "Immediate action is advised. Focus on re-engagement strategies like personalized offers or direct customer service outreach."
    else:
        color = color_stay
        text = "LOW CHURN RISK (STAY)"
        emoji = "✅"
        box_class = "prediction-stay"
        action_text = "Customer is likely to remain. Focus on retention through loyalty programs and positive service reinforcement."

    st.markdown(f"""
    <div class='prediction-box {box_class}'>
        <h2 style='color:{color}; font-size: 2.5rem; margin-bottom: 5px;'>{emoji} {text}</h2>
        <div class='subtitle' style='color:{color};'>
            Probability of Churn = <b>{proba:.1%}</b>
        </div>
        <p class='muted' style='font-size: 1.1rem; color: #A9BCCF;'>
            (Model Threshold for Churn is {best_threshold:.3f})
        </p>
        <p style='color: #E0E7E9; font-weight: 600;'>
            💡 **Recommendation:** {action_text}
        </p>
    </div>
    """, unsafe_allow_html=True)


    # --- Prediction Details ---
    st.markdown("<div class='data-box'><h3>📊 Prediction Analysis and Features</h3>", unsafe_allow_html=True)
    
    # Adding a prompt for a diagram that adds instructive value
    st.markdown("""
    Understanding where the customer falls in the **Churn Funnel** helps with intervention strategies. 
    

[Image of Customer Churn Funnel Stages]

    """, unsafe_allow_html=True)
    
    st.markdown("<h4>Input Features Used for Prediction:</h4>", unsafe_allow_html=True)
    show = ["age","total_orders","sessions","avg_order_value","total_spent","satisfaction_score","tenure_days","recency_days","region_encoded"]
    df_small = X[show].T.rename(columns={0:"Value"}).reset_index().rename(columns={"index":"Feature"})
    st.dataframe(df_small, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    with st.expander("🔬 Top 6 Feature Importances (What Drove the Prediction)"):
        try:
            est = model
            if hasattr(model,"named_steps"):
                est = model.named_steps[list(model.named_steps.keys())[-1]]
            
            if hasattr(est,"feature_importances_"):
                fi = np.array(est.feature_importances_)
                feat = np.array(X.columns)
                idx = np.argsort(fi)[::-1][:6]
                
                # Plot setup
                fig, ax = plt.subplots(figsize=(7,3.5))
                ax.barh(feat[idx][::-1], fi[idx][::-1], color="#62B8D2", alpha=0.9, height=0.7)
                ax.set_xlabel("Feature Importance Score", fontsize=10, color='#A9BCCF')
                ax.set_title("Top 6 Influencing Features", fontsize=12, color='#E0E7E9')
                
                # Set axes/tick colors for dark mode
                ax.tick_params(axis='y', labelsize=10, colors='#A9BCCF')
                ax.tick_params(axis='x', labelsize=9, colors='#A9BCCF')
                ax.set_facecolor("#1E3144")
                fig.patch.set_facecolor('#1E3144')
                
                ax.grid(axis='x', linestyle='--', alpha=0.3, color='#486581')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("Feature importances are not available for this model type.")
        except Exception as e:
            st.write("Could not compute importances:", e)
    
    st.markdown("</div>", unsafe_allow_html=True) # Close data-box


    # --- Project Overview Card (Simplified and user-friendly) ---
    st.markdown("""
    <div class='card'>
        <div class='section-title'>⚙️ Project Overview: Model Methodology 🔬</div>
        <div class='muted'>
            <p style='font-weight: 600; color: #92CDDD; font-size: 1.1rem; margin-bottom: 20px;'>
            This model was built to be accurate, robust, and explainable. Here is the simplified methodology:
            </p>
            <ol style='color: #CCD6E0; padding-left: 20px;'>
                <li>
                    <b>Data Preprocessing & Cleaning 🧹:</b> We cleaned raw data by filling missing values (like **age** or **satisfaction score**) using median values from the training set to prevent skew.
                </li>
                <li>
                    <b>Feature Engineering 💡:</b> We created powerful predictive features from raw inputs, such as **Spend Per Order**, **Activity Ratio** (sessions per day), and **Complaints Per Order**. We also handled categorical data (like membership type) with encoding.
                </li>
                <li>
                    <b>Handling Outliers & Scale 📊:</b> We used a technique called **Winsorization** to safely cap extreme values (like *Total Spent*) to prevent them from negatively impacting the model.
                </li>
                <li>
                    <b>Modeling & Imbalance Strategy 🤖:</b> The final model is a **Random Forest Classifier**. Crucially, we used **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learned how to identify rare churn cases effectively.
                </li>
                <li>
                    <b>Evaluation & Deployment Threshold 🎯:</b> Instead of the default 50% probability, we optimized the model's decision threshold to maximize the **F1-Score** (balancing precision and recall for churn), which is saved and used for the prediction shown above.
                </li>
                <li>
                    <b>Artifacts for Production 📦:</b> The fully trained model (`rf_final_model.pkl`) and all necessary transformation statistics (`train_stats.pkl`) were saved to ensure the app runs predictions exactly as they were trained.
                </li>
            </ol>
            <p class='muted' style='margin-top: 25px; font-style: italic;'>
            Summary: We transform raw customer data into powerful, engineered features, train a specialized Random Forest model to handle rare churn events, and use a business-optimized threshold for clear decision-making.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)