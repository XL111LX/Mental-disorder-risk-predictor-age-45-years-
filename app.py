# ============================================================
# Mental Disorder Risk Predictor
# Streamlit 在线预测工具
# 适用人群：年龄 >= 45 岁
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ---- 1. 加载模型、scaler、变量名 ----
model  = joblib.load("logistic_model.pkl")
scaler = joblib.load("logistic_scaler.pkl")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# ---- 2. 页面配置 ----
st.set_page_config(
    page_title="Mental Disorder Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Mental Disorder Risk Predictor")
st.write("An interpretable online prediction tool for mental disorder risk based on the Logistic regression model.")

st.info("📌 **Applicable Population**: This prediction tool is designed for adults aged **>= 45 years**. It may not be applicable to younger populations.")

st.markdown("---")

# ---- 3. 侧边栏 ----
st.sidebar.header("📋 Patient Characteristics")
st.sidebar.warning("⚠️ For adults aged >= 45 years only.")

with st.sidebar.expander("📖 Variable Definitions", expanded=False):
    st.markdown("""
    **1. Genders of children**
    - 0 = All daughters
    - 1 = Has sons (with or without daughters)

    **2. Highest degree obtained by children**
    - 0 = All children have below-college degrees
    - 1 = At least one child attended college

    **3. Income level of children**
    - 0 = Low income: All children have annual income < 10,000
    - 1 = High income: At least one child has annual income >= 10,000

    **4. Visits with children**
    - 0 = Less than once a year or never
    - 1 = At least once a year

    **5. Life satisfaction**
    - 1 = Not at all satisfied
    - 2 = Not very satisfied
    - 3 = Somewhat satisfied
    - 4 = Very satisfied
    - 5 = Completely satisfied

    **6. Chronic disease**
    - 0 = No; 1 = Yes
    - Includes: hypertension, diabetes, heart disease, stroke, arthritis, cancer, dyslipidemia, liver disease, kidney disease, digestive disorders, asthma, chronic lung disease, etc.

    **7. ADL disability**
    - 0 = No; 1 = Yes
    - 6 items: dressing, walking across a room, bathing/showering, eating, getting in/out of bed, using the toilet
    - Difficulty in ANY item = 1

    **8. IADL disability**
    - 0 = No; 1 = Yes
    - 5 items: meal preparation, grocery shopping, making phone calls, taking medication, managing money
    - Difficulty in ANY item = 1

    **9. Gender**
    - 0 = Female; 1 = Male
    """)

# ---- 4. 输入控件 ----
child_gender = st.sidebar.selectbox(
    "Genders of children", (0, 1), index=0,
    help="0 = All daughters; 1 = Has sons (with or without daughters)"
)
child_education = st.sidebar.selectbox(
    "Highest degree obtained by children", (0, 1), index=0,
    help="0 = All below-college; 1 = At least one attended college"
)
child_income = st.sidebar.selectbox(
    "Income level of children", (0, 1), index=0,
    help="0 = All children have annual income < 10,000; 1 = At least one child has annual income >= 10,000"
)
child_face_to_face = st.sidebar.selectbox(
    "Visits with children", (0, 1), index=0,
    help="0 = <Once/year or never; 1 = >=Once/year"
)
gender = st.sidebar.selectbox(
    "Gender", ("Female", "Male"), index=0,
    help="Patient gender"
)
life_sat = st.sidebar.slider(
    "Life satisfaction (1-5)", 1, 5, 3,
    help="1 = Not at all; 2 = Not very; 3 = Somewhat; 4 = Very; 5 = Completely"
)
chronic_base = st.sidebar.selectbox(
    "Chronic disease", (0, 1), index=0,
    help="0 = No; 1 = Yes (hypertension, diabetes, heart disease, stroke, arthritis, cancer, etc.)"
)
adl = st.sidebar.selectbox(
    "ADL disability", (0, 1), index=0,
    help="0 = No; 1 = Yes (6 items; difficulty in ANY item = 1)"
)
iadl = st.sidebar.selectbox(
    "IADL disability", (0, 1), index=0,
    help="0 = No; 1 = Yes (5 items; difficulty in ANY item = 1)"
)

# ---- 5. 构建输入 ----
input_dict = {
    "child_gender": int(child_gender),
    "child_max_education_level.x": int(child_education),
    "child_income_level": int(child_income),
    "child_face_to_face": int(child_face_to_face),
    "life_sat": int(life_sat),
    "chronic_base": int(chronic_base),
    "ADL_disability": int(adl),
    "IADL_disability": int(iadl),
    "gender_Male": 1 if gender == "Male" else 0
}

input_df = pd.DataFrame([input_dict])[feature_names]

st.subheader("📥 Input Features")
st.dataframe(input_df, use_container_width=True, hide_index=True)

# ---- 6. 预测 ----
input_scaled = scaler.transform(input_df)
prob = model.predict_proba(input_scaled)[0, 1]

st.markdown("---")
st.subheader("🎯 Prediction Result")
st.metric("Predicted Probability", f"{prob*100:.1f}%")
st.progress(float(prob))

if prob < 0.3:
    st.success("**Low Risk** (< 30%): Routine follow-up recommended")
elif prob < 0.6:
    st.warning("**Intermediate Risk** (30%-60%): Enhanced monitoring recommended")
else:
    st.error("**High Risk** (>= 60%): Early intervention recommended")

# ---- 7. 变量定义表 ----
st.markdown("---")
st.subheader("📖 Variable Definitions")
var_def = pd.DataFrame({
    "Variable": [
        "Genders of children",
        "Highest degree obtained by children",
        "Income level of children",
        "Visits with children",
        "Life satisfaction",
        "Chronic disease",
        "ADL disability",
        "IADL disability",
        "Gender"
    ],
    "Definition": [
        "0 = All daughters; 1 = Has sons",
        "0 = All below-college; 1 = At least one attended college",
        "0 = All children have annual income < 10,000; 1 = At least one child has annual income >= 10,000",
        "0 = <Once/year; 1 = >=Once/year",
        "1 = Not at all; 2 = Not very; 3 = Somewhat; 4 = Very; 5 = Completely",
        "0 = No; 1 = Yes (hypertension, diabetes, heart disease, stroke, arthritis, cancer, etc.)",
        "0 = No; 1 = Yes (6 items; difficulty in ANY item = 1)",
        "0 = No; 1 = Yes (5 items; difficulty in ANY item = 1)",
        "0 = Female; 1 = Male"
    ]
})
st.dataframe(var_def, use_container_width=True, hide_index=True)

# ---- 8. 模型性能 ----
st.markdown("---")
st.subheader("📊 Model Performance (External Validation, HRS Wave 9, 2008)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC", "0.776")
col2.metric("Accuracy", "0.663")
col3.metric("Sensitivity", "0.814")
col4.metric("Specificity", "0.625")
st.caption("External validation cohort: HRS Wave 9 (2008, n = 908) | Event rate: 20.2%")

# ---- 9. 使用说明 ----
st.markdown("---")
st.subheader("📝 How to Use")
st.markdown("""
1. Input patient characteristics in the left sidebar
2. View the predicted risk and risk stratification
3. Make clinical decisions based on risk level
   - Low Risk (< 30%): Routine follow-up
   - Intermediate Risk (30%-60%): Enhanced monitoring
   - High Risk (>= 60%): Early intervention

**⚠️ Important Notes:**
- This tool is designed for **adults aged >= 45 years**.
- It should not be used for younger populations.
- Model developed using CHARLS data; externally validated using HRS Wave 9 (2008).
""")

# ---- 10. 底部 ----
st.markdown("---")
st.caption("⚠️ This tool is for research purposes only and does not replace clinical diagnosis.")
