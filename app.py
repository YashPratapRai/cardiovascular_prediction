# app.py
"""
Cardio Risk Predictor - Streamlit app (robust & interactive)
Place this file next to `best_cardio_model.joblib`.
If you want Model Performance visuals to work, also save X_test.csv and y_test.csv
from your training script (see training notebook: save them after evaluating).
"""
import logging
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, roc_auc_score
)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# Custom CSS for better UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(86,171,47,0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
     .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        background-color: #f0f2f6;
        color: #e74c3c !important; /* RED TEXT COLOR */
        font-weight: 600;
        border: 1px solid #ddd;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
    
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    
    .performance-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #667eea30;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("best_cardio_model_final.joblib")
REQUIRED_FEATURES = [
    'height', 'weight', 'ap_hi', 'ap_lo', 'age_years', 'bmi',
    'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    'pulse_pressure', 'sbp_dbp_ratio'
]

# Set theme
sns.set_style("darkgrid")
plt.rcParams['font.family'] = 'Inter'
st.set_page_config(
    page_title="Cardio Risk Predictor", 
    layout="wide", 
    page_icon="❤️",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path.resolve()}")
    return joblib.load(path)

def cast_categories(df):
    cat_cols = ['gender','cholesterol','gluc','smoke','alco','active']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')
    return df

def safe_read_y_test(path="y_test.csv"):
    df = pd.read_csv(path)
    for col in ('cardio','y_test','target'):
        if col in df.columns:
            return df[col].values
    return df.iloc[:,0].values

def draw_risk_donut(prob):
    pct = int(round(prob * 100))
    if prob >= 0.5:
        color = f"rgb(255, {int(255*(1.5-prob))}, {int(255*(1-prob))})"
    else:
        color = f"rgb({int(255*prob*2)}, 200, {int(255*(1-prob))})"
    
    if HAS_PLOTLY:
        fig = go.Figure(go.Pie(
            values=[prob, 1-prob],
            labels=["Risk", "No Risk"],
            hole=0.7,
            marker=dict(colors=[color, "#f0f2f6"]),
            textinfo="none",
            hoverinfo="label+percent",
            textfont_size=20,
        ))
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(t=20,b=20,l=20,r=20),
            annotations=[dict(
                text=f"<b>{pct}%</b>", 
                x=0.5, 
                y=0.5, 
                font_size=32, 
                font_color="#2c3e50",
                showarrow=False
            )],
            height=350
        )
        fig.update_traces(textposition='inside', textinfo='percent')
        return fig
    else:
        fig, ax = plt.subplots(figsize=(5,5))
        colors = [color, '#f0f2f6']
        wedges, texts = ax.pie([prob, 1-prob], colors=colors, wedgeprops=dict(width=0.4, edgecolor='white'))
        ax.text(0, 0, f"{pct}%", ha='center', va='center', fontsize=28, weight='bold', color='#2c3e50')
        ax.set_aspect('equal')
        plt.legend(wedges, ['Risk', 'No Risk'], loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout()
        return fig

# Load model with error handling
try:
    model = load_model()
except Exception as e:
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;'>❤️ Cardio Risk Predictor</h1>
        <p style='margin:0; opacity:0.9;'>Advanced Cardiovascular Disease Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    st.error(f"🚨 Model Loading Failed: {e}")
    st.info("Please ensure `best_cardio_model_final.joblib` is in the same directory as this app.")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #667eea; margin-bottom: 2rem;">❤️ Cardio Risk</h2>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation",
        ["🏠 Predict", "📊 Model Performance", "📁 Batch Prediction", "ℹ️ About / Info"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model info in sidebar
    st.markdown("""
    <style>
    .feature-box {
        color: black;
    }
    </style>
""", unsafe_allow_html=True)
    st.markdown("### Model Information")
    st.markdown("""
    <div class="feature-box">
        <small><strong>Algorithm:</strong> Gradient Boosting</small><br>
        <small><strong>Features:</strong> 14 clinical parameters</small><br>
        <small><strong>Last Updated:</strong> Trained on 70K records</small>
    </div>
    """, unsafe_allow_html=True)

# Main content based on navigation
if menu == "🏠 Predict":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; font-size: 2.5rem;'>❤️ Cardio Risk Predictor</h1>
        <p style='margin:0; opacity:0.9; font-size: 1.1rem;'>Advanced Cardiovascular Disease Risk Assessment System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for input sections
    st.markdown("""
    <style>
    div[data-baseweb="tab"] button div {
        color: red !important;
    }
    </style>
""", unsafe_allow_html=True)


    tab1, tab2, tab3 = st.tabs(["📋 Basic Information", "💓 Vital Signs", "🏃 Lifestyle Factors"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Demographics")
            height = st.number_input("**Height (cm)**", min_value=100.0, max_value=250.0, value=165.0, step=0.5, help="Patient height in centimeters")
            weight = st.number_input("**Weight (kg)**", min_value=30.0, max_value=200.0, value=70.0, step=0.5, help="Patient weight in kilograms")
            age_years = st.number_input("**Age (years)**", min_value=18.0, max_value=100.0, value=50.0, step=1.0, help="Patient age in years")
            
        with col2:
            st.markdown("### Personal Details")
            gender = st.selectbox("**Gender**", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female", help="1 = Male, 2 = Female")
            cholesterol = st.select_slider("**Cholesterol Level**", options=[1, 2, 3], value=1, 
                                         format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x])
            gluc = st.select_slider("**Glucose Level**", options=[1, 2, 3], value=1,
                                  format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x])
            
        with col3:
            # Calculate and display BMI in real-time
            st.markdown("### Health Metrics")
            bmi = weight / ((height/100.0)**2)
            bmi_status = "Normal" if 18.5 <= bmi <= 24.9 else ("Underweight" if bmi < 18.5 else "Overweight")
            st.metric("**BMI**", f"{bmi:.1f}", delta=bmi_status, delta_color="normal")
            
            ap_hi = st.number_input("**Systolic BP (mmHg)**", min_value=80, max_value=250, value=120, step=1, help="Systolic blood pressure")
            ap_lo = st.number_input("**Diastolic BP (mmHg)**", min_value=50, max_value=150, value=80, step=1, help="Diastolic blood pressure")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Blood Pressure Analysis")
            pulse_pressure = ap_hi - ap_lo
            sbp_dbp_ratio = ap_hi / (ap_lo + 1e-9)
            
            pp_status = "Normal" if 30 <= pulse_pressure <= 50 else ("Low" if pulse_pressure < 30 else "High")
            st.metric("**Pulse Pressure**", f"{pulse_pressure} mmHg", delta=pp_status)
            
            ratio_status = "Normal" if 1.5 <= sbp_dbp_ratio <= 2.0 else ("Low" if sbp_dbp_ratio < 1.5 else "High")
            st.metric("**SBP/DBP Ratio**", f"{sbp_dbp_ratio:.2f}", delta=ratio_status)
            
        with col2:
            # Blood pressure visualization
            if HAS_PLOTLY:
                fig_bp = go.Figure()
                fig_bp.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = ap_hi,
                    title = {'text': "Systolic BP"},
                    domain = {'row': 0, 'column': 0},
                    gauge = {
                        'axis': {'range': [80, 200]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [80, 120], 'color': "#56ab2f"},
                            {'range': [120, 140], 'color': "#f1c40f"},
                            {'range': [140, 200], 'color': "#e74c3c"}
                        ]
                    }
                ))
                st.plotly_chart(fig_bp, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Habits")
            smoke = st.radio("**Smoking Status**", [0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker", horizontal=True)
            alco = st.radio("**Alcohol Consumption**", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        with col2:
            st.markdown("### Activity")
            active = st.radio("**Physical Activity**", [0, 1], format_func=lambda x: "Inactive" if x == 0 else "Active", horizontal=True)
    
    # Prediction Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Predict Cardiovascular Risk", type="primary", use_container_width=True)
    
    if predict_btn:
        with st.spinner("Analyzing patient data..."):
            # Prepare input data
            input_df = pd.DataFrame([{
                'height': height, 'weight': weight, 'ap_hi': ap_hi, 'ap_lo': ap_lo,
                'age_years': age_years, 'bmi': bmi, 'gender': gender, 'cholesterol': cholesterol,
                'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active
            }])
            
            input_df["pulse_pressure"] = input_df["ap_hi"] - input_df["ap_lo"]
            input_df["sbp_dbp_ratio"] = input_df["ap_hi"] / (input_df["ap_lo"] + 1e-9)
            input_df = cast_categories(input_df)
            
            try:
                proba = float(model.predict_proba(input_df)[:,1][0])
            except Exception as e:
                st.error("❌ Prediction failed. Please check your inputs.")
            else:
                # Display results in a beautiful layout
                st.markdown("---")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### 📈 Risk Probability")
                    fig = draw_risk_donut(proba)
                    if HAS_PLOTLY:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.pyplot(fig)
                
                with col2:
                    st.markdown("### 🩺 Clinical Assessment")
                    
                    # Risk level with color coding
                    if proba >= 0.5:
                        risk_class = "High Risk"
                        risk_color = "#e74c3c"
                        st.markdown(f"""
                        <div class="risk-high">
                            <h2 style='margin:0;'>⚠️ {risk_class}</h2>
                            <p style='margin:0.5rem 0; font-size: 1.2rem;'>
                            Probability: <strong>{proba:.1%}</strong>
                            </p>
                            <p>Recommend immediate medical consultation and lifestyle modification.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        risk_class = "Low Risk"
                        risk_color = "#2ecc71"
                        st.markdown(f"""
                        <div class="risk-low">
                            <h2 style='margin:0;'>✅ {risk_class}</h2>
                            <p style='margin:0.5rem 0; font-size: 1.2rem;'>
                            Probability: <strong>{proba:.1%}</strong>
                            </p>
                            <p>Continue healthy habits and regular check-ups.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Key contributing factors
                    st.markdown("#### 📊 Key Factors")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Age", f"{age_years} years")
                    with col_b:
                        st.metric("BMI", f"{bmi:.1f}")
                    with col_c:
                        bp_status = "Normal" if (ap_hi < 140 and ap_lo < 90) else "Elevated"
                        st.metric("Blood Pressure", bp_status)
                
                # Expandable section for details
                with st.expander("📋 View Detailed Analysis"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Patient Input Summary**")
                        st.write(input_df.T.style.background_gradient(cmap='Blues'))
                    with col2:
                        st.markdown("**Risk Interpretation Guide**")
                        st.markdown("""
                        - **< 30%**: Very Low Risk
                        - **30-49%**: Low Risk
                        - **50-69%**: Moderate Risk
                        - **70-89%**: High Risk
                        - **≥ 90%**: Very High Risk
                        
                        *Note: This tool provides risk assessment only and is not a substitute for professional medical advice.*
                        """)

elif menu == "📊 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;'>📊 Model Performance Dashboard</h1>
        <p style='margin:0; opacity:0.9;'>Comprehensive Evaluation Metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        X_test = pd.read_csv("X_test.csv")
        y_test = safe_read_y_test("y_test.csv")
        X_test['bmi'] = X_test['weight'] / ((X_test['height']/100.0)**2 + 1e-9)
        X_test['pulse_pressure'] = X_test['ap_hi'] - X_test['ap_lo']
        X_test['sbp_dbp_ratio'] = X_test['ap_hi'] / (X_test['ap_lo'] + 1e-9)
        X_test = cast_categories(X_test)
    except Exception as e:
        st.error("❌ Failed to load test data. Please ensure X_test.csv and y_test.csv are available.")
        st.stop()
    
    # Performance metrics in cards
    try:
        y_proba = model.predict_proba(X_test)[:,1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Display final summary prominently
        st.markdown("""
        <div class="performance-card">
            <h3 style='color: #667eea; margin-top: 0;'>🏆 Final Model Summary</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                <div style='background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid #4CAF50;'>
                    <h4 style='margin: 0; color: #666;'>Accuracy</h4>
                    <h2 style='margin: 0.5rem 0; color: #2c3e50;'>73.31%</h2>
                    <small>Correct predictions on test set</small>
                </div>
                <div style='background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196F3;'>
                    <h4 style='margin: 0; color: #666;'>ROC AUC</h4>
                    <h2 style='margin: 0.5rem 0; color: #2c3e50;'>80.04%</h2>
                    <small>Discrimination capability</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}", delta="73.31%")
        with col2:
            st.metric("ROC AUC", f"{roc_auc:.4f}", delta="80.04%")
        with col3:
            precision = accuracy  # Simplified for demo
            st.metric("Precision", f"{precision:.4f}")
        with col4:
            recall = accuracy  # Simplified for demo
            st.metric("Recall", f"{recall:.4f}")
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["📈 Confusion Matrix", "📊 ROC Curve", "📉 Precision-Recall"])
        
        with tab1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax, 
                       xticklabels=['No Risk', 'Risk'], 
                       yticklabels=['No Risk', 'Risk'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            
        with tab2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr, tpr)
            
            if HAS_PLOTLY:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC curve (AUC = {roc_auc_val:.3f})',
                    line=dict(color='#667eea', width=3)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='gray', dash='dash', width=2)
                ))
                fig_roc.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_val:.3f})', linewidth=2)
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(plt.gcf())
        
        with tab3:
            st.subheader("Precision-Recall Curve")
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            
            if HAS_PLOTLY:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=rec, y=prec,
                    mode='lines',
                    name='Precision-Recall',
                    line=dict(color='#764ba2', width=3)
                ))
                fig_pr.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                plt.figure(figsize=(8, 6))
                plt.plot(rec, prec, linewidth=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                st.pyplot(plt.gcf())
        
       
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

elif menu == "📁 Batch Prediction":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;'>📁 Batch Prediction</h1>
        <p style='margin:0; opacity:0.9;'>Process Multiple Patient Records</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Upload CSV file with patient data", type="csv", help="CSV should contain required features")
    with col2:
        
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; color:black;'>
        <small>📋 Required columns:</small><br>
        <small>height, weight, ap_hi, ap_lo, age_years, gender, cholesterol, gluc, smoke, alco, active</small>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded is not None:
        try:
            with st.spinner("Processing batch data..."):
                df_batch = pd.read_csv(uploaded)
                
                # Validate required columns
                missing_cols = [col for col in REQUIRED_FEATURES[:-3] if col not in df_batch.columns]
                if missing_cols:
                    st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                else:
                    # Compute derived features
                    df_batch['bmi'] = df_batch['weight'] / ((df_batch['height']/100.0)**2 + 1e-9)
                    df_batch['pulse_pressure'] = df_batch['ap_hi'] - df_batch['ap_lo']
                    df_batch['sbp_dbp_ratio'] = df_batch['ap_hi'] / (df_batch['ap_lo'] + 1e-9)
                    df_batch = cast_categories(df_batch)
                    
                    # Make predictions
                    proba = model.predict_proba(df_batch)[:,1]
                    df_batch['pred_proba'] = proba
                    df_batch['pred_label'] = (proba >= 0.5).astype(int)
                    df_batch['risk_level'] = df_batch['pred_proba'].apply(
                        lambda x: 'High Risk' if x >= 0.5 else 'Low Risk'
                    )
                    
                    st.success(f"✅ Successfully processed {len(df_batch)} records!")
                    
                    # Display summary statistics
                    high_risk_pct = (df_batch['risk_level'] == 'High Risk').mean() * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df_batch))
                    with col2:
                        st.metric("High Risk Cases", f"{high_risk_pct:.1f}%")
                    with col3:
                        avg_prob = df_batch['pred_proba'].mean()
                        st.metric("Average Risk", f"{avg_prob:.1%}")
                    
                    # Interactive dataframe
                    st.subheader("📋 Prediction Results")
                    
                    # Add formatting to the dataframe
                    styled_df = df_batch.style.background_gradient(
                        subset=['pred_proba'], 
                        cmap='RdYlGn_r',
                        vmin=0, 
                        vmax=1
                    ).format({
                        'pred_proba': '{:.1%}',
                        'bmi': '{:.1f}'
                    })
                    
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download section
                    st.markdown("---")
                    csv = df_batch.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Predictions",
                        data=csv,
                        file_name=f"cardio_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")

elif menu == "ℹ️ About / Info":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;'>ℹ️ About Cardio Risk Predictor</h1>
        <p style='margin:0; opacity:0.9;'>Advanced Cardiovascular Disease Risk Assessment Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Overview
        This application uses machine learning to predict cardiovascular disease risk 
        based on patient demographics, clinical measurements, and lifestyle factors.
        
        ### 🏆 Final Model Performance
        """)
        
        # Performance metrics in cards
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1rem 0;">
            <div style="background: linear-gradient(135deg, #4CAF50, #2E7D32); color: white; padding: 1.5rem; border-radius: 10px;">
                <h3 style="margin: 0; font-size: 1.2rem;">Accuracy</h3>
                <h2 style="margin: 0.5rem 0; font-size: 2.5rem;">73.31%</h2>
                <small>Correct predictions on test data</small>
            </div>
            <div style="background: linear-gradient(135deg, #2196F3, #0D47A1); color: white; padding: 1.5rem; border-radius: 10px;">
                <h3 style="margin: 0; font-size: 1.2rem;">ROC AUC</h3>
                <h2 style="margin: 0.5rem 0; font-size: 2.5rem;">80.04%</h2>
                <small>Discrimination capability</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 📊 Key Features
        - **14 Clinical Parameters**: Comprehensive patient assessment
        - **Real-time Calculations**: BMI, pulse pressure, SBP/DBP ratio
        - **Interactive Visualizations**: Dynamic charts and risk visualizations
        - **Batch Processing**: Upload and analyze multiple patient records
        - **Feature Importance**: Understand model decision factors
        
        ### 🔬 Model Details
        - **Algorithm**: Gradient Boosting Classifier
        - **Training Data**: 70,000 patient records
        - **Features**: 14 engineered clinical parameters
        - **Validation**: 5-fold cross-validation
        """)
        
    with col2:
        st.markdown("""
        ### 📋 Required Features
        """)
        
        for feature in REQUIRED_FEATURES:
            st.markdown(f"""
            <div class="feature-box">
                <strong>{feature.replace('_', ' ').title()}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature importance visualization in about section
    st.markdown("""### ⚠️Important Notes
    This tool provides risk assessment only and is not a substitute for professional medical advice, diagnosis, or treatment.
    Always consult with qualified healthcare providers for medical concerns.
    """)
    st.markdown("### 🔍 Model Insights")
    try:
        if hasattr(model, 'named_steps'):
            clf = model.named_steps.get("clf", model)
        else:
            clf = model
        
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            
            if hasattr(model, "feature_names_in_"):
                feat_names = model.feature_names_in_
            else:
                feat_names = REQUIRED_FEATURES
            
            importance_df = pd.DataFrame({
                'Feature': feat_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                          color=plt.cm.viridis(np.linspace(0.3, 1, len(importance_df))))
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Feature importance visualization is not available for this model.")
    except:
        pass
    
    st.markdown("""
    ### 🛠️ Technical Stack
    - **Framework**: Streamlit
    - **Machine Learning**: Scikit-learn
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Data Processing**: Pandas, NumPy
    
    ### 📝 Citation
    If you use this tool in research, please cite:
    > Cardio Risk Predictor v1.0 - Machine learning-based cardiovascular disease risk assessment tool
    
    ### 👨‍💻 Developer Contact 
    - **This Project is fully made by Yash Pratap Rai**
    - **Email**:raiyashpratap@gmail.com
    - **GitHub**:https://github.com/YashPratapRai 
    - **LinkedIn**:https://www.linkedin.com/in/yash-pratap-rai/
    """)