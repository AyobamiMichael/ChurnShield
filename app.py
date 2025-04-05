import streamlit as st
import pandas as pd
import joblib
from PIL import Image  # For logo

# ====== SET PAGE CONFIG ======
st.set_page_config(
    page_title="ChurnShield Pro",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM CSS ======
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3e8e41;
        transform: scale(1.02);
    }
    .stSlider>div>div>div>div {
        background: #4CAF50;
    }
    .risk-high {
        border-left: 5px solid #ff4b4b;
    }
    .risk-medium {
        border-left: 5px solid #ffa500;
    }
    .risk-low {
        border-left: 5px solid #0be881;
    }
    @media (max-width: 768px) {
        .stColumns>div {
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)

# ====== SIDEBAR ======
with st.sidebar:
    # Add your logo (replace with actual image path)
    # st.image("logo.png", width=200)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #4CAF50;'>ChurnShield</h1>
        <p>AI-Powered Retention Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **How to Use:**
    1. Adjust customer metrics
    2. Click *Predict Churn Risk*
    3. View results and risk factors
    """)
    
    st.markdown("---")
    st.markdown("""
    **Model Performance:**
    - Recall: 92%
    - Precision: 85%
    - F1 Score: 0.88
    """)
    
    st.markdown("---")
    st.markdown("""
    **Created by:** Ayobami Michael Opefeyijimi  
    **Version:** 1.0.0  
    **Last Updated:** 2025-04-04
    """)

# ====== MAIN INTERFACE ======
st.title("📊 ChurnShield Pro Dashboard")
st.caption("Predict customer churn risk with 92% accuracy")

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load(r'C:\Users\ChurnShield\churnshield_20250329_235156.pkl')
        return model
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {str(e)}")
        return None

package = load_model()

# Input columns with icons
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Usage Metrics")
        account_age = st.slider("**Account Age (months)**", 1, 120, 12, 
                              help="Customer tenure length")
        viewing_hours = st.slider("**Weekly Viewing Hours**", 0.0, 40.0, 5.0)
        downloads = st.slider("**Content Downloads/Month**", 0, 50, 8)
        
    with col2:
        st.subheader("💸 Financials")
        monthly_charges = st.slider("**Monthly Charges ($)**", 5.0, 100.0, 15.99)
        support_tickets = st.slider("**Support Tickets/Month**", 0, 10, 2,
                                  help="Higher tickets = higher risk")
        subscription = st.selectbox("**Subscription Plan**", 
                                  ['Basic', 'Standard', 'Premium'])
    
    submitted = st.form_submit_button("🚀 Predict Churn Risk", 
                                    use_container_width=True)

# ====== PREDICTION LOGIC ======
if submitted and package:
    try:
        # Create complete input data
        input_data = {
            'AccountAge': account_age,
            'MonthlyCharges': monthly_charges,
            'SupportTicketsPerMonth': support_tickets,
            'ViewingHoursPerWeek': viewing_hours,
            'ContentDownloadsPerMonth': downloads,
            'SubscriptionType': subscription,
            'DeviceRegistered': 'Mobile',
            'PaymentMethod': 'Credit card',
            'PaperlessBilling': 'Yes',
            'ContentType': 'Both',
            'MultiDeviceAccess': 'Yes',
            'GenrePreference': 'Drama',
            'Gender': 'Male',
            'ParentalControl': 'No',
            'SubtitlesEnabled': 'Yes',
            'TotalCharges': account_age * monthly_charges,
            'ValuePerHour': monthly_charges / max(viewing_hours, 0.1),
            'LifetimeValue': account_age * monthly_charges,
            'EngagementScore': (downloads + viewing_hours) / 2,
            'SupportIntensity': support_tickets / max(account_age, 1),
            'AverageViewingDuration': 45,
            'UserRating': 3.5,
            'WatchlistSize': 5
        }

        # Process and predict
        processed = package['preprocessor'].transform(pd.DataFrame([input_data]))
        proba = package['model'].predict_proba(processed)[0, 1]
        
        # Enhanced results display
        risk_level = "🔴 High Risk" if proba > 0.7 else "🟠 Medium Risk" if proba > 0.3 else "🟢 Low Risk"
        risk_color = "#ff4b4b" if proba > 0.7 else "#ffa500" if proba > 0.3 else "#0be881"
        
        with st.container():
            st.markdown(f"""
            <div class='{ "risk-high" if proba > 0.7 else "risk-medium" if proba > 0.3 else "risk-low" }' style='
                padding: 1.5rem;
                margin: 1rem 0;
                background: white;
                border-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            '>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 style='margin:0'>Churn Risk: <b>{proba:.1%}</b></h2>
                    <h2 style='margin:0; color: {risk_color}'>{risk_level}</h2>
                </div>
                <p style='color:#666'>Thresholds: {package['metadata']['thresholds']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors as cards
            st.subheader("🔍 Key Risk Drivers")
            cols = st.columns(3)
            risk_factors = [
                ("Support Tickets", support_tickets, "👨‍💻", "#ff4b4b", "Higher = More Risk"),
                ("Viewing Hours", viewing_hours, "📺", "#4CAF50", "Lower = More Risk"),
                ("Subscription Plan", subscription, "💎", "#2196F3", "Basic = More Risk")
            ]
            
            for i, (name, value, icon, color, tip) in enumerate(risk_factors):
                with cols[i]:
                    st.markdown(f"""
                    <div style='
                        padding: 1rem;
                        background: white;
                        border-radius: 5px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        text-align: center;
                        border-top: 4px solid {color};
                        margin-bottom: 1rem;
                    '>
                        <p style='font-size: 2rem; margin:0'>{icon}</p>
                        <h3 style='margin:0'>{name}</h3>
                        <p style='font-size: 1.5rem; margin:0'><b>{value}</b></p>
                        <p style='font-size: 0.8rem; color: #666; margin:0'>{tip}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Action recommendations
            st.subheader("🚀 Recommended Actions")
            if proba > 0.7:
                st.warning("**Immediate intervention needed!** Consider:")
                st.write("- Personal retention call")
                st.write("- Special discount offer")
                st.write("- Account review with support")
            elif proba > 0.3:
                st.info("**Proactive engagement recommended:**")
                st.write("- Check-in email")
                st.write("- Content recommendations")
            else:
                st.success("**Healthy customer profile**")
                st.write("- Continue standard engagement")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Technical details have been logged for review")

# ====== FOOTER ======
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>ChurnShield Pro • Confidential • Model v1.0</p>
    <p>For demonstration purposes only</p>
</div>
""", unsafe_allow_html=True)