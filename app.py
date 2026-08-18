import streamlit as st
import airportsdata
from datetime import datetime
from agents import agent_0_validator, agent_1_meteorologist, agent_2_data_scientist, agent_3_dispatcher, agent_4_chatbot

# 1. PAGE CONFIGURATION (Must be the first Streamlit command)
st.set_page_config(page_title="AeroResolve AI", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# 2. CUSTOM SAAS UI INJECTION
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background and App container */
    .stApp {
        background-color: #0b0f19;
        background-image: radial-gradient(circle at top right, #1a233a, #0b0f19);
        color: #e2e8f0;
    }

    /* Hide Streamlit default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Custom Gradient Button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.5);
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
    }

    /* Custom Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        backdrop-filter: blur(5px);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.875rem;
        font-weight: 700;
    }
    .metric-unit {
        color: #64748b;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Custom Alert Boxes */
    .alert-high {
        background: linear-gradient(90deg, rgba(220, 38, 38, 0.1) 0%, rgba(153, 27, 27, 0.2) 100%);
        border-left: 4px solid #ef4444;
        padding: 20px;
        border-radius: 8px;
        color: #fca5a5;
    }
    .alert-mod {
        background: linear-gradient(90deg, rgba(217, 119, 6, 0.1) 0%, rgba(180, 83, 9, 0.2) 100%);
        border-left: 4px solid #f59e0b;
        padding: 20px;
        border-radius: 8px;
        color: #fcd34d;
    }
    .alert-low {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 95, 70, 0.2) 100%);
        border-left: 4px solid #10b981;
        padding: 20px;
        border-radius: 8px;
        color: #6ee7b7;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #f8fafc !important;
        border-bottom-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. APP HEADER
st.markdown("<h1 style='text-align: center; color: #f8fafc; font-weight: 700; letter-spacing: -1px;'>✈️ AeroResolve AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 30px;'>Next-Generation Flight Logistics & Risk Assessment</p>", unsafe_allow_html=True)

# 4. LOAD MAPPING DATA
@st.cache_data
def load_maps():
    all_airports = airportsdata.load('IATA')
    airport_map = {f"{d.get('city', 'Unknown')} - {d.get('name', 'Airport')} ({code})": code 
                   for code, d in all_airports.items() if d.get('country') == 'US'}
    return dict(sorted(airport_map.items()))

AIRPORT_MAP = load_maps()
AIRLINE_MAP = {
    "American Airlines (AA)": "AA", "Delta Air Lines (DL)": "DL",
    "United Airlines (UA)": "UA", "Southwest Airlines (WN)": "WN",
    "JetBlue Airways (B6)": "B6", "Alaska Airlines (AS)": "AS",
    "FedEx Express (FX)": "FX", "UPS Airlines (5X)": "5X"
}

# 5. MODERN SIDEBAR
with st.sidebar:
    st.markdown("<h3 style='color: #f8fafc; margin-bottom: 20px;'>🛠️ Mission Parameters</h3>", unsafe_allow_html=True)
    airline_name = st.selectbox("Carrier", options=list(AIRLINE_MAP.keys()))
    origin_name = st.selectbox("Origin", options=list(AIRPORT_MAP.keys()), index=100)
    dest_name = st.selectbox("Destination", options=list(AIRPORT_MAP.keys()), index=200)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #f8fafc; margin-bottom: 20px;'>⏱️ Schedule</h3>", unsafe_allow_html=True)
    flight_date = st.date_input("Date", min_value=datetime.today())
    dep_hour = st.slider("Departure Time (24h)", 0, 23, 17, format="%02d:00")
    
    raw_airline_code = AIRLINE_MAP[airline_name]
    origin_code = AIRPORT_MAP[origin_name]
    dest_code = AIRPORT_MAP[dest_name]

# 6. TABBED INTERFACE
tab1, tab2 = st.tabs(["🛫 Command Center", "💬 AI Assistant"])

# ==========================================
# TAB 1: COMMAND CENTER
# ==========================================
with tab1:
    # Beautiful status pill
    st.markdown(f"""
    <div style='background-color: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); padding: 12px 20px; border-radius: 8px; margin-bottom: 24px; display: flex; align-items: center; justify-content: center;'>
        <span style='color: #60a5fa; font-weight: 600;'>🎯 Target Mission:</span>
        <span style='color: #f8fafc; margin-left: 10px;'>{airline_name} | {origin_code} ➔ {dest_code} | {flight_date.strftime('%d %b %Y')} @ {dep_hour:02d}:00</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Execute Risk Assessment", type="primary"):
        with st.status("Initializing AeroResolve AI...", expanded=True) as status:
            
            st.write("🛂 Agent 0: Validating Route Feasibility...")
            validation = agent_0_validator(raw_airline_code, origin_code, dest_code)
            
            if not validation.get("feasible", True):
                status.update(label="Mission Aborted: Invalid Route", state="error", expanded=True)
                st.error(f"🚨 **OPERATIONAL DISCREPANCY DETECTED**\n\n{validation.get('reason', 'Invalid route parameters.')}")
                st.stop() # Stops execution right here! Saves weather tokens & ML compute!
            
            st.write("📡 Fetching GPS and live weather telemetry...")
            weather = agent_1_meteorologist(origin_code, dest_code, flight_date, dep_hour)
            
            st.write("🧮 Running XGBoost risk calculation...")
            risk = agent_2_data_scientist(weather, raw_airline_code, flight_date, dep_hour)
            
            st.write("📚 Consulting FAA Knowledge Base (RAG)...")
            report = agent_3_dispatcher(risk, flight_date)
            if risk.get("prediction_available", False):
                status.update(label="Mission Assessment Complete!", state="complete", expanded=False)
            else:
                status.update(label="Mission Assessment Incomplete", state="error", expanded=True)
        
        # --- CUSTOM HTML METRICS DASHBOARD ---
        st.markdown("<h3 style='color: #f8fafc; margin-top: 30px; margin-bottom: 15px;'>📊 Live Telemetry</h3>", unsafe_allow_html=True)
        
        def metric_value(value, decimals=1):
            if value is None:
                return "N/A"
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return "N/A"

        metrics_html = f"""
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;'>
            <div class='metric-card'>
                <div class='metric-title'>Origin Wind</div>
                <div class='metric-value'>{metric_value(risk.get('wind_origin'))} <span class='metric-unit'>km/h</span></div>
            </div>
            <div class='metric-card'>
                <div class='metric-title'>Origin Visibility</div>
                <div class='metric-value'>{metric_value(risk.get('visib_origin'))} <span class='metric-unit'>km</span></div>
            </div>
            <div class='metric-card'>
                <div class='metric-title'>Dest Wind</div>
                <div class='metric-value'>{metric_value(risk.get('wind_dest'))} <span class='metric-unit'>km/h</span></div>
            </div>
            <div class='metric-card'>
                <div class='metric-title'>Dest Precip</div>
                <div class='metric-value'>{metric_value(risk.get('precip_dest'))} <span class='metric-unit'>mm</span></div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #f8fafc; margin-bottom: 15px;'>⚠️ AI Risk Analysis</h3>", unsafe_allow_html=True)
        
        # Dynamic Custom Risk Alert Box
        risk_prob = risk.get('risk_prob')
        if risk_prob is None:
            st.error(f"Risk prediction unavailable: {risk.get('risk_level', 'Model or weather data failed.')}")
            for weather_error in risk.get("weather_errors", []):
                st.warning(weather_error)
        else:
            risk_percentage = risk_prob * 100
        if risk_prob is not None and risk_prob >= 0.60:
            st.markdown(f"<div class='alert-high'><strong>🚨 CRITICAL RISK: {risk_percentage:.1f}%</strong><br><br>{risk['risk_level']}</div>", unsafe_allow_html=True)
        elif risk_prob is not None and risk_prob >= 0.40:
            st.markdown(f"<div class='alert-mod'><strong>⚠️ MODERATE RISK: {risk_percentage:.1f}%</strong><br><br>{risk['risk_level']}</div>", unsafe_allow_html=True)
        elif risk_prob is not None:
            st.markdown(f"<div class='alert-low'><strong>✅ CLEAR TO FLY: {risk_percentage:.1f}%</strong><br><br>{risk['risk_level']}</div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- EXPANDABLE REPORT ---
        with st.expander("📄 View Executive Dispatcher Briefing", expanded=True):
            st.markdown(report)

# ==========================================
# TAB 2: AI ASSISTANT
# ==========================================
with tab2:
    st.markdown("<h3 style='color: #f8fafc;'>💬 Ask AeroResolve</h3>", unsafe_allow_html=True)
    st.caption("Ask questions regarding aviation rules, weather limits, or FAA protocols.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I assist your logistics planning today?"}]
        st.session_state.chat_count = 0 

    MAX_CHATS = 5 

    # Draw chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input with Rate Limit
    if st.session_state.chat_count >= MAX_CHATS:
        st.error("🔒 Security Guardrail: Query limit reached for this session.")
    elif prompt := st.chat_input("Ask about crosswind limits, etc..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_count += 1
        st.chat_message("user").write(prompt)
        
        with st.spinner("Analyzing..."):
            response = agent_4_chatbot(
                prompt, st.session_state.messages,
                raw_airline_code, origin_code, dest_code,
                flight_date, dep_hour
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
