import os
import math
import json
import requests
import joblib
import airportsdata
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta


from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# Initialized Groq LLM & HuggingFace Embeddings
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_PATH = os.path.join(
    BASE_DIR,
    "faiss_aviation_index"
)

# Loading model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.pkl"
)
xgb_model = joblib.load(MODEL_PATH)


ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "target_encoder.pkl"
)

target_encoder = joblib.load(ENCODER_PATH)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)

feature_columns = joblib.load(FEATURE_PATH)

try:
    vector_db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Failed to load FAISS index: {e}")
    vector_db = None


def get_flight_distance(origin, dest):
    """Calculates the exact distance in miles between two airports using GPS coordinates."""
    airports = airportsdata.load('IATA')
    if origin in airports and dest in airports:
        lat1, lon1 = airports[origin]['lat'], airports[origin]['lon']
        lat2, lon2 = airports[dest]['lat'], airports[dest]['lon']
        R = 3958.8

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return int(R * c)
    return 1000


def agent_0_validator(airline, origin, dest):
    """Acts as a gatekeeper to prevent wasting resources on impossible flights."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the AeroResolve Pre-Flight Gatekeeper. Your job is to validate if a commercial or cargo route is logically and legally possible.
        Check if the airline operates at these airports, if the runways support commercial jets, or if they are restricted (e.g., military bases, small general aviation fields).
        CRITICAL: You MUST respond in valid JSON format ONLY, with two keys: "feasible" (boolean) and "reason" (short string explanation). Do not add any conversational text.
        Example: {{"feasible": false, "reason": "Nellis AFB (LSV) is a restricted military airfield and does not accept commercial Delta flights."}}"""),
        ("user", "Airline: {airline} | Origin: {origin} | Dest: {dest}")
    ])

    try:
        response = (prompt | llm).invoke({
            "airline": airline,
            "origin": origin,
            "dest": dest
        }).content

        if isinstance(response, list):
            response = response[0].get('text', str(response))

        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx != -1 and end_idx != -1:
            clean_json = response[start_idx:end_idx+1]
            return json.loads(clean_json)
        else:
            raise ValueError(f"No valid JSON brackets found. Raw LLM response: {response}")

    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            return {"feasible": False, "reason": "⏳ **API Rate Limit Exceeded:** Please wait a few seconds before trying again."}

        print(f"⚠️ Agent 0 Parsing Error: {e}")
        return {"feasible": False, "reason": f"System Safety Override: Unable to validate route parameters securely. (Error: {e})"}

def agent_1_meteorologist(origin, dest, flight_date, dep_hour):
    airports = airportsdata.load('IATA')
    weather_data = {"origin": origin, "dest": dest}

    today = date.today()

    if isinstance(flight_date, datetime):
        flight_date = flight_date.date()

    days_diff = (flight_date - today).days

    # Open-Meteo forecast is used for today through the next 13 days
    target_date = today + timedelta(days=days_diff) if 0 <= days_diff < 14 else today
    target_time_str = f"{target_date.strftime('%Y-%m-%d')}T{dep_hour:02d}:00"

    for loc_type, airport_code in [("origin", origin), ("dest", dest)]:
        if airport_code in airports:
            lat = airports[airport_code]['lat']
            lon = airports[airport_code]['lon']

            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m,wind_speed_10m,visibility,precipitation"
                f"&timezone=auto&forecast_days=14"
            )

            try:
                res = requests.get(url, timeout=5)

                if res.status_code == 200:
                    data = res.json()
                    hourly = data.get('hourly', {})
                    times = hourly.get('time', [])

                    if target_time_str in times:
                        idx = times.index(target_time_str)
                    else:
                        day_offset = max(0, min(days_diff, 13)) if 0 <= days_diff < 14 else 0
                        idx = (day_offset * 24) + int(dep_hour)
                        idx = min(idx, len(times) - 1)

                    temp = hourly['temperature_2m'][idx] if 'temperature_2m' in hourly else 20.0
                    wind = hourly['wind_speed_10m'][idx] if 'wind_speed_10m' in hourly else 12.0
                    precip = hourly['precipitation'][idx] if 'precipitation' in hourly else 0.0
                    visib_raw = hourly['visibility'][idx] if 'visibility' in hourly else 10000.0

                    visib = (visib_raw / 1000.0) if visib_raw is not None else 10.0

                    weather_data.update({
                        f"temp_{loc_type}": round(float(temp if temp is not None else 20.0), 1),
                        f"wind_{loc_type}": round(float(wind if wind is not None else 10.0), 1),
                        f"visib_{loc_type}": round(float(visib), 1),
                        f"precip_{loc_type}": round(float(precip if precip is not None else 0.0), 1)
                    })
                else:
                    raise ValueError("API Error")

            except Exception as e:
                print(f"⚠️ Weather API Fetch Fallback for {airport_code}: {e}")
                weather_data.update({
                    f"temp_{loc_type}": 22.0,
                    f"wind_{loc_type}": 14.0,
                    f"visib_{loc_type}": 10.0,
                    f"precip_{loc_type}": 0.0
                })

    return weather_data


def agent_2_data_scientist(weather_data, airline, flight_date, dep_hour):
    """Predict flight delay risk using the trained XGBoost model."""

    origin = weather_data["origin"]
    dest = weather_data["dest"]

    arr_hour = (dep_hour + 3) % 24
    distance = get_flight_distance(origin, dest)

    # Weather
    temp_orig = weather_data.get("temp_origin", 20)
    precip_orig = weather_data.get("precip_origin", 0)
    wind_orig = weather_data.get("wind_origin", 0)
    visib_orig = weather_data.get("visib_origin", 10)

    temp_dest = weather_data.get("temp_dest", 20)
    precip_dest = weather_data.get("precip_dest", 0)
    wind_dest = weather_data.get("wind_dest", 0)
    visib_dest = weather_data.get("visib_dest", 10)

    # Same features used during training
    input_data = pd.DataFrame([{
        "Year": flight_date.year,
        "Month": flight_date.month,
        "DayofMonth": flight_date.day,
        "DayOfWeek": flight_date.weekday() + 1,

        "Reporting_Airline": airline,
        "Origin": origin,
        "Dest": dest,

        "CRSDepTime": dep_hour * 100,
        "CRSArrTime": arr_hour * 100,
        "Distance": distance,

        "DEP_HOUR": dep_hour,
        "ARR_HOUR": arr_hour,

        "temp_origin": temp_orig,
        "precip_origin": precip_orig,
        "wind_origin": wind_orig,
        "visib_origin": visib_orig,

        "temp_dest": temp_dest,
        "precip_dest": precip_dest,
        "wind_dest": wind_dest,
        "visib_dest": visib_dest,

        "dep_hour_sin": np.sin(2 * np.pi * dep_hour / 24),
        "dep_hour_cos": np.cos(2 * np.pi * dep_hour / 24),

        "arr_hour_sin": np.sin(2 * np.pi * arr_hour / 24),
        "arr_hour_cos": np.cos(2 * np.pi * arr_hour / 24),

        "is_morning_rush": int(6 <= dep_hour <= 9),
        "is_evening_rush": int(15 <= dep_hour <= 19),

        "weather_severity_origin":
            precip_orig * 2 + wind_orig * 0.5 + (10 - visib_orig),

        "weather_severity_dest":
            precip_dest * 2 + wind_dest * 0.5 + (10 - visib_dest),

        "storm_flag": int(
            precip_orig > 5 or
            precip_dest > 5 or
            wind_orig > 35 or
            wind_dest > 35
        ),

        "ROUTE": f"{origin}_{dest}",

        # Keep same feature structure as training.
        # These are historical traffic-frequency features.
        "origin_traffic_volume": 1000,
        "dest_traffic_volume": 1000,
        "route_traffic_volume": 100
    }])

    # Apply the SAME encoder used during model training
    input_data = target_encoder.transform(input_data)

    # Ensure exact feature order
    input_data = input_data[feature_columns]

    # XGBoost prediction
    prob = float(xgb_model.predict_proba(input_data)[0][1])

    if prob >= 0.60:
        risk_level = "HIGH RISK (Delay Probable)"
    elif prob >= 0.40:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"

    return {
        **weather_data,
        "risk_prob": prob,
        "risk_level": risk_level
    }

def agent_3_dispatcher(data, flight_date):
    rag_context = ""

    if vector_db:
        docs = vector_db.similarity_search(
            "What are the rules for crosswinds, visibility, and delays?",
            k=2
        )
        rag_context = "\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are the Chief Dispatcher. Today's date is {datetime.now().strftime('%B %d, %Y')}. "
            f"The flight is scheduled for {flight_date.strftime('%B %d, %Y')}. "
            "Use weather, ML risk, and FAA rules to write a brief decision."
        ),
        (
            "user",
            "Route: {origin} to {dest}\nML Risk: {risk_level} ({risk_prob})\nRules: {protocols}"
        )
    ])

    try:
        response = (prompt | llm).invoke({
            "origin": data.get('origin', 'N/A'),
            "dest": data.get('dest', 'N/A'),
            "risk_level": data.get('risk_level', 'UNKNOWN'),
            "risk_prob": f"{data.get('risk_prob', 0) * 100:.1f}%",
            "protocols": rag_context
        }).content

        if isinstance(response, list):
            return response[0].get('text', str(response))

        return str(response)

    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            return "⏳ **API Rate Limit Exceeded:** The system is overwhelmed. Please wait a few seconds before trying again."

        return f"⚠️ **Error generating report:** {str(e)}"


def agent_4_chatbot(user_message, chat_history, airline, origin, dest, flight_date, dep_hour):
    """Aviation chatbot using live weather API and ML risk prediction."""
    
    weather = agent_1_meteorologist(origin, dest, flight_date, dep_hour)
    risk = agent_2_data_scientist(weather, airline, flight_date, dep_hour)

    weather_context = f"""
Flight: {airline} {origin} → {dest}
Date: {flight_date.strftime('%d %b %Y')}
Departure: {dep_hour:02d}:00

Origin Weather:
Temperature: {weather.get('temp_origin', 'N/A')} °C
Wind: {weather.get('wind_origin', 'N/A')} km/h
Visibility: {weather.get('visib_origin', 'N/A')} km
Precipitation: {weather.get('precip_origin', 'N/A')} mm

Destination Weather:
Temperature: {weather.get('temp_dest', 'N/A')} °C
Wind: {weather.get('wind_dest', 'N/A')} km/h
Visibility: {weather.get('visib_dest', 'N/A')} km
Precipitation: {weather.get('precip_dest', 'N/A')} mm

ML Delay Risk:
Probability: {risk.get('risk_prob', 0) * 100:.1f}%
Risk Level: {risk.get('risk_level', 'N/A')}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the AeroResolve AI Assistant.

Answer questions about aviation, weather, and flight logistics.

You have access to weather API data and an XGBoost delay-risk prediction provided below.

IMPORTANT:
- Use the provided weather data instead of saying you do not have weather access.
- Use the ML risk probability when discussing possible delays.
- Clearly distinguish between ML delay risk and actual airline/ATC flight status.
- Never claim that a flight is definitely delayed or definitely on-time.
- For weather questions, report the provided weather values.
- Be concise and professional.
- If asked about programming, coding, recipes, essays, or unrelated topics, politely refuse.

DATA:
{weather_context}
"""),
        ("user", "{message}")
    ])

    try:
        response = (prompt | llm).invoke({
            "message": user_message,
            "weather_context": weather_context
        }).content

        if isinstance(response, list):
            return response[0].get('text', str(response))

        return str(response)

    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            return "⏳ **API Rate Limit Exceeded:** Please wait a few seconds and try again."

        return f"⚠️ **Error:** {str(e)}"