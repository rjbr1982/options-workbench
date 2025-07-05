import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import requests
from bs4 import BeautifulSoup
import json
from google.cloud import firestore
from google.oauth2 import service_account

# --- Firestore Initialization ---
def init_firestore_connection():
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_json)
        db = firestore.Client(credentials=creds)
        return db
    except Exception as e:
        st.error(f"שגיאה בהתחברות למסד הנתונים: {e}")
        st.warning("וודא שהגדרת את סודות האפליקציה (gcp_service_account) כראוי ב-Streamlit Cloud.")
        return None

# --- Profile Management Functions (Firestore) ---
def load_profiles_from_db(db, user_key):
    if db is None: return {}
    try:
        doc_ref = db.collection("user_profiles").document(user_key)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("profiles", {})
        else:
            doc_ref.set({"profiles": {}})
            return {}
    except Exception as e:
        st.error(f"שגיאה בטעינת פרופילים: {e}")
        return {}

def save_profile_to_db(db, user_key, profile_name, profile_data):
    if db is None: return False
    try:
        doc_ref = db.collection("user_profiles").document(user_key)
        doc_ref.update({f"profiles.{profile_name}": profile_data})
        return True
    except Exception as e:
        st.error(f"שגיאה בשמירת פרופיל: {e}")
        return False

def delete_profile_from_db(db, user_key, profile_name):
    if db is None: return False
    try:
        doc_ref = db.collection("user_profiles").document(user_key)
        doc_ref.update({f"profiles.{profile_name}": firestore.DELETE_FIELD})
        return True
    except Exception as e:
        st.error(f"שגיאה במחיקת פרופיל: {e}")
        return False

# --- Built-in Profiles ---
BUILT_IN_PROFILES = {
    "ברירת מחדל (שמרני)": {
        "min_stock_price": 20, "max_stock_price": 70, "max_pe_ratio": 40,
        "min_avg_daily_volume": 2000000, "min_iv_threshold": 0.30,
        "min_dte": 30, "max_dte": 60, "target_delta_directional": 0.30,
        "target_delta_neutral": 0.15, "spread_width": 5.0
    },
}

# --- Login Function ---
def check_access_key():
    st.header("🔑 כניסה לשולחן העבודה האישי")
    
    def key_entered():
        user_key = st.session_state["access_key"]
        # ### מתוקן: קריאה נכונה מתוך קטגוריית הסודות ###
        app_secrets = st.secrets.get("app_secrets", {})
        valid_keys = app_secrets.get("VALID_KEYS", [])

        if user_key in valid_keys:
            st.session_state["key_correct"] = True
            st.session_state["user_key"] = user_key
            del st.session_state["access_key"]
        else:
            st.session_state["key_correct"] = False

    if st.session_state.get("key_correct", False):
        return True

    st.text_input(
        "נא להזין מפתח גישה אישי:", type="password", on_change=key_entered, key="access_key"
    )
    if "key_correct" in st.session_state and not st.session_state.key_correct:
        st.error("😕 מפתח הגישה שגוי.")
    return False

# --- Main App Function ---
def run_app():
    db = init_firestore_connection()
    user_key = st.session_state.get("user_key")

    if db is None or user_key is None:
        st.stop()

    st.title("שולחן העבודה של מנהל התיק - אוטונומי")
    st.sidebar.header("הגדרות סריקה וסינון")
    
    custom_profiles = load_profiles_from_db(db, user_key)
    all_profiles = {**BUILT_IN_PROFILES, **custom_profiles}

    if 'selected_profile_name' not in st.session_state or st.session_state.selected_profile_name not in all_profiles:
        st.session_state.selected_profile_name = "ברירת מחדל (שמרני)"

    def on_profile_change():
        st.session_state.selected_profile_name = st.session_state.profile_selector

    selected_profile_name = st.sidebar.selectbox(
        "בחר פרופיל סינון:", options=list(all_profiles.keys()),
        key='profile_selector', on_change=on_profile_change,
        index=list(all_profiles.keys()).index(st.session_state.selected_profile_name)
    )
    selected_profile = all_profiles[selected_profile_name]

    st.sidebar.subheader("קריטריונים מותאמים אישית")
    current_criteria = {}
    param_labels = {
        "min_stock_price": "מחיר מניה מינימלי", "max_stock_price": "מחיר מניה מקסימלי",
        "max_pe_ratio": "יחס P/E מקסימלי", "min_avg_daily_volume": "ווליום יומי ממוצע מינימלי",
        "min_iv_threshold": "סף IV גלום מינימלי", "min_dte": "DTE מינימלי", "max_dte": "DTE מקסימלי",
        "target_delta_directional": "דלתא יעד כיוונית", "target_delta_neutral": "דלתא יעד ניטרלית",
        "spread_width": "רוחב המרווח"
    }
    for key, value in selected_profile.items():
        label = f"{param_labels.get(key, key.replace('_', ' ').title())}"
        if "price" in key or "width" in key:
            current_criteria[key] = st.sidebar.number_input(f"{label} ($):", min_value=0.0, value=float(value), step=0.5, key=f"criteria_{key}")
        elif "volume" in key:
            current_criteria[key] = st.sidebar.number_input(f"{label}:", min_value=0, value=int(value), step=100_000, format="%d", key=f"criteria_{key}")
        elif "pe_ratio" in key or "dte" in key:
            current_criteria[key] = st.sidebar.number_input(f"{label}:", min_value=1, value=int(value), step=1, key=f"criteria_{key}")
        elif "delta" in key or "iv" in key:
            current_criteria[key] = st.sidebar.slider(f"{label}:", 0.0, 1.5 if "iv" in key else 0.5, float(value), 0.01, "%.2f", key=f"criteria_{key}")

    st.sidebar.subheader("ניהול פרופילים")
    new_profile_name = st.sidebar.text_input("שם פרופיל לשמירה/עדכון:", key="new_profile_name_input")
    if st.sidebar.button("💾 שמור פרופיל נוכחי"):
        if new_profile_name:
            if save_profile_to_db(db, user_key, new_profile_name, current_criteria):
                st.sidebar.success(f"פרופיל '{new_profile_name}' נשמר!")
                st.experimental_rerun()
        else:
            st.sidebar.warning("יש לתת שם לפרופיל לפני השמירה.")

    deletable_profiles = list(custom_profiles.keys())
    if deletable_profiles:
        profile_to_delete = st.sidebar.selectbox("בחר פרופיל למחיקה:", options=deletable_profiles)
        if st.sidebar.button("🗑️ מחק פרופיל נבחר", type="primary"):
            if delete_profile_from_db(db, user_key, profile_to_delete):
                st.sidebar.success(f"פרופיל '{profile_to_delete}' נמחק!")
                if st.session_state.selected_profile_name == profile_to_delete:
                    st.session_state.selected_profile_name = "ברירת מחדל (שמרני)"
                st.experimental_rerun()
    
    # ... (The rest of the app logic, like analysis, remains the same)
    # ... (It has been omitted here for brevity but should be in your actual file)

# --- App Entry Point ---
st.set_page_config(layout="wide", page_title="שולחן העבודה של מנהל התיק")
try:
    with open('style.css') as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError: pass

if check_access_key():
    run_app()
