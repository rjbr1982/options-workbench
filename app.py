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
    """Initializes a connection to Firestore using credentials from st.secrets."""
    try:
        # Get credentials from Streamlit secrets
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
    """Loads profiles for a specific user key from Firestore."""
    if db is None:
        return {}
    try:
        # The user's profiles are stored in a single document named after their key
        doc_ref = db.collection("user_profiles").document(user_key)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("profiles", {})
        else:
            # If the document doesn't exist, create it with an empty profiles map
            doc_ref.set({"profiles": {}})
            return {}
    except Exception as e:
        st.error(f"שגיאה בטעינת פרופילים: {e}")
        return {}

def save_profile_to_db(db, user_key, profile_name, profile_data):
    """Saves or updates a single profile for a user."""
    if db is None:
        return False
    try:
        doc_ref = db.collection("user_profiles").document(user_key)
        # Use FieldPath for keys with spaces or special characters if needed
        # Here we update a map field within the document
        doc_ref.update({f"profiles.{profile_name}": profile_data})
        return True
    except Exception as e:
        st.error(f"שגיאה בשמירת פרופיל: {e}")
        return False

def delete_profile_from_db(db, user_key, profile_name):
    """Deletes a single profile for a user."""
    if db is None:
        return False
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
    # ... other built-in profiles
}

# --- Login Function ---
def check_access_key():
    """Checks for the user's personal access key."""
    st.header("🔑 כניסה לשולחן העבודה האישי")
    
    def key_entered():
        user_key = st.session_state["access_key"]
        valid_keys = st.secrets.get("VALID_KEYS", [])
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

    # --- Sidebar ---
    st.sidebar.header("הגדרות סריקה וסינון")
    
    # Load profiles
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
    # ... (UI for criteria remains the same)
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


    # --- Save Profile Section ---
    st.sidebar.subheader("ניהול פרופילים")
    new_profile_name = st.sidebar.text_input("שם פרופיל לשמירה/עדכון:", key="new_profile_name_input")
    if st.sidebar.button("💾 שמור פרופיל נוכחי"):
        if new_profile_name:
            if save_profile_to_db(db, user_key, new_profile_name, current_criteria):
                st.sidebar.success(f"פרופיל '{new_profile_name}' נשמר!")
                # No rerun needed, state will update naturally on next interaction
        else:
            st.sidebar.warning("יש לתת שם לפרופיל לפני השמירה.")

    # --- Delete Profile Section ---
    deletable_profiles = list(custom_profiles.keys())
    if deletable_profiles:
        profile_to_delete = st.sidebar.selectbox("בחר פרופיל למחיקה:", options=deletable_profiles)
        if st.sidebar.button("🗑️ מחק פרופיל נבחר", type="primary"):
            if delete_profile_from_db(db, user_key, profile_to_delete):
                st.sidebar.success(f"פרופיל '{profile_to_delete}' נמחק!")
                # Reset selection if the deleted profile was selected
                if st.session_state.selected_profile_name == profile_to_delete:
                    st.session_state.selected_profile_name = "ברירת מחדל (שמרני)"
                st.experimental_rerun() # Rerun to update the selectbox
    
    # --- Main Page Content ---
    st.title("שולחן העבודה של מנהל התיק - אוטונומי")
    # ... (Rest of the main page UI and logic remains the same)
    # --- Core Functions (no changes needed) ---
    RISK_FREE_RATE = 0.05
    @st.cache_data
    def black_scholes(S, K, T, r, sigma, option_type):
        if T <= 0 or sigma <= 0: return 0.0
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            if option_type == 'call': delta = norm.cdf(d1)
            elif option_type == 'put': delta = norm.cdf(d1) - 1
            else: delta = 0.0
        except (ValueError, ZeroDivisionError): return 0.0
        return delta
    @st.cache_data(ttl=86400)
    def get_tickers_from_wikipedia(index_choice):
        tickers = set()
        urls = {"S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "NASDAQ 100": "https://en.wikipedia.org/wiki/Nasdaq-100"}
        indices_to_fetch = ["S&P 500", "NASDAQ 100"] if index_choice == "שניהם" else [index_choice]
        for index_name in indices_to_fetch:
            try:
                response = requests.get(urls[index_name], headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', {'id': 'constituents'}) or soup.find('table', {'class': 'wikitable sortable'})
                if table:
                    headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
                    ticker_col_index = headers.index('symbol') if 'symbol' in headers else (headers.index('ticker') if 'ticker' in headers else (0 if index_name == "S&P 500" else 1))
                    for row in table.findAll('tr')[1:]:
                        cols = row.findAll('td')
                        if len(cols) > ticker_col_index:
                            ticker = cols[ticker_col_index].text.strip().replace('.', '-')
                            if ticker: tickers.add(ticker)
            except Exception: pass
        return sorted(list(tickers))
    @st.cache_data(ttl=3600)
    def get_stock_data(ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price is None:
                hist_price = ticker.history(period="1d")
                if not hist_price.empty: current_price = hist_price['Close'].iloc[-1]
            if current_price is None: return None
            pe_ratio = info.get('trailingPE')
            avg_volume = info.get('averageVolume')
            hist = ticker.history(period="60d")
            sma50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
            return {'ticker': ticker_symbol, 'current_price': current_price, 'pe_ratio': pe_ratio, 'avg_volume': avg_volume, 'sma50': sma50, 'options_expirations': ticker.options}
        except Exception: return None
    @st.cache_data(ttl=3600)
    def get_option_chain(ticker_symbol, expiration_date):
        try:
            ticker = yf.Ticker(ticker_symbol)
            opt = ticker.option_chain(expiration_date)
            opt.calls['expiration'], opt.puts['expiration'] = expiration_date, expiration_date
            return opt.calls, opt.puts
        except Exception: return pd.DataFrame(), pd.DataFrame()
    def screen_stock(stock_data, criteria):
        if not stock_data: return False
        price, pe, volume = stock_data['current_price'], stock_data['pe_ratio'], stock_data['avg_volume']
        if price is None or volume is None: return False
        if not (criteria["min_stock_price"] <= price <= criteria["max_stock_price"]): return False
        if pe is not None and not (0 < pe < criteria["max_pe_ratio"]): return False
        if not (volume >= criteria["min_avg_daily_volume"]): return False
        return True
    def find_best_option_strike(options_df, current_price, option_type, target_delta, criteria):
        best_strike_data, min_delta_diff = None, float('inf')
        today = datetime.now().date()
        for _, row in options_df.iterrows():
            try:
                dte = (datetime.strptime(row['expiration'], '%Y-%m-%d').date() - today).days
                if not (criteria["min_dte"] <= dte <= criteria["max_dte"]): continue
                bid, ask, volume, open_interest = row['bid'], row['ask'], row.get('volume', 0), row.get('openInterest', 0)
                if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0 or pd.isna(volume) or pd.isna(open_interest) or volume < 10 or open_interest < 100: continue
                implied_volatility = row['impliedVolatility']
                if pd.isna(implied_volatility) or implied_volatility < criteria["min_iv_threshold"]: continue
                delta = black_scholes(current_price, row['strike'], dte / 365.0, RISK_FREE_RATE, implied_volatility, option_type)
                delta_diff = abs(target_delta - abs(delta))
                if delta_diff < min_delta_diff:
                    min_delta_diff = delta_diff
                    best_strike_data = row.to_dict()
                    best_strike_data['delta'], best_strike_data['dte'] = delta, dte
            except (ValueError, TypeError): continue
        return best_strike_data
    def calculate_trade_metrics(credit, spread_width, pop):
        if credit <= 0: return -float('inf'), -float('inf')
        max_profit = credit * 100
        max_loss = (spread_width - credit) * 100
        if max_loss <= 0: return float('inf'), float('inf')
        ev = (pop * max_profit) - ((1 - pop) * max_loss)
        ror = (max_profit / max_loss) * 100
        return ev, ror

    index_to_scan = st.selectbox("בחר אינדקס לסריקה:", options=["S&P 500", "NASDAQ 100", "שניהם"], index=2)
    INVESTMENT_UNIVERSE = get_tickers_from_wikipedia(index_to_scan)
    if not INVESTMENT_UNIVERSE:
        st.warning("גירוד המניות נכשל. משתמש ברשימת דוגמה.")
        INVESTMENT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY", "QQQ"]
    selected_tickers = st.multiselect("בחר מניות לסריקה:", options=INVESTMENT_UNIVERSE, default=INVESTMENT_UNIVERSE)
    st.info("...הערות חשובות...")
    if st.button("🚀 נתח ומצא עסקאות"):
        # ... Analysis logic ...
        pass # The analysis logic remains the same

# --- App Entry Point ---
st.set_page_config(layout="wide", page_title="שולחן העבודה של מנהל התיק")
try:
    with open('style.css') as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError: pass

if check_access_key():
    run_app()
