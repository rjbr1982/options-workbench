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

# --- Firestore Initialization & Profile Management ---
@st.cache_resource
def init_firestore_connection():
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_json)
        db = firestore.Client(credentials=creds)
        return db
    except Exception as e:
        st.error(f"שגיאה קריטית בהתחברות למסד הנתונים. ודא שהגדרות הסודות נכונות. שגיאה: {e}")
        return None

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
    except Exception:
        return {}

def save_profile_to_db(db, user_key, profile_name, profile_data):
    if db is None: return False
    try:
        doc_ref = db.collection("user_profiles").document(user_key)
        doc_ref.update({f"profiles.{profile_name}": profile_data})
        return True
    except Exception:
        return False

def delete_profile_from_db(db, user_key, profile_name):
    if db is None: return False
    try:
        doc_ref = db.collection("user_profiles").document(user_key)
        doc_ref.update({f"profiles.{profile_name}": firestore.DELETE_FIELD})
        return True
    except Exception:
        return False

# --- Built-in Profiles ---
BUILT_IN_PROFILES = {
    "ברירת מחדל (שמרני)": {
        "min_stock_price": 20.0, "max_stock_price": 70.0, "max_pe_ratio": 40.0,
        "min_avg_daily_volume": 2000000, "min_iv_threshold": 0.30,
        "min_dte": 30, "max_dte": 60, "target_delta_directional": 0.30,
        "target_delta_neutral": 0.15, "spread_width": 5.0
    },
     "טווח רחב (למציאת יותר מניות)": {
        "min_stock_price": 10.0, "max_stock_price": 500.0, "max_pe_ratio": 60.0,
        "min_avg_daily_volume": 1000000, "min_iv_threshold": 0.20,
        "min_dte": 20, "max_dte": 90, "target_delta_directional": 0.35,
        "target_delta_neutral": 0.20, "spread_width": 10.0
    },
}

# --- Login Function ---
def check_access_key():
    if "key_correct" not in st.session_state:
        st.session_state.key_correct = False

    if st.session_state.key_correct:
        return True

    st.header("🔑 כניסה לשולחן העבודה האישי")
    user_key_input = st.text_input("נא להזין מפתח גישה אישי:", type="password", key="access_key_input")

    if st.button("כניסה"):
        app_secrets = st.secrets.get("app_secrets", {})
        valid_keys = app_secrets.get("VALID_KEYS", [])
        if user_key_input in valid_keys:
            st.session_state.key_correct = True
            st.session_state.user_key = user_key_input
            st.experimental_rerun()
        else:
            st.error("😕 מפתח הגישה שגוי.")
    return False

# --- Core Logic Functions ---
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

# --- Main App Function ---
def run_app():
    db = init_firestore_connection()
    user_key = st.session_state.get("user_key")

    if db is None or user_key is None:
        st.error("לא ניתן להתחבר למסד הנתונים. בדוק את הגדרות האפליקציה.")
        st.stop()

    st.title("שולחן העבודה של מנהל התיק - אוטונומי")
    st.sidebar.header("הגדרות סריקה וסינון")
    
    # --- Profile Management UI ---
    custom_profiles = load_profiles_from_db(db, user_key)
    all_profiles = {**BUILT_IN_PROFILES, **custom_profiles}
    profile_names = list(all_profiles.keys())
    
    if "selected_profile_name" not in st.session_state or st.session_state.selected_profile_name not in profile_names:
        st.session_state.selected_profile_name = profile_names[0]

    # Use the index to manage the selectbox state reliably
    try:
        current_index = profile_names.index(st.session_state.selected_profile_name)
    except ValueError:
        current_index = 0
        st.session_state.selected_profile_name = profile_names[current_index]

    selected_profile_name = st.sidebar.selectbox(
        "בחר פרופיל סינון:",
        options=profile_names,
        index=current_index,
        key="profile_selector"
    )
    st.session_state.selected_profile_name = selected_profile_name
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
                # No rerun needed, the state will update on the next natural interaction
        else:
            st.sidebar.warning("יש לתת שם לפרופיל לפני השמירה.")

    deletable_profiles = list(custom_profiles.keys())
    if deletable_profiles:
        profile_to_delete = st.sidebar.selectbox("בחר פרופיל למחיקה:", options=deletable_profiles, key="delete_selector")
        if st.sidebar.button("🗑️ מחק פרופיל נבחר", type="primary"):
            if delete_profile_from_db(db, user_key, profile_to_delete):
                st.sidebar.success(f"פרופיל '{profile_to_delete}' נמחק!")
                if st.session_state.selected_profile_name == profile_to_delete:
                    st.session_state.selected_profile_name = "ברירת מחדל (שמרני)"
                st.experimental_rerun() # Rerun is needed here to force update the list
    
    # --- Main Page Content ---
    index_to_scan = st.selectbox("בחר אינדקס לסריקה:", options=["S&P 500", "NASDAQ 100", "שניהם"], index=2)
    INVESTMENT_UNIVERSE = get_tickers_from_wikipedia(index_to_scan)
    if not INVESTMENT_UNIVERSE:
        st.warning("גירוד המניות נכשל. משתמש ברשימת דוגמה.")
        INVESTMENT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY", "QQQ"]
    selected_tickers = st.multiselect("בחר מניות לסריקה:", options=INVESTMENT_UNIVERSE, default=INVESTMENT_UNIVERSE)
    st.info("""**הערות חשובות:** הנתונים אינם בזמן אמת. הכלי אינו בודק דוחות רווחים.""")

    if st.button("🚀 נתח ומצא עסקאות"):
        # ... Full analysis logic ...
        if not selected_tickers:
            st.warning("אנא בחר לפחות מניה אחת לסריקה.")
        else:
            st.subheader("תוצאות ניתוח")
            all_suitable_deals = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            spread_width = current_criteria["spread_width"]

            for i, ticker_symbol in enumerate(selected_tickers):
                status_text.text(f"סורק מניה: {ticker_symbol} ({i+1}/{len(selected_tickers)})")
                progress_bar.progress((i + 1) / len(selected_tickers))
                stock_data = get_stock_data(ticker_symbol)
                is_suitable_stock = screen_stock(stock_data, current_criteria)
                if not is_suitable_stock: continue
                current_price, sma50 = stock_data['current_price'], stock_data['sma50']
                if not stock_data['options_expirations']: continue
                for expiration_date in stock_data['options_expirations']:
                    try:
                        exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d').date()
                        dte = (exp_dt - datetime.now().date()).days
                        if not (current_criteria["min_dte"] <= dte <= current_criteria["max_dte"]): continue
                    except (ValueError, TypeError): continue
                    calls_df, puts_df = get_option_chain(ticker_symbol, expiration_date)
                    if puts_df.empty or calls_df.empty: continue
                    
                    if sma50 and current_price > sma50:
                        sold_put = find_best_option_strike(puts_df, current_price, 'put', current_criteria["target_delta_directional"], current_criteria)
                        if sold_put:
                            bought_strike = sold_put['strike'] - spread_width
                            bought_put_series = puts_df[puts_df['strike'] == bought_strike]
                            if not bought_put_series.empty:
                                bought_put = bought_put_series.iloc[0]
                                credit = sold_put['bid'] - bought_put['ask']
                                if credit > 0:
                                    pop = 1 - abs(sold_put['delta'])
                                    ev, ror = calculate_trade_metrics(credit, spread_width, pop)
                                    if ev > 0: all_suitable_deals.append({'מניה': ticker_symbol, 'אסטרטגיה': 'Bull Put', 'מחיר מניה': f"${current_price:.2f}", 'SMA50': f"${sma50:.2f}", 'ת. פקיעה': expiration_date, 'DTE': dte, 'סטרייקים': f"${bought_strike:.2f} / ${sold_put['strike']:.2f}", 'דלתא (נמכר)': f"{sold_put['delta']:.2f}", 'IV (נמכר)': f"{sold_put['impliedVolatility']:.2%}", 'פרמיה': f"${credit:.2f}", 'תוחלת רווח (EV)': f"${ev:.2f}", 'תשואה על סיכון (ROR)': f"{ror:.1f}%" if ror != float('inf') else '∞', 'הסתברות לרווח (POP)': f"{pop:.1%}"})
                    
                    if sma50 and current_price < sma50:
                        sold_call = find_best_option_strike(calls_df, current_price, 'call', current_criteria["target_delta_directional"], current_criteria)
                        if sold_call:
                            bought_strike = sold_call['strike'] + spread_width
                            bought_call_series = calls_df[calls_df['strike'] == bought_strike]
                            if not bought_call_series.empty:
                                bought_call = bought_call_series.iloc[0]
                                credit = sold_call['bid'] - bought_call['ask']
                                if credit > 0:
                                    pop = 1 - abs(sold_call['delta'])
                                    ev, ror = calculate_trade_metrics(credit, spread_width, pop)
                                    if ev > 0: all_suitable_deals.append({'מניה': ticker_symbol, 'אסטרטגיה': 'Bear Call', 'מחיר מניה': f"${current_price:.2f}", 'SMA50': f"${sma50:.2f}", 'ת. פקיעה': expiration_date, 'DTE': dte, 'סטרייקים': f"${sold_call['strike']:.2f} / ${bought_strike:.2f}", 'דלתא (נמכר)': f"{sold_call['delta']:.2f}", 'IV (נמכר)': f"{sold_call['impliedVolatility']:.2%}", 'פרמיה': f"${credit:.2f}", 'תוחלת רווח (EV)': f"${ev:.2f}", 'תשואה על סיכון (ROR)': f"{ror:.1f}%" if ror != float('inf') else '∞', 'הסתברות לרווח (POP)': f"{pop:.1%}"})

            progress_bar.empty()
            status_text.empty()
            if all_suitable_deals:
                deals_df = pd.DataFrame(all_suitable_deals)
                deals_df['EV_numeric'] = pd.to_numeric(deals_df['תוחלת רווח (EV)'].str.replace('$', ''), errors='coerce').fillna(0)
                deals_df['ROR_numeric'] = pd.to_numeric(deals_df['תשואה על סיכון (ROR)'].str.replace('%', '').replace('∞', 'inf'), errors='coerce').fillna(0)
                deals_df['POP_numeric'] = pd.to_numeric(deals_df['הסתברות לרווח (POP)'].str.replace('%', ''), errors='coerce').fillna(0)
                max_ev, max_ror, max_pop = deals_df['EV_numeric'].max(), deals_df[np.isfinite(deals_df['ROR_numeric'])]['ROR_numeric'].max(), deals_df['POP_numeric'].max()
                norm_ev = deals_df['EV_numeric'] / max_ev if max_ev > 0 else 0
                norm_ror = deals_df['ROR_numeric'] / max_ror if max_ror > 0 and max_ror != np.inf else 0
                norm_pop = deals_df['POP_numeric'] / max_pop if max_pop > 0 else 0
                deals_df['ציון'] = ((norm_ev * 0.45) + (norm_pop * 0.45) + (norm_ror * 0.10)) * 100
                deals_df = deals_df.sort_values(by='ציון', ascending=False)
                display_cols = ['ציון', 'מניה', 'אסטרטגיה', 'מחיר מניה', 'ת. פקיעה', 'DTE', 'סטרייקים', 'פרמיה', 'תשואה על סיכון (ROR)', 'הסתברות לרווח (POP)', 'תוחלת רווח (EV)', 'דלתא (נמכר)', 'IV (נמכר)', 'SMA50']
                deals_df_display = deals_df[[col for col in display_cols if col in deals_df.columns]]
                st.dataframe(deals_df_display.style.format({'ציון': '{:.1f}'}), use_container_width=True)
                st.success(f"הניתוח הושלם! נמצאו {len(all_suitable_deals)} עסקאות פוטנציאליות.")
            else:
                st.warning("לא נמצאו עסקאות מתאימות. נסה לשנות את קריטריוני הסינון.")

# --- App Entry Point ---
st.set_page_config(layout="wide", page_title="שולחן העבודה של מנהל התיק")
try:
    with open('style.css') as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError: pass

if check_access_key():
    run_app()
