import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import requests
from bs4 import BeautifulSoup
import traceback
import json

# --- הגדרות קבועות ---
RISK_FREE_RATE = 0.05
# ### חדש: סיסמת ברירת מחדל אם לא הוגדרה ב-Secrets ###
DEFAULT_PASSWORD = "12345"

# --- הגדרת פרופילים מובנים ---
BUILT_IN_PROFILES = {
    "ברירת מחדל (שמרני)": {
        "min_stock_price": 20, "max_stock_price": 70, "max_pe_ratio": 40,
        "min_avg_daily_volume": 2000000, "min_iv_threshold": 0.30,
        "min_dte": 30, "max_dte": 60, "target_delta_directional": 0.30,
        "target_delta_neutral": 0.15, "spread_width": 5.0
    },
    "טווח רחב (למציאת יותר מניות)": {
        "min_stock_price": 10, "max_stock_price": 500, "max_pe_ratio": 60,
        "min_avg_daily_volume": 1000000, "min_iv_threshold": 0.20,
        "min_dte": 20, "max_dte": 90, "target_delta_directional": 0.35,
        "target_delta_neutral": 0.20, "spread_width": 10.0
    },
    "אופציות קצרות טווח (ניסיוני)": {
        "min_stock_price": 50, "max_stock_price": 200, "max_pe_ratio": 50,
        "min_avg_daily_volume": 3000000, "min_iv_threshold": 0.40,
        "min_dte": 7, "max_dte": 30, "target_delta_directional": 0.40,
        "target_delta_neutral": 0.25, "spread_width": 2.5
    }
}

# ### חדש: פונקציית הגנת סיסמה ###
def check_password():
    """בודקת אם המשתמש הזין את הסיסמה הנכונה. מחזירה True אם כן."""

    # פונקציה שמופעלת עם הזנת הסיסמה
    def password_entered():
        # בודקת את הסיסמה שהוזנה מול הסיסמה ב-Secrets או ברירת המחדל
        if st.session_state["password"] == st.secrets.get("PASSWORD", DEFAULT_PASSWORD):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # מוחקת את הסיסמה מהזיכרון
        else:
            st.session_state["password_correct"] = False

    # אם הסיסמה כבר אושרה בסשן הנוכחי, החזר True
    if st.session_state.get("password_correct", False):
        return True

    # אם לא, הצג את שדה הקלט לסיסמה
    st.text_input(
        "נא להזין סיסמה כדי לגשת לשולחן העבודה", type="password", on_change=password_entered, key="password"
    )
    # אם הוזנה סיסמה שגויה, הצג הודעת שגיאה
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("😕 הסיסמה שהוזנה שגויה. נסה שוב.")
    return False


# ### חדש: כל האפליקציה עטופה בפונקציה אחת ###
def run_app():
    # --- ניהול פרופילים עם st.session_state ---
    if 'custom_profiles' not in st.session_state:
        st.session_state.custom_profiles = {}
    if 'selected_profile_name' not in st.session_state:
        st.session_state.selected_profile_name = "ברירת מחדל (שמרני)"

    def load_custom_profiles():
        return st.session_state.custom_profiles

    def save_custom_profile(name, profile_data):
        st.session_state.custom_profiles[name] = profile_data
        st.success(f"פרופיל '{name}' נשמר בהצלחה!")

    # --- פונקציות חישוב ודליית נתונים (ללא שינוי) ---
    def black_scholes(S, K, T, r, sigma, option_type):
        if T <= 0 or sigma <= 0: return 0, 0, 0, 0, 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        try:
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                delta = norm.cdf(d1)
            elif option_type == 'put':
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                delta = norm.cdf(d1) - 1
            else: return 0, 0, 0, 0, 0
        except (ValueError, ZeroDivisionError): return 0, 0, 0, 0, 0
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'call' else (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        return price, delta, gamma, theta, vega

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
                else: st.warning(f"לא נמצאה טבלת {index_name} בויקיפדיה.")
            except Exception as e: st.error(f"שגיאה בדליית {index_name}: {e}")
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
            options_expirations = ticker.options
            return {'ticker': ticker_symbol, 'current_price': current_price, 'pe_ratio': pe_ratio, 'avg_volume': avg_volume, 'sma50': sma50, 'options_expirations': options_expirations}
        except Exception: return None

    @st.cache_data(ttl=3600)
    def get_option_chain(ticker_symbol, expiration_date):
        try:
            ticker = yf.Ticker(ticker_symbol)
            opt = ticker.option_chain(expiration_date)
            opt.calls['expiration'] = expiration_date
            opt.puts['expiration'] = expiration_date
            return opt.calls, opt.puts
        except Exception: return pd.DataFrame(), pd.DataFrame()

    def screen_stock(stock_data, criteria):
        if not stock_data: return False, "אין נתונים"
        price, pe, volume = stock_data['current_price'], stock_data['pe_ratio'], stock_data['avg_volume']
        if price is None or volume is None: return False, "נתונים חסרים (מחיר/ווליום)"
        if not (criteria["min_stock_price"] <= price <= criteria["max_stock_price"]): return False, f"מחיר מחוץ לטווח ({price:.2f})"
        if pe is not None and not (0 < pe < criteria["max_pe_ratio"]): return False, f"P/E לא מתאים ({pe:.2f})"
        if not (volume >= criteria["min_avg_daily_volume"]): return False, f"ווליום נמוך ({volume:,})"
        return True, "עבר סינון מניה"

    def find_best_option_strike(options_df, current_price, option_type, target_delta, criteria):
        best_strike_data, min_delta_diff = None, float('inf')
        today = datetime.now().date()
        for _, row in options_df.iterrows():
            try:
                expiration_date = datetime.strptime(row['expiration'], '%Y-%m-%d').date()
                dte = (expiration_date - today).days
                if not (criteria["min_dte"] <= dte <= criteria["max_dte"]): continue
                bid, ask, volume, open_interest = row['bid'], row['ask'], row.get('volume', 0), row.get('openInterest', 0)
                if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0 or pd.isna(volume) or pd.isna(open_interest) or volume < 10 or open_interest < 100: continue
                implied_volatility = row['impliedVolatility']
                if pd.isna(implied_volatility) or implied_volatility < criteria["min_iv_threshold"]: continue
                _, delta, _, _, _ = black_scholes(current_price, row['strike'], dte / 365.0, RISK_FREE_RATE, implied_volatility, option_type)
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

    # --- ממשק המשתמש של Streamlit ---
    st.set_page_config(layout="wide", page_title="שולחן העבודה של מנהל התיק - אוטונומי")
    try:
        with open('style.css') as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError: st.warning("קובץ style.css לא נמצא.")

    st.title("שולחן העבודה של מנהל התיק - אוטונומי")
    st.markdown("הכלי המרכזי שלך לקבלת החלטות, המבוסס על 'ספר החוקים' שלך.")

    st.sidebar.header("הגדרות סריקה וסינון")
    
    custom_profiles = load_custom_profiles()
    all_profiles = {**BUILT_IN_PROFILES, **custom_profiles}

    def on_profile_change():
        st.session_state.selected_profile_name = st.session_state.profile_selector

    profile_name = st.sidebar.selectbox(
        "בחר פרופיל סינון:", options=list(all_profiles.keys()),
        key='profile_selector', on_change=on_profile_change
    )

    if st.session_state.selected_profile_name not in all_profiles:
        st.session_state.selected_profile_name = list(all_profiles.keys())[0]

    selected_profile = all_profiles[st.session_state.selected_profile_name]

    st.sidebar.subheader("קריטריונים מותאמים אישית")
    current_criteria = {}
    # לולאה חכמה ליצירת שדות הקלט
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

    st.sidebar.subheader("שמירת פרופיל חדש")
    new_profile_name = st.sidebar.text_input("שם לפרופיל החדש:")
    if st.sidebar.button("💾 שמור פרופיל נוכחי"):
        if new_profile_name:
            if new_profile_name in BUILT_IN_PROFILES:
                st.sidebar.error("לא ניתן לדרוס פרופיל מובנה.")
            else:
                save_custom_profile(new_profile_name, current_criteria)
                st.experimental_rerun()
        else:
            st.sidebar.warning("יש לתת שם לפרופיל לפני השמירה.")

    index_to_scan = st.selectbox("בחר אינדקס לסריקה:", options=["S&P 500", "NASDAQ 100", "שניהם"], index=2)
    INVESTMENT_UNIVERSE = get_tickers_from_wikipedia(index_to_scan)
    if not INVESTMENT_UNIVERSE:
        st.warning("גירוד המניות נכשל. משתמש ברשימת דוגמה.")
        INVESTMENT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY", "QQQ"]
    selected_tickers = st.multiselect("בחר מניות לסריקה:", options=INVESTMENT_UNIVERSE, default=INVESTMENT_UNIVERSE)

    st.info("""...הערות חשובות (כמו קודם)...""")

    if st.button("🚀 נתח ומצא עסקאות"):
        # ... לוגיקת הניתוח נשארת זהה לחלוטין ...
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
                is_suitable_stock, _ = screen_stock(stock_data, current_criteria)
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
                    # Bull Put Spread
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
                    # Bear Call Spread
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
                    # Iron Condor
                    sold_put_ic = find_best_option_strike(puts_df, current_price, 'put', current_criteria["target_delta_neutral"], current_criteria)
                    sold_call_ic = find_best_option_strike(calls_df, current_price, 'call', current_criteria["target_delta_neutral"], current_criteria)
                    if sold_put_ic and sold_call_ic:
                        bought_put_strike, bought_call_strike = sold_put_ic['strike'] - spread_width, sold_call_ic['strike'] + spread_width
                        bought_put_series, bought_call_series = puts_df[puts_df['strike'] == bought_put_strike], calls_df[calls_df['strike'] == bought_call_strike]
                        if not bought_put_series.empty and not bought_call_series.empty:
                            bought_put, bought_call = bought_put_series.iloc[0], bought_call_series.iloc[0]
                            total_credit = (sold_put_ic['bid'] - bought_put['ask']) + (sold_call_ic['bid'] - bought_call['ask'])
                            if total_credit > 0:
                                pop_ic = 1 - (abs(sold_put_ic['delta']) + abs(sold_call_ic['delta']))
                                ev_ic, ror_ic = calculate_trade_metrics(total_credit, spread_width, pop_ic)
                                if ev_ic > 0: all_suitable_deals.append({'מניה': ticker_symbol, 'אסטרטגיה': 'Iron Condor', 'מחיר מניה': f"${current_price:.2f}", 'SMA50': f"${sma50:.2f}" if sma50 else "N/A", 'ת. פקיעה': expiration_date, 'DTE': dte, 'סטרייקים': f"P: ${bought_put_strike:.2f}/${sold_put_ic['strike']:.2f} | C: ${sold_call_ic['strike']:.2f}/${bought_call_strike:.2f}", 'דלתא (נמכר)': f"P: {sold_put_ic['delta']:.2f}, C: {sold_call_ic['delta']:.2f}", 'IV (נמכר)': f"P: {sold_put_ic['impliedVolatility']:.2%}, C: {sold_call_ic['impliedVolatility']:.2%}", 'פרמיה': f"${total_credit:.2f}", 'תוחלת רווח (EV)': f"${ev_ic:.2f}", 'תשואה על סיכון (ROR)': f"{ror_ic:.1f}%" if ror_ic != float('inf') else '∞', 'הסתברות לרווח (POP)': f"{pop_ic:.1%}"})
            
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

# --- נקודת הכניסה לאפליקציה ---
# מריצה את האפליקציה רק אם הסיסמה נכונה
if check_password():
    run_app()
