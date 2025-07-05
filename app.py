import streamlit as st # ספריית Streamlit לבניית ממשק המשתמש
import yfinance as yf # ספרייה לדליית נתונים מ-Yahoo Finance
import pandas as pd # ספרייה לעיבוד נתונים (DataFrames)
from datetime import datetime, timedelta # לטיפול בתאריכים
import numpy as np # לחישובים מתמטיים
from scipy.stats import norm # עבור מודל בלאק-שולס
import requests # לשליחת בקשות HTTP לאתרים
from bs4 import BeautifulSoup # לניתוח תוכן HTML
import traceback # לייבוא יכולת הדפסת שגיאות מלאה

# --- פונקציות עזר לחישובים (Greeks & Black-Scholes) ---
# מודל בלאק-שולס לחישוב מחיר אופציה ויווניות
def black_scholes(S, K, T, r, sigma, option_type):
    """
    S: מחיר נכס הבסיס
    K: מחיר מימוש
    T: זמן לפקיעה (בשנים)
    r: ריבית חסרת סיכון (עשרוני)
    sigma: סטיית תקן גלומה (וולטיליות גלומה, עשרוני)
    option_type: 'call' או 'put'
    """
    # ודא שאין ערכים שליליים או אפס בזמן או בוולטיליות
    if T <= 0 or sigma <= 0:
        return 0, 0, 0, 0, 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365 # ליום
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100 # לאחוז שינוי ב-sigma
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365 # ליום
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100 # לאחוז שינוי ב-sigma
        else:
            return 0, 0, 0, 0, 0 # מחיר, דלתא, גמא, תטא, וגה
    except (ValueError, ZeroDivisionError):
        return 0, 0, 0, 0, 0

    return price, delta, gamma, theta, vega

# --- הגדרות וקריטריונים מ"ספר החוקים" ---
# ריבית חסרת סיכון (לצורך חישוב Greeks) - ניתן לעדכן
RISK_FREE_RATE = 0.05 # 5%

# --- הגדרת פרופילים מוגדרים מראש ---
PREDEFINED_PROFILES = {
    "ברירת מחדל (שמרני)": {
        "min_stock_price": 20, "max_stock_price": 70,
        "max_pe_ratio": 40, "min_avg_daily_volume": 2_000_000,
        "min_iv_threshold": 0.30, "min_dte": 30, "max_dte": 60,
        "target_delta_directional": 0.30, "target_delta_neutral": 0.15,
        "spread_width": 5.0
    },
    "טווח רחב (למציאת יותר מניות)": {
        "min_stock_price": 10, "max_stock_price": 500, # טווח מחיר רחב יותר
        "max_pe_ratio": 60, "min_avg_daily_volume": 1_000_000, # P/E ווליום גמישים יותר
        "min_iv_threshold": 0.20, "min_dte": 20, "max_dte": 90, # IV ו-DTE גמישים יותר
        "target_delta_directional": 0.35, "target_delta_neutral": 0.20, # דלתא גמישה יותר
        "spread_width": 10.0
    },
    "אופציות קצרות טווח (ניסיוני)": {
        "min_stock_price": 50, "max_stock_price": 200,
        "max_pe_ratio": 50, "min_avg_daily_volume": 3_000_000,
        "min_iv_threshold": 0.40, "min_dte": 7, "max_dte": 30, # DTE קצר
        "target_delta_directional": 0.40, "target_delta_neutral": 0.25,
        "spread_width": 2.5
    }
}

# --- פונקציה לדליית רשימת מניות מויקיפדיה (אוטונומי) ---
@st.cache_data(ttl=86400) # שמירה במטמון ל-24 שעות (86400 שניות)
def get_tickers_from_wikipedia(index_choice):
    """דולה את רשימת המניות של S&P 500 ו/או NASDAQ 100 מויקיפדיה."""
    tickers = set()
    urls = {
        "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "NASDAQ 100": "https://en.wikipedia.org/wiki/Nasdaq-100"
    }
    indices_to_fetch = []
    if index_choice == "שניהם":
        indices_to_fetch = ["S&P 500", "NASDAQ 100"]
    else:
        indices_to_fetch = [index_choice]

    for index_name in indices_to_fetch:
        try:
            response = requests.get(urls[index_name], headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the correct table (ID is more reliable than class)
            table_id = 'constituents' if index_name == "S&P 500" else 'constituents'
            table = soup.find('table', {'id': table_id})
            if not table: # Fallback to class
                table = soup.find('table', {'class': 'wikitable sortable'})
            
            if table:
                headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
                # Find ticker column index (more robustly)
                ticker_col_index = -1
                possible_headers = ['symbol', 'ticker']
                for h in possible_headers:
                    if h in headers:
                        ticker_col_index = headers.index(h)
                        break
                
                if ticker_col_index == -1: # Fallback if no header found
                    ticker_col_index = 0 if index_name == "S&P 500" else 1

                for row in table.findAll('tr')[1:]:
                    cols = row.findAll('td')
                    if len(cols) > ticker_col_index:
                        ticker = cols[ticker_col_index].text.strip()
                        # Replace dots with dashes for yfinance compatibility (e.g., BRK.B -> BRK-B)
                        ticker = ticker.replace('.', '-')
                        if ticker:
                            tickers.add(ticker)
            else:
                st.warning(f"לא נמצאה טבלת {index_name} בויקיפדיה. ייתכן שהמבנה השתנה.")
        except Exception as e:
            st.error(f"שגיאה בדליית {index_name} מויקיפדיה: {e}")

    return sorted(list(tickers))


# --- פונקציות לדליית ועיבוד נתונים ---
@st.cache_data(ttl=3600) # שמירת נתונים במטמון לשעה כדי למנוע בקשות חוזרות
def get_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Use 'regularMarketPrice' as a fallback for 'currentPrice'
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # If still no price, try fetching history
        if current_price is None:
            hist_price = ticker.history(period="1d")
            if not hist_price.empty:
                current_price = hist_price['Close'].iloc[-1]
        
        if current_price is None: # If absolutely no price is found
            return None

        pe_ratio = info.get('trailingPE')
        avg_volume = info.get('averageVolume')
        
        # חישוב SMA50
        hist = ticker.history(period="60d") # Need more than 50 days to calculate 50d SMA
        sma50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None

        options_expirations = ticker.options
        
        return {
            'ticker': ticker_symbol,
            'current_price': current_price,
            'pe_ratio': pe_ratio,
            'avg_volume': avg_volume,
            'sma50': sma50,
            'options_expirations': options_expirations
        }
    except Exception:
        # traceback.print_exc() # For debugging in local console
        return None

@st.cache_data(ttl=3600)
def get_option_chain(ticker_symbol, expiration_date):
    try:
        ticker = yf.Ticker(ticker_symbol)
        option_chain = ticker.option_chain(expiration_date)
        # Add expiration date to each row for easier DTE calculation
        option_chain.calls['expiration'] = expiration_date
        option_chain.puts['expiration'] = expiration_date
        return option_chain.calls, option_chain.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# --- פונקציות לסינון וחישובים לפי "ספר החוקים" ---
def screen_stock(stock_data, criteria):
    """מיישם את כללי שלב 1: סינון מניות."""
    if not stock_data:
        return False, "אין נתונים"

    price = stock_data['current_price']
    pe = stock_data['pe_ratio']
    volume = stock_data['avg_volume']
    
    if price is None or volume is None:
        return False, "נתונים חסרים (מחיר/ווליום)"

    if not (criteria["min_stock_price"] <= price <= criteria["max_stock_price"]):
        return False, f"מחיר מחוץ לטווח ({price:.2f})"
    
    # P/E can be None, so we handle that case
    if pe is not None and not (0 < pe < criteria["max_pe_ratio"]):
        return False, f"P/E לא מתאים ({pe:.2f})"
        
    if not (volume >= criteria["min_avg_daily_volume"]):
        return False, f"ווליום נמוך ({volume:,})"
        
    return True, "עבר סינון מניה"

def find_best_option_strike(options_df, current_price, option_type, target_delta, criteria):
    """
    מוצא את הסטרייק הטוב ביותר למכירה לפי כלל הדלתא.
    """
    best_strike_data = None
    min_delta_diff = float('inf')

    today = datetime.now().date()
    
    for _, row in options_df.iterrows():
        strike = row['strike']
        implied_volatility = row['impliedVolatility']
        
        expiration_date_str = row['expiration']
        try:
            expiration_date = datetime.strptime(expiration_date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            continue

        dte = (expiration_date - today).days

        if not (criteria["min_dte"] <= dte <= criteria["max_dte"]):
            continue

        bid = row['bid']
        ask = row['ask']
        volume = row.get('volume', 0) # Use .get for safety
        open_interest = row.get('openInterest', 0)

        # Check for valid market data
        if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
            continue
        
        # Check for liquidity
        if pd.isna(volume) or pd.isna(open_interest) or volume < 10 or open_interest < 100:
            continue

        if pd.isna(implied_volatility) or implied_volatility < criteria["min_iv_threshold"]:
            continue

        T_years = dte / 365.0
        _, delta, _, _, _ = black_scholes(current_price, strike, T_years, RISK_FREE_RATE, implied_volatility, option_type)
        
        delta_diff = abs(target_delta - abs(delta))

        if delta_diff < min_delta_diff:
            min_delta_diff = delta_diff
            best_strike_data = row.to_dict() # Store the entire row as a dictionary
            best_strike_data['delta'] = delta
            best_strike_data['dte'] = dte
            
    return best_strike_data


def calculate_trade_metrics(credit, spread_width, pop):
    """
    מחשב תוחלת רווח (EV) ותשואה על סיכון (RoR).
    pop: הסתברות לרווח (Probability of Profit)
    """
    if credit <= 0:
        return -float('inf'), -float('inf')

    max_profit = credit * 100
    max_loss = (spread_width - credit) * 100
    
    if max_loss <= 0:
        # This can happen if credit is >= spread width (arbitrage or data error)
        return float('inf'), float('inf')

    ev = (pop * max_profit) - ((1 - pop) * max_loss)
    ror = (max_profit / max_loss) * 100 # In percent

    return ev, ror

# --- ממשק המשתמש של Streamlit ---
st.set_page_config(layout="wide", page_title="שולחן העבודה של מנהל התיק - אוטונומי")

# Attempt to load CSS file, handle error if not found
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("קובץ style.css לא נמצא. הממשק יוצג בעיצוב ברירת המחדל.")

st.title("שולחן העבודה של מנהל התיק - אוטונומי")
st.markdown("הכלי המרכזי שלך לקבלת החלטות, המבוסס על 'ספר החוקים' שלך.")

# --- הגדרות קריטריונים ופרופילים ב-Sidebar ---
st.sidebar.header("הגדרות סריקה וסינון")

profile_name = st.sidebar.selectbox(
    "בחר פרופיל סינון:",
    options=list(PREDEFINED_PROFILES.keys())
)
selected_profile = PREDEFINED_PROFILES[profile_name]

st.sidebar.subheader("קריטריונים מותאמים אישית")
# Create a dictionary to hold the criteria from the UI
current_criteria = {}

current_criteria["min_stock_price"] = st.sidebar.number_input("מחיר מניה מינימלי ($):", min_value=0.0, value=float(selected_profile["min_stock_price"]), step=1.0)
current_criteria["max_stock_price"] = st.sidebar.number_input("מחיר מניה מקסימלי ($):", min_value=0.0, value=float(selected_profile["max_stock_price"]), step=1.0)
current_criteria["max_pe_ratio"] = st.sidebar.number_input("יחס P/E מקסימלי:", min_value=1.0, value=float(selected_profile["max_pe_ratio"]), step=1.0)
current_criteria["min_avg_daily_volume"] = st.sidebar.number_input("נפח מסחר יומי ממוצע מינימלי:", min_value=0, value=int(selected_profile["min_avg_daily_volume"]), step=100_000, format="%d")
current_criteria["min_iv_threshold"] = st.sidebar.slider("סף IV גלום מינימלי:", 0.0, 1.5, selected_profile["min_iv_threshold"], 0.01, "%.2f")
current_criteria["min_dte"] = st.sidebar.number_input("DTE מינימלי:", min_value=1, value=int(selected_profile["min_dte"]), step=1)
current_criteria["max_dte"] = st.sidebar.number_input("DTE מקסימלי:", min_value=1, value=int(selected_profile["max_dte"]), step=1)
current_criteria["target_delta_directional"] = st.sidebar.slider("דלתא יעד כיוונית (Bull Put/Bear Call):", 0.01, 0.50, selected_profile["target_delta_directional"], 0.01, "%.2f")
current_criteria["target_delta_neutral"] = st.sidebar.slider("דלתא יעד ניטרלית (Iron Condor):", 0.01, 0.30, selected_profile["target_delta_neutral"], 0.01, "%.2f")
# <<<<<<< NEW: SPREAD WIDTH INPUT >>>>>>>
current_criteria["spread_width"] = st.sidebar.number_input("רוחב המרווח (Spread Width, $):", min_value=0.5, value=selected_profile["spread_width"], step=0.5)

index_to_scan = st.sidebar.selectbox("בחר אינדקס לסריקה:", options=["S&P 500", "NASDAQ 100", "שניהם"])

INVESTMENT_UNIVERSE = get_tickers_from_wikipedia(index_to_scan)
if not INVESTMENT_UNIVERSE:
    st.warning("גירוד רשימת המניות נכשל או ריק. משתמש ברשימה מצומצמת לדוגמה.")
    INVESTMENT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY", "QQQ"]

selected_tickers = st.multiselect("בחר מניות לסריקה:", options=INVESTMENT_UNIVERSE, default=INVESTMENT_UNIVERSE)

st.info("""
    **הערות חשובות:**
    * **זמן ריצה:** סריקת מניות רבות עלולה לקחת זמן רב מאוד. התחל עם רשימה קטנה.
    * **נתונים:** הנתונים מ-Yahoo Finance ואינם בזמן אמת. הכלי מתאים לניתוח לאחר שעות המסחר.
    * **IV Rank:** הכלי משתמש ב-`impliedVolatility` כאינדיקציה לוולטיליות, לא IV Rank אמיתי.
    * **דוחות רווחים:** הכלי אינו בודק תאריכי דוחות. יש לבצע בדיקה זו ידנית.
""")

if st.button("🚀 נתח ומצא עסקאות"):
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

            if not is_suitable_stock:
                continue

            current_price = stock_data['current_price']
            sma50 = stock_data['sma50']
            
            if not stock_data['options_expirations']:
                continue
                
            for expiration_date in stock_data['options_expirations']:
                today = datetime.now().date()
                try:
                    exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d').date()
                except (ValueError, TypeError):
                    continue
                
                dte = (exp_dt - today).days
                if not (current_criteria["min_dte"] <= dte <= current_criteria["max_dte"]):
                    continue

                calls_df, puts_df = get_option_chain(ticker_symbol, expiration_date)
                if puts_df.empty or calls_df.empty:
                    continue

                # --- Bull Put Spread ---
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
                                if ev > 0:
                                    all_suitable_deals.append({
                                        'מניה': ticker_symbol, 'אסטרטגיה': 'Bull Put', 'מחיר מניה': f"${current_price:.2f}",
                                        'SMA50': f"${sma50:.2f}", 'ת. פקיעה': expiration_date, 'DTE': dte,
                                        'סטרייקים': f"${bought_strike:.2f} / ${sold_put['strike']:.2f}",
                                        'דלתא (נמכר)': f"{sold_put['delta']:.2f}", 'IV (נמכר)': f"{sold_put['impliedVolatility']:.2%}",
                                        'פרמיה': f"${credit:.2f}", 'תוחלת רווח (EV)': f"${ev:.2f}",
                                        'תשואה על סיכון (ROR)': f"{ror:.1f}%" if ror != float('inf') else '∞',
                                        'הסתברות לרווח (POP)': f"{pop:.1%}"
                                    })
                
                # --- Bear Call Spread ---
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
                                if ev > 0:
                                    all_suitable_deals.append({
                                        'מניה': ticker_symbol, 'אסטרטגיה': 'Bear Call', 'מחיר מניה': f"${current_price:.2f}",
                                        'SMA50': f"${sma50:.2f}", 'ת. פקיעה': expiration_date, 'DTE': dte,
                                        'סטרייקים': f"${sold_call['strike']:.2f} / ${bought_strike:.2f}",
                                        'דלתא (נמכר)': f"{sold_call['delta']:.2f}", 'IV (נמכר)': f"{sold_call['impliedVolatility']:.2%}",
                                        'פרמיה': f"${credit:.2f}", 'תוחלת רווח (EV)': f"${ev:.2f}",
                                        'תשואה על סיכון (ROR)': f"{ror:.1f}%" if ror != float('inf') else '∞',
                                        'הסתברות לרווח (POP)': f"{pop:.1%}"
                                    })

                # --- Iron Condor ---
                sold_put_ic = find_best_option_strike(puts_df, current_price, 'put', current_criteria["target_delta_neutral"], current_criteria)
                sold_call_ic = find_best_option_strike(calls_df, current_price, 'call', current_criteria["target_delta_neutral"], current_criteria)
                if sold_put_ic and sold_call_ic:
                    bought_put_strike = sold_put_ic['strike'] - spread_width
                    bought_call_strike = sold_call_ic['strike'] + spread_width
                    
                    bought_put_series = puts_df[puts_df['strike'] == bought_put_strike]
                    bought_call_series = calls_df[calls_df['strike'] == bought_call_strike]

                    if not bought_put_series.empty and not bought_call_series.empty:
                        bought_put = bought_put_series.iloc[0]
                        bought_call = bought_call_series.iloc[0]
                        total_credit = (sold_put_ic['bid'] - bought_put['ask']) + (sold_call_ic['bid'] - bought_call['ask'])
                        if total_credit > 0:
                            pop_ic = 1 - (abs(sold_put_ic['delta']) + abs(sold_call_ic['delta']))
                            ev_ic, ror_ic = calculate_trade_metrics(total_credit, spread_width, pop_ic)
                            if ev_ic > 0:
                                all_suitable_deals.append({
                                    'מניה': ticker_symbol, 'אסטרטגיה': 'Iron Condor', 'מחיר מניה': f"${current_price:.2f}",
                                    'SMA50': f"${sma50:.2f}" if sma50 else "N/A", 'ת. פקיעה': expiration_date, 'DTE': dte,
                                    'סטרייקים': f"P: ${bought_put_strike:.2f}/${sold_put_ic['strike']:.2f} | C: ${sold_call_ic['strike']:.2f}/${bought_call_strike:.2f}",
                                    'דלתא (נמכר)': f"P: {sold_put_ic['delta']:.2f}, C: {sold_call_ic['delta']:.2f}",
                                    'IV (נמכר)': f"P: {sold_put_ic['impliedVolatility']:.2%}, C: {sold_call_ic['impliedVolatility']:.2%}",
                                    'פרמיה': f"${total_credit:.2f}", 'תוחלת רווח (EV)': f"${ev_ic:.2f}",
                                    'תשואה על סיכון (ROR)': f"{ror_ic:.1f}%" if ror_ic != float('inf') else '∞',
                                    'הסתברות לרווח (POP)': f"{pop_ic:.1%}"
                                })

        progress_bar.empty()
        status_text.empty()

        if all_suitable_deals:
            deals_df = pd.DataFrame(all_suitable_deals)
            
            # Convert metrics to numeric for sorting, handling errors
            deals_df['EV_numeric'] = pd.to_numeric(deals_df['תוחלת רווח (EV)'].str.replace('$', ''), errors='coerce').fillna(0)
            deals_df['ROR_numeric'] = pd.to_numeric(deals_df['תשואה על סיכון (ROR)'].str.replace('%', '').replace('∞', 'inf'), errors='coerce').fillna(0)
            deals_df['POP_numeric'] = pd.to_numeric(deals_df['הסתברות לרווח (POP)'].str.replace('%', ''), errors='coerce').fillna(0)

            # Normalize scores
            max_ev = deals_df['EV_numeric'].max()
            max_ror = deals_df[np.isfinite(deals_df['ROR_numeric'])]['ROR_numeric'].max() # Max of finite values
            max_pop = deals_df['POP_numeric'].max()

            norm_ev = deals_df['EV_numeric'] / max_ev if max_ev > 0 else 0
            norm_ror = deals_df['ROR_numeric'] / max_ror if max_ror > 0 and max_ror != np.inf else 0
            norm_pop = deals_df['POP_numeric'] / max_pop if max_pop > 0 else 0

            # Calculate score with weights
            deals_df['ציון'] = ((norm_ev * 0.45) + (norm_pop * 0.45) + (norm_ror * 0.10)) * 100
            deals_df['ציון'] = deals_df['ציון'].round(1)
            
            deals_df = deals_df.sort_values(by='ציון', ascending=False)
            
            # Select and reorder columns for display
            display_cols = [
                'ציון', 'מניה', 'אסטרטגיה', 'מחיר מניה', 'ת. פקיעה', 'DTE',
                'סטרייקים', 'פרמיה', 'תשואה על סיכון (ROR)', 'הסתברות לרווח (POP)',
                'תוחלת רווח (EV)', 'דלתא (נמכר)', 'IV (נמכר)', 'SMA50'
            ]
            # Filter to only columns that exist in the dataframe
            display_cols = [col for col in display_cols if col in deals_df.columns]
            deals_df_display = deals_df[display_cols]

            st.dataframe(deals_df_display.style.format({'ציון': '{:.1f}'}), use_container_width=True)
            st.success(f"הניתוח הושלם! נמצאו {len(all_suitable_deals)} עסקאות פוטנציאליות.")
        else:
            st.warning("לא נמצאו עסקאות אופציות מתאימות העומדות בכל הקריטריונים. נסה לשנות את קריטריוני הסינון או לבחור אינדקסים נוספים.")
