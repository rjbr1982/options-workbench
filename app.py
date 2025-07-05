import streamlit as st # ספריית Streamlit לבניית ממשק המשתמש
import yfinance as yf # ספרייה לדליית נתונים מ-Yahoo Finance
import pandas as pd # ספרייה לעיבוד נתונים (DataFrames)
from datetime import datetime, timedelta # לטיפול בתאריכים
import numpy as np # לחישובים מתמטיים
from scipy.stats import norm # עבור מודל בלאק-שולס
import requests # לשליחת בקשות HTTP לאתרים
from bs4 import BeautifulSoup # לניתוח תוכן HTML

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
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

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
        "target_delta_directional": 0.30, "target_delta_neutral": 0.15
    },
    "טווח רחב (למציאת יותר מניות)": {
        "min_stock_price": 10, "max_stock_price": 500, # טווח מחיר רחב יותר
        "max_pe_ratio": 60, "min_avg_daily_volume": 1_000_000, # P/E ווליום גמישים יותר
        "min_iv_threshold": 0.20, "min_dte": 20, "max_dte": 90, # IV ו-DTE גמישים יותר
        "target_delta_directional": 0.35, "target_delta_neutral": 0.20 # דלתא גמישה יותר
    },
    "אופציות קצרות טווח (ניסיוני)": {
        "min_stock_price": 50, "max_stock_price": 200,
        "max_pe_ratio": 50, "min_avg_daily_volume": 3_000_000,
        "min_iv_threshold": 0.40, "min_dte": 7, "max_dte": 30, # DTE קצר
        "target_delta_directional": 0.40, "target_delta_neutral": 0.25
    }
}

# --- פונקציה לדליית רשימת מניות מויקיפדיה (אוטונומי) ---
@st.cache_data(ttl=86400) # שמירה במטמון ל-24 שעות (86400 שניות)
def get_tickers_from_wikipedia(index_choice):
    """דולה את רשימת המניות של S&P 500 ו/או NASDAQ 100 מויקיפדיה."""
    tickers = set()

    if index_choice in ["S&P 500", "שניהם"]:
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            response = requests.get(sp500_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable sortable'})
            if table:
                for row in table.findAll('tr')[1:]:
                    ticker = row.findAll('td')[0].text.strip()
                    tickers.add(ticker)
            else:
                st.warning("לא נמצאה טבלת S&P 500 בויקיפדיה. ייתכן שהמבנה השתנה.")
        except Exception as e:
            st.error(f"שגיאה בדליית S&P 500 מויקיפדיה: {e}")

    if index_choice in ["NASDAQ 100", "שניהם"]:
        nasdaq100_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        try:
            response = requests.get(nasdaq100_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable sortable'})
            if table:
                for row in table.findAll('tr')[1:]:
                    try:
                        ticker = row.findAll('td')[1].text.strip() # סמל המניה הוא לרוב בעמודה השנייה
                        tickers.add(ticker)
                    except IndexError:
                        pass
            else:
                st.warning("לא נמצאה טבלת NASDAQ 100 בויקיפדיה. ייתכן שהמבנה השתנה.")
        except Exception as e:
            st.error(f"שגיאה בדליית NASDAQ 100 מויקיפדיה: {e}")

    return sorted(list(tickers))


# --- פונקציות לדליית ועיבוד נתונים ---
@st.cache_data(ttl=3600) # שמירת נתונים במטמון לשעה כדי למנוע בקשות חוזרות
def get_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        hist = ticker.history(period="50d") # עבור SMA50
        
        # נתוני מניה
        current_price = info.get('currentPrice')
        pe_ratio = info.get('trailingPE')
        avg_volume = info.get('averageVolume')
        
        # חישוב SMA50
        sma50 = hist['Close'].rolling(window=50).mean().iloc[-1] if not hist.empty else None

        # נתוני אופציות
        options_expirations = ticker.options
        
        return {
            'ticker': ticker_symbol,
            'current_price': current_price,
            'pe_ratio': pe_ratio,
            'avg_volume': avg_volume,
            'sma50': sma50,
            'options_expirations': options_expirations
        }
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_option_chain(ticker_symbol, expiration_date):
    try:
        ticker = yf.Ticker(ticker_symbol)
        option_chain = ticker.option_chain(expiration_date)
        return option_chain.calls, option_chain.puts
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

# --- פונקציות לסינון וחישובים לפי "ספר החוקים" ---
def screen_stock(stock_data, criteria):
    """מיישם את כללי שלב 1: סינון מניות."""
    if not stock_data:
        return False, "אין נתונים"

    price = stock_data['current_price']
    pe = stock_data['pe_ratio']
    volume = stock_data['avg_volume']
    
    # וודא שכל הנתונים קיימים לפני הבדיקה
    if price is None or pe is None or volume is None:
        return False, "נתונים חסרים (מחיר/PE/ווליום)"

    # טווח מחיר
    if not (criteria["min_stock_price"] <= price <= criteria["max_stock_price"]):
        return False, f"מחיר מחוץ לטווח ({price:.2f})"
    
    # חוזק עסקי (P/E)
    if not (pe > 0 and pe < criteria["max_pe_ratio"]):
        return False, f"P/E לא מתאים ({pe:.2f})"
        
    # נזילות
    if not (volume >= criteria["min_avg_daily_volume"]):
        return False, f"ווליום נמוך ({volume:,})"
        
    return True, "עבר סינון מניה"

def find_best_option_strike(options_df, current_price, option_type, target_delta, criteria):
    """
    מוצא את הסטרייק הטוב ביותר לפי כלל הדלתא הבטוחה.
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
        except ValueError:
            continue

        dte = (expiration_date - today).days

        if not (criteria["min_dte"] <= dte <= criteria["max_dte"]):
            continue

        bid = row['bid']
        ask = row['ask']
        volume = row['volume']
        open_interest = row['openInterest']

        if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0 or volume == 0 or open_interest == 0:
            continue

        if pd.isna(implied_volatility) or implied_volatility <= 0 or implied_volatility < criteria["min_iv_threshold"]:
            continue # מסנן גם לפי IV Threshold

        T_years = dte / 365.0
        if T_years <= 0:
            continue

        _, delta, _, _, _ = black_scholes(current_price, strike, T_years, RISK_FREE_RATE, implied_volatility, option_type)
        
        if option_type == 'call':
            if delta > target_delta:
                continue
            delta_diff = target_delta - delta
        else: # Put
            if delta > 0: # דלתא של Put צריכה להיות שלילית
                continue
            if abs(delta) > target_delta:
                continue
            delta_diff = abs(target_delta - abs(delta))

        if delta_diff < min_delta_diff:
            min_delta_diff = delta_diff
            best_strike_data = {
                'strike': strike,
                'delta': delta,
                'bid': bid,
                'ask': ask,
                'implied_volatility': implied_volatility,
                'volume': volume,
                'open_interest': open_interest,
                'dte': dte,
                'expiration': expiration_date_str
            }
    return best_strike_data


def calculate_trade_metrics(strategy_type, credit, spread_width, pop):
    """
    מחשב תוחלת רווח (EV) ותשואה על סיכון (RoR).
    pop: הסתברות לרווח (Probability of Profit)
    """
    max_profit = credit * 100 # פרמיה מקסימלית * 100 מניות
    max_loss = (spread_width - credit) * 100 # רוחב המרווח - פרמיה * 100 מניות
    
    if max_loss <= 0:
        return -float('inf'), -float('inf')

    ev = (pop * max_profit) - ((1 - pop) * max_loss)
    ror = (max_profit / max_loss) * 100 # באחוזים

    return ev, ror

# --- ממשק המשתמש של Streamlit ---
st.set_page_config(layout="wide", page_title="שולחן העבודה של מנהל התיק - אוטונומי")

# טעינת קובץ ה-CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("שולחן העבודה של מנהל התיק - אוטונומי")
st.markdown("הכלי המרכזי שלך לקבלת החלטות, המבוסס על 'ספר החוקים' שלך.")

# --- הגדרות קריטריונים ופרופילים ב-Sidebar ---
st.sidebar.header("הגדרות סריקה וסינון")

# בחירת פרופיל
profile_name = st.sidebar.selectbox(
    "בחר פרופיל סינון:",
    options=list(PREDEFINED_PROFILES.keys())
)
selected_profile = PREDEFINED_PROFILES[profile_name]

# הגדרת קריטריונים באמצעות שדות קלט
st.sidebar.subheader("קריטריונים מותאמים אישית")

min_stock_price = st.sidebar.number_input(
    "מחיר מניה מינימלי ($):",
    min_value=0.0, value=float(selected_profile["min_stock_price"]), step=1.0
)
max_stock_price = st.sidebar.number_input(
    "מחיר מניה מקסימלי ($):",
    min_value=0.0, value=float(selected_profile["max_stock_price"]), step=1.0
)
max_pe_ratio = st.sidebar.number_input(
    "יחס P/E מקסימלי:",
    min_value=1.0, value=float(selected_profile["max_pe_ratio"]), step=1.0
)
min_avg_daily_volume = st.sidebar.number_input(
    "נפח מסחר יומי ממוצע מינימלי:",
    min_value=0, value=int(selected_profile["min_avg_daily_volume"]), step=100_000
)
min_iv_threshold = st.sidebar.slider(
    "סף IV גלום מינימלי (%):",
    min_value=0.0, max_value=1.0, value=selected_profile["min_iv_threshold"], step=0.01, format="%.2f"
)
min_dte = st.sidebar.number_input(
    "DTE מינימלי:",
    min_value=1, value=int(selected_profile["min_dte"]), step=1
)
max_dte = st.sidebar.number_input(
    "DTE מקסימלי:",
    min_value=1, value=int(selected_profile["max_dte"]), step=1
)
target_delta_directional = st.sidebar.slider(
    "דלתא יעד כיוונית (Bull Put/Bear Call):",
    min_value=0.01, max_value=0.50, value=selected_profile["target_delta_directional"], step=0.01, format="%.2f"
)
target_delta_neutral = st.sidebar.slider(
    "דלתא יעד ניטרלית (Iron Condor):",
    min_value=0.01, max_value=0.30, value=selected_profile["target_delta_neutral"], step=0.01, format="%.2f"
)

# עדכון אובייקט הקריטריונים
current_criteria = {
    "min_stock_price": min_stock_price, "max_stock_price": max_stock_price,
    "max_pe_ratio": max_pe_ratio, "min_avg_daily_volume": min_avg_daily_volume,
    "min_iv_threshold": min_iv_threshold, "min_dte": min_dte, "max_dte": max_dte,
    "target_delta_directional": target_delta_directional, "target_delta_neutral": target_delta_neutral
}


# בחירת אינדקסים לסריקה
index_to_scan = st.sidebar.selectbox(
    "בחר אינדקס לסריקה:",
    options=["S&P 500", "NASDAQ 100", "שניהם"]
)

# טוען את יקום ההשקעה באופן דינמי לפי הבחירה
INVESTMENT_UNIVERSE = get_tickers_from_wikipedia(index_to_scan)
if not INVESTMENT_UNIVERSE: # אם הגירוד נכשל, נחזור לרשימה קטנה לדוגמה
    st.warning("גירוד רשימת המניות נכשל או ריק. משתמש ברשימה מצומצמת לדוגמה.")
    INVESTMENT_UNIVERSE = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V", "PG", "JNJ",
        "KO", "PEP", "MCD", "WMT", "HD", "CRM", "ADBE", "NFLX", "CMCSA", "PYPL",
        "QCOM", "INTC", "AMD", "CSCO", "SBUX", "COST", "LLY", "UNH", "XOM", "CVX",
        "ORCL", "BAC", "WFC", "DIS", "NKE", "BA", "SPY", "QQQ"
    ]


selected_tickers = st.multiselect(
    "בחר מניות לסריקה (ברירת מחדל: כל המניות מהאינדקס הנבחר):",
    options=INVESTMENT_UNIVERSE,
    default=INVESTMENT_UNIVERSE # ברירת מחדל לבחור את כל המניות מהרשימה
)

st.info("""
    **הערות חשובות:**
    * **זמן ריצה:** סריקת מניות רבות (במיוחד S&P 500 ו-NASDAQ 100) עלולה לקחת **זמן רב מאוד** (עשרות דקות ואף שעות) עקב מגבלות קצב של ספק הנתונים.
    * **נתונים:** הנתונים נדלים מ-Yahoo Finance ואינם בזמן אמת (מעוכבים או סוף יום).
    * **IV Rank:** הכלי משתמש ב-`impliedVolatility` כאינדיקציה לוולטיליות גבוהה, במקום IV Rank אמיתי שדורש נתונים היסטוריים.
    * **דוחות רווחים:** הכלי אינו בודק תאריכי דוחות רווחים עתידיים. יש לבצע בדיקה זו ידנית.
""")

if st.button("נתח ומצא את העסקאות הטובות ביותר"):
    if not selected_tickers:
        st.warning("אנא בחר לפחות מניה אחת לסריקה.")
    else:
        st.subheader("תוצאות ניתוח")
        all_suitable_deals = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker_symbol in enumerate(selected_tickers):
            status_text.text(f"סורק מניה: {ticker_symbol} ({i+1}/{len(selected_tickers)})")
            progress_bar.progress((i + 1) / len(selected_tickers))

            stock_data = get_stock_data(ticker_symbol)
            is_suitable_stock, reason = screen_stock(stock_data, current_criteria) # העבר את הקריטריונים

            if not is_suitable_stock:
                continue # לא מציג הודעות סינון שליליות

            current_price = stock_data['current_price']
            sma50 = stock_data['sma50']
            
            suitable_options_found_for_ticker = False
            if stock_data['options_expirations']:
                for expiration_date in stock_data['options_expirations']:
                    calls_df, puts_df = get_option_chain(ticker_symbol, expiration_date)
                    
                    today = datetime.now().date()
                    try:
                        exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d').date()
                    except ValueError:
                        continue
                    dte = (exp_dt - today).days

                    if not (current_criteria["min_dte"] <= dte <= current_criteria["max_dte"]):
                        continue

                    # --- אסטרטגיה כיוונית (Bull Put / Bear Call) ---
                    best_put_directional = find_best_option_strike(puts_df, current_price, 'put', current_criteria["target_delta_directional"], current_criteria)
                    best_call_directional = find_best_option_strike(calls_df, current_price, 'call', current_criteria["target_delta_directional"], current_criteria)

                    if best_put_directional and current_price > sma50: # Bull Put Spread
                        credit = best_put_directional['bid'] - SPREAD_WIDTH
                        if credit > 0:
                            pop = 1 - abs(best_put_directional['delta'])
                            ev, ror = calculate_trade_metrics('Bull Put', credit, SPREAD_WIDTH, pop)
                            if ev > 0:
                                all_suitable_deals.append({
                                    'מניה': ticker_symbol,
                                    'אסטרטגיה': 'Bull Put',
                                    'מחיר מניה': f"${current_price:.2f}",
                                    'SMA50': f"${sma50:.2f}",
                                    'ת. פקיעה': best_put_directional['expiration'],
                                    'DTE': best_put_directional['dte'],
                                    'דלתא (נמכר)': f"{best_put_directional['delta']:.2f}",
                                    'סטרייק (נמכר)': f"${best_put_directional['strike']:.2f}",
                                    'IV גלום (נמכר)': f"{best_put_directional['implied_volatility']:.2%}",
                                    'פרמיה (מוערך)': f"${credit:.2f}",
                                    'תוחלת רווח (EV)': f"${ev:.2f}",
                                    'תשואה על סיכון (ROR)': f"{ror:.1f}%" if ror != float('inf') else '∞',
                                    'הסתברות לרווח (POP)': f"{pop:.1%}",
                                    'הוראת GTC (קנה חזרה)': f"${(credit / 2):.2f}",
                                    'תאריך יעד לניהול': (exp_dt - timedelta(days=21)).strftime('%Y-%m-%d')
                                })
                                suitable_options_found_for_ticker = True

                    if best_call_directional and current_price < sma50: # Bear Call Spread
                        credit = best_call_directional['bid'] - SPREAD_WIDTH
                        if credit > 0:
                            pop = 1 - abs(best_call_directional['delta'])
                            ev, ror = calculate_trade_metrics('Bear Call', credit, SPREAD_WIDTH, pop)
                            if ev > 0:
                                all_suitable_deals.append({
                                    'מניה': ticker_symbol,
                                    'אסטרטגיה': 'Bear Call',
                                    'מחיר מניה': f"${current_price:.2f}",
                                    'SMA50': f"${sma50:.2f}",
                                    'ת. פקיעה': best_call_directional['expiration'],
                                    'DTE': best_call_directional['dte'],
                                    'דלתא (נמכר)': f"{best_call_directional['delta']:.2f}",
                                    'סטרייק (נמכר)': f"${best_call_directional['strike']:.2f}",
                                    'IV גלום (נמכר)': f"{best_call_directional['implied_volatility']:.2%}",
                                    'פרמיה (מוערך)': f"${credit:.2f}",
                                    'תוחלת רווח (EV)': f"${ev:.2f}",
                                    'תשואה על סיכון (ROR)': f"{ror:.1f}%" if ror != float('inf') else '∞',
                                    'הסתברות לרווח (POP)': f"{pop:.1%}",
                                    'הוראת GTC (קנה חזרה)': f"${(credit / 2):.2f}",
                                    'תאריך יעד לניהול': (exp_dt - timedelta(days=21)).strftime('%Y-%m-%d')
                                })
                                suitable_options_found_for_ticker = True

                    # --- אסטרטגיה ניטרלית (Iron Condor) ---
                    best_put_neutral = find_best_option_strike(puts_df, current_price, 'put', current_criteria["target_delta_neutral"], current_criteria)
                    best_call_neutral = find_best_option_strike(calls_df, current_price, 'call', current_criteria["target_delta_neutral"], current_criteria)

                    if best_put_neutral and best_call_neutral:
                        credit_put_side = best_put_neutral['bid'] - SPREAD_WIDTH
                        credit_call_side = best_call_neutral['bid'] - SPREAD_WIDTH
                        
                        if credit_put_side > 0 and credit_call_side > 0:
                            total_credit = credit_put_side + credit_call_side
                            pop_ic = 1 - (abs(best_put_neutral['delta']) + abs(best_call_neutral['delta']))
                            
                            ev_ic, ror_ic = calculate_trade_metrics('Iron Condor', total_credit, SPREAD_WIDTH, pop_ic)
                            if ev_ic > 0:
                                all_suitable_deals.append({
                                    'מניה': ticker_symbol,
                                    'אסטרטגיה': 'Iron Condor',
                                    'מחיר מניה': f"${current_price:.2f}",
                                    'SMA50': f"${sma50:.2f}",
                                    'ת. פקיעה': best_put_neutral['expiration'],
                                    'DTE': best_put_neutral['dte'],
                                    'דלתא פוט (נמכר)': f"{best_put_neutral['delta']:.2f}",
                                    'סטרייק פוט (נמכר)': f"${best_put_neutral['strike']:.2f}",
                                    'IV גלום פוט': f"{best_put_neutral['implied_volatility']:.2%}",
                                    'דלתא קול (נמכר)': f"{best_call_neutral['delta']:.2f}",
                                    'סטרייק קול (נמכר)': f"${best_call_neutral['strike']:.2f}",
                                    'IV גלום קול': f"{best_call_neutral['implied_volatility']:.2%}",
                                    'פרמיה כוללת (מוערך)': f"${total_credit:.2f}",
                                    'תוחלת רווח (EV)': f"${ev_ic:.2f}",
                                    'תשואה על סיכון (ROR)': f"{ror_ic:.1f}%" if ror_ic != float('inf') else '∞',
                                    'הסתברות לרווח (POP)': f"{pop_ic:.1%}",
                                    'הוראת GTC (קנה חזרה)': f"${(total_credit / 2):.2f}",
                                    'תאריך יעד לניהול': (exp_dt - timedelta(days=21)).strftime('%Y-%m-%d')
                                })
                                suitable_options_found_for_ticker = True
                
            # אם לא נמצאו אופציות מתאימות עבור המניה הספציפית, נרשום זאת
            # אבל לא נציג את זה אם לא נמצאו עסקאות בכלל
            # if not suitable_options_found_for_ticker and is_suitable_stock:
            #     st.write(f"**{ticker_symbol}:** 🤷 לא נמצאו עסקאות אופציות מתאימות לפי הקריטריונים.")
            # elif not stock_data['options_expirations'] and is_suitable_stock:
            #     st.write(f"**{ticker_symbol}:** 🤷 לא נמצאו תאריכי פקיעה לאופציות.")


        progress_bar.empty()
        status_text.empty()

        if all_suitable_deals:
            deals_df = pd.DataFrame(all_suitable_deals)
            
            # חישוב ציון לדירוג
            deals_df['EV_numeric'] = deals_df['תוחלת רווח (EV)'].str.replace('$', '').astype(float)
            deals_df['ROR_numeric'] = deals_df['תשואה על סיכון (ROR)'].str.replace('%', '').replace('∞', np.inf).astype(float)
            deals_df['POP_numeric'] = deals_df['הסתברות לרווח (POP)'].str.replace('%', '').astype(float)

            max_ev = deals_df['EV_numeric'].max()
            max_ror = deals_df['ROR_numeric'].max()
            max_pop = deals_df['POP_numeric'].max()

            norm_ev = deals_df['EV_numeric'] / max_ev if max_ev > 0 else 0
            norm_ror = deals_df['ROR_numeric'] / max_ror if max_ror > 0 else 0
            norm_pop = deals_df['POP_numeric'] / max_pop if max_pop > 0 else 0

            deals_df['ציון'] = (
                (norm_ev * 0.45) +
                (norm_pop * 0.45) +
                (norm_ror * 0.10)
            )
            
            deals_df = deals_df.sort_values(by='ציון', ascending=False)
            
            # הסתרת עמודות העזר המספריות
            deals_df = deals_df.drop(columns=['EV_numeric', 'ROR_numeric', 'POP_numeric'])

            st.dataframe(deals_df, use_container_width=True)
            st.success("הניתוח הושלם! העסקאות המומלצות ביותר מוצגות בטבלה.")
        else:
            st.warning("לא נמצאו עסקאות אופציות מתאימות העומדות בכל הקריטריונים שהוגדרו. נסה לשנות את קריטריוני הסינון בצד (Sidebar) או לבחור אינדקסים נוספים.")


