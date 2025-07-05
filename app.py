import streamlit as st # ספריית Streamlit לבניית ממשק המשתמש
import yfinance as yf # ספרייה לדליית נתונים מ-Yahoo Finance
import pandas as pd # ספרייה לעיבוד נתונים (DataFrames)
from datetime import datetime, timedelta # לטיפול בתאריכים
import numpy as np # לחישובים מתמטיים
from scipy.stats import norm # עבור מודל בלאק-שולס

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

# שלב 1: סינון מניות
MIN_STOCK_PRICE = 20
MAX_STOCK_PRICE = 70
MAX_PE_RATIO = 40
MIN_AVG_DAILY_VOLUME = 2_000_000
MIN_IV_THRESHOLD = 0.30 # סף ל-Implied Volatility (במקום IV Rank)

# שלב 2: בחירת האופציה
MIN_DTE = 30
MAX_DTE = 60
SPREAD_WIDTH = 1 # רוחב המרווח בדולרים

# דלתא יעד לאסטרטגיות
TARGET_DELTA_DIRECTIONAL = 0.30
TARGET_DELTA_NEUTRAL = 0.15

# --- רשימת מניות לסריקה (יקום ההשקעה) ---
# זוהי רשימה חלקית לדוגמה.
# ביישום אמיתי, תצטרך מקור לכל מניות S&P 500 ו-NASDAQ 100.
# ניתן למצוא קבצי CSV או APIs (לרוב בתשלום) המספקים רשימות אלו.
INVESTMENT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V", "PG", "JNJ",
    "KO", "PEP", "MCD", "WMT", "HD", "CRM", "ADBE", "NFLX", "CMCSA", "PYPL",
    "QCOM", "INTC", "AMD", "CSCO", "SBUX", "COST", "LLY", "UNH", "XOM", "CVX",
    "ORCL", "BAC", "WFC", "DIS", "NKE", "BA", "SPY", "QQQ" # הוספתי SPY ו-QQQ כ-ETFs
]

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
        st.error(f"שגיאה בדליית נתונים עבור {ticker_symbol}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_option_chain(ticker_symbol, expiration_date):
    try:
        ticker = yf.Ticker(ticker_symbol)
        option_chain = ticker.option_chain(expiration_date)
        return option_chain.calls, option_chain.puts
    except Exception as e:
        st.error(f"שגיאה בדליית שרשרת אופציות עבור {ticker_symbol} בתאריך {expiration_date}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- פונקציות לסינון וחישובים לפי "ספר החוקים" ---
def screen_stock(stock_data):
    """מיישם את כללי שלב 1: סינון מניות."""
    if not stock_data:
        return False, "אין נתונים"

    price = stock_data['current_price']
    pe = stock_data['pe_ratio']
    volume = stock_data['avg_volume']
    
    if not (price and pe and volume):
        return False, "נתונים חסרים (מחיר/PE/ווליום)"

    # טווח מחיר
    if not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
        return False, f"מחיר מחוץ לטווח ({price})"
    
    # חוזק עסקי (P/E)
    if not (pe > 0 and pe < MAX_PE_RATIO):
        return False, f"P/E לא מתאים ({pe})"
        
    # נזילות
    if not (volume >= MIN_AVG_DAILY_VOLUME):
        return False, f"ווליום נמוך ({volume})"
        
    # הערה: IV Rank ודוחות רווחים לא נבדקים כאן אוטומטית.
    # IV Rank יטופל ברמת האופציה (impliedVolatility).
    # דוחות רווחים דורשים בדיקה ידנית או API נוסף.

    return True, "עבר סינון מניה"

def find_best_option_strike(options_df, current_price, option_type, target_delta, is_directional=True):
    """
    מוצא את הסטרייק הטוב ביותר לפי כלל הדלתא הבטוחה.
    options_df: DataFrame של אופציות (calls או puts)
    current_price: מחיר נכס הבסיס הנוכחי
    option_type: 'call' או 'put'
    target_delta: יעד הדלתא (0.30 או 0.15)
    is_directional: True לאסטרטגיה כיוונית, False לניטרלית (משפיע על כיוון הדלתא)
    """
    best_strike_data = None
    min_delta_diff = float('inf')

    # חישוב DTE
    today = datetime.now().date()
    
    for _, row in options_df.iterrows():
        strike = row['strike']
        implied_volatility = row['impliedVolatility']
        last_trade_date_ts = row['lastTradeDate']
        
        # המרת זמן לפקיעה בשנים
        # yfinance מספק תאריך פקיעה בפורמט ISO, נשתמש בו ישירות
        # נשתמש ב-lastTradeDate כנקודת יחוס אם קיים, אחרת היום
        if pd.isna(last_trade_date_ts):
            # אם אין lastTradeDate, נניח שהאופציה עדיין לא נסחרה היום
            # ונחשב DTE מתאריך הפקיעה
            expiration_date = datetime.strptime(row['expiration'], '%Y-%m-%d').date()
            dte = (expiration_date - today).days
        else:
            # אם יש lastTradeDate, נשתמש בו כנקודת יחוס
            last_trade_dt = datetime.fromtimestamp(last_trade_date_ts).date()
            expiration_date = datetime.strptime(row['expiration'], '%Y-%m-%d').date()
            dte = (expiration_date - last_trade_dt).days


        if not (MIN_DTE <= dte <= MAX_DTE):
            continue # מדלג על אופציות מחוץ לטווח DTE

        # נתוני נזילות וביד/אסק
        bid = row['bid']
        ask = row['ask']
        volume = row['volume']
        open_interest = row['openInterest']

        # וודא שיש נזילות מינימלית
        if volume == 0 or open_interest == 0 or bid == 0 or ask == 0:
            continue

        # וודא ש-implied_volatility תקין (לא NaN או 0)
        if pd.isna(implied_volatility) or implied_volatility <= 0:
            continue

        # חישוב דלתא באמצעות בלאק-שולס
        T_years = dte / 365.0
        if T_years <= 0: # מונע חלוקה באפס או שורש של מספר שלילי
            continue

        _, delta, _, _, _ = black_scholes(current_price, strike, T_years, RISK_FREE_RATE, implied_volatility, option_type)
        
        # כלל הדלתא הבטוחה: קרובה ליעד אבל מתחתיו
        if option_type == 'call': # עבור Call, דלתא חיובית, רוצים מתחת ליעד
            if delta > target_delta:
                continue # אם הדלתא גבוהה מדי, לא מתאים
            delta_diff = target_delta - delta # רוצים שההפרש יהיה חיובי וקטן
        else: # עבור Put, דלתא שלילית, רוצים מתחת ליעד (כלומר, יותר שלילית)
            if delta < -target_delta: # לדוגמה, אם יעד 0.30, רוצים דלתא בין -0.29 ל-0.01
                continue # אם הדלתא נמוכה מדי (יותר שלילית), לא מתאים
            delta_diff = abs(target_delta - abs(delta)) # רוצים שההפרש יהיה חיובי וקטן

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
                'expiration': row['expiration']
            }
    return best_strike_data

def calculate_trade_metrics(strategy_type, credit, spread_width, pop):
    """
    מחשב תוחלת רווח (EV) ותשואה על סיכון (RoR).
    pop: הסתברות לרווח (Probability of Profit)
    """
    max_profit = credit * 100 # פרמיה מקסימלית * 100 מניות
    max_loss = (spread_width - credit) * 100 # רוחב המרווח - פרמיה * 100 מניות
    
    # אם max_loss שלילי, זה אומר שהקרדיט גדול מרוחב המרווח, מצב לא הגיוני או שגיאה בנתונים.
    # נניח ש-max_loss תמיד חיובי עבור אסטרטגיית מרווח.
    if max_loss <= 0: # מונע חלוקה באפס או RoR אינסופי לא הגיוני
        return -float('inf'), -float('inf') # EV ו-RoR שליליים מאוד

    ev = (pop * max_profit) - ((1 - pop) * max_loss)
    ror = (max_profit / max_loss) * 100 # באחוזים

    return ev, ror

# --- ממשק המשתמש של Streamlit ---
st.set_page_config(layout="wide", page_title="שולחן העבודה של מנהל התיק - אוטונומי")

st.markdown("""
    <style>
        .stApp {
            background-color: #111827;
            color: #E5E7EB;
            font-family: 'Heebo', sans-serif;
        }
        .stButton > button {
            background-color: #3B82F6;
            color: white;
            font-weight: bold;
            padding: 0.75rem 1.5rem;
            border-radius: 0.375rem;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #2563EB;
        }
        .stTextInput > div > div > input {
            background-color: #374151;
            color: #F9FAFB;
            border: 1px solid #4B5563;
            border-radius: 0.375rem;
            padding: 0.5rem;
        }
        .stSelectbox > div > div {
            background-color: #374151;
            color: #F9FAFB;
            border: 1px solid #4B5563;
            border-radius: 0.375rem;
        }
        .stTable, .stDataFrame {
            background-color: #1F2937;
            color: #E5E7EB;
            border: 1px solid #374151;
            border-radius: 0.5rem;
        }
        .css-1r6slb0 { /* Header for dataframe */
            background-color: #374151;
            color: #E5E7EB;
        }
        .css-1r6slb0 th { /* Specific header cells */
            background-color: #374151;
            color: #E5E7EB;
        }
        .css-1dbjc4n.e1tzin5v1 { /* Main content area */
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
        }
        .reportview-container .main .block-container{
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }
        .stAlert {
            background-color: #1F2937;
            color: #E5E7EB;
            border-color: #374151;
        }
    </style>
""", unsafe_allow_html=True)

st.title("שולחן העבודה של מנהל התיק - אוטונומי")
st.markdown("הכלי המרכזי שלך לקבלת החלטות, המבוסס על 'ספר החוקים' שלך.")

st.subheader("הגדרות סריקה")
selected_tickers = st.multiselect(
    "בחר מניות לסריקה (מומלץ לבחור מספר מצומצם לניסוי ראשוני):",
    options=INVESTMENT_UNIVERSE,
    default=["AAPL", "MSFT", "SPY"]
)

st.info("""
    **הערות חשובות:**
    * **נתונים:** הנתונים נדלים מ-Yahoo Finance ואינם בזמן אמת (מעוכבים או סוף יום).
    * **IV Rank:** הכלי משתמש ב-`impliedVolatility` כאינדיקציה לוולטיליות גבוהה, במקום IV Rank אמיתי שדורש נתונים היסטוריים.
    * **דוחות רווחים:** הכלי אינו בודק תאריכי דוחות רווחים עתידיים. יש לבצע בדיקה זו ידנית.
    * **זמן ריצה:** סריקת מניות רבות עלולה לקחת זמן.
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
            is_suitable_stock, reason = screen_stock(stock_data)

            if not is_suitable_stock:
                st.write(f"**{ticker_symbol}:** ❌ לא עבר סינון מניה. סיבה: {reason}")
                continue
            
            st.write(f"**{ticker_symbol}:** ✅ עבר סינון מניה.")

            current_price = stock_data['current_price']
            sma50 = stock_data['sma50']
            
            # חיפוש אופציות מתאימות
            suitable_options_found = False
            for expiration_date in stock_data['options_expirations']:
                calls_df, puts_df = get_option_chain(ticker_symbol, expiration_date)
                
                # חישוב DTE
                today = datetime.now().date()
                exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d').date()
                dte = (exp_dt - today).days

                if not (MIN_DTE <= dte <= MAX_DTE):
                    continue # מדלג על תאריכי פקיעה מחוץ לטווח DTE

                # --- אסטרטגיה כיוונית (Bull Put / Bear Call) ---
                best_put_directional = find_best_option_strike(puts_df, current_price, 'put', TARGET_DELTA_DIRECTIONAL)
                best_call_directional = find_best_option_strike(calls_df, current_price, 'call', TARGET_DELTA_DIRECTIONAL)

                if best_put_directional and current_price > sma50: # Bull Put Spread
                    credit = best_put_directional['bid'] - SPREAD_WIDTH # Credit for selling put, buying further OTM put
                    if credit > 0:
                        pop = 1 - abs(best_put_directional['delta']) # הסתברות לרווח (קירוב)
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
                            suitable_options_found = True

                if best_call_directional and current_price < sma50: # Bear Call Spread
                    credit = best_call_directional['bid'] - SPREAD_WIDTH # Credit for selling call, buying further OTM call
                    if credit > 0:
                        pop = 1 - abs(best_call_directional['delta']) # הסתברות לרווח (קירוב)
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
                            suitable_options_found = True

                # --- אסטרטגיה ניטרלית (Iron Condor) ---
                best_put_neutral = find_best_option_strike(puts_df, current_price, 'put', TARGET_DELTA_NEUTRAL, is_directional=False)
                best_call_neutral = find_best_option_strike(calls_df, current_price, 'call', TARGET_DELTA_NEUTRAL, is_directional=False)

                if best_put_neutral and best_call_neutral:
                    credit_put_side = best_put_neutral['bid'] - SPREAD_WIDTH
                    credit_call_side = best_call_neutral['bid'] - SPREAD_WIDTH
                    
                    if credit_put_side > 0 and credit_call_side > 0:
                        total_credit = credit_put_side + credit_call_side
                        # הסתברות לרווח עבור Iron Condor היא מורכבת יותר, נשתמש בקירוב
                        # 1 - (דלתא פוט + דלתא קול) כקירוב גס לסיכוי שהמחיר ייצא מחוץ לטווח
                        # לכן POP = (1 - (abs(דלתא פוט) + abs(דלתא קול)))
                        # או פשוט 1 - (הסתברות לנגיעה בסטרייק התחתון + הסתברות לנגיעה בסטרייק העליון)
                        # נשתמש בקירוב פשוט יותר: 1 - (abs(דלתא פוט) + abs(דלתא קול))
                        pop_ic = 1 - (abs(best_put_neutral['delta']) + abs(best_call_neutral['delta']))
                        
                        ev_ic, ror_ic = calculate_trade_metrics('Iron Condor', total_credit, SPREAD_WIDTH, pop_ic)
                        if ev_ic > 0:
                            all_suitable_deals.append({
                                'מניה': ticker_symbol,
                                'אסטרטגיה': 'Iron Condor',
                                'מחיר מניה': f"${current_price:.2f}",
                                'SMA50': f"${sma50:.2f}",
                                'ת. פקיעה': best_put_neutral['expiration'], # תאריך פקיעה זהה לשני הצדדים
                                'DTE': best_put_neutral['dte'],
                                'דלתא פוט (נמכר)': f"{best_put_neutral['delta']:.2f}",
                                'סטרייק פוט (נמכר)': f"${best_put_neutral['strike']:.2f}",
                                'IV גלום פוט': f"{best_put_neutral['implied_volatility']:.2%}",
                                'דלתא קול (נמכר)': f"{best_call_neutral['delta']:.2f}",
                                'סטרייק קול (נמכר)': f"${best_call_neutral['strike']:.2f}",
                                'IV גלום קול': f"{best_call_neutral['implied_volatility']:.2%}",
                                'פרמיה כוללת (מוערך)': f"${total_credit:.2f}",
                                'תוחלת רווח (EV)': f"${ev_ic:.2f}",
                                'תשואה על