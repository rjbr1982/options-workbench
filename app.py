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

# --- פונקציה לדליית רשימת מניות מויקיפדיה (אוטונומי) ---
@st.cache_data(ttl=86400) # שמירה במטמון ל-24 שעות (86400 שניות)
def get_sp500_nasdaq100_tickers():
    """דולה את רשימת המניות של S&P 500 ו-NASDAQ 100 מויקיפדיה."""
    tickers = set()

    # S&P 500
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        response = requests.get(sp500_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # הטבלה הראשונה בדף היא לרוב זו עם הסמלים
        table = soup.find('table', {'class': 'wikitable sortable'})
        if table:
            for row in table.findAll('tr')[1:]: # מדלג על שורת הכותרת
                ticker = row.findAll('td')[0].text.strip()
                tickers.add(ticker)
        else:
            st.warning("לא נמצאה טבלת S&P 500 בויקיפדיה. ייתכן שהמבנה השתנה.")
    except Exception as e:
        st.error(f"שגיאה בדליית S&P 500 מויקיפדיה: {e}")

    # NASDAQ 100
    nasdaq100_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        response = requests.get(nasdaq100_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # חפש את הטבלה המתאימה (לרוב יש כמה)
        table = soup.find('table', {'class': 'wikitable sortable'})
        if table:
            for row in table.findAll('tr')[1:]: # מדלג על שורת הכותרת
                # סמל המניה הוא לרוב בעמודה השנייה או השלישית, ננסה את הראשונה
                # ייתכן שצריך להתאים את האינדקס [1] או [2] בהתאם למבנה הדף
                try:
                    ticker = row.findAll('td')[1].text.strip() # נסה עמודה שנייה
                    tickers.add(ticker)
                except IndexError:
                    pass # אם אין עמודה שנייה, ננסה עמודה אחרת או נדלג
        else:
            st.warning("לא נמצאה טבלת NASDAQ 100 בויקיפדיה. ייתכן שהמבנה השתנה.")
    except Exception as e:
        st.error(f"שגיאה בדליית NASDAQ 100 מויקיפדיה: {e}")

    # המרת ה-set לרשימה ממוינת
    return sorted(list(tickers))

# טוען את יקום ההשקעה באופן דינמי
INVESTMENT_UNIVERSE = get_sp500_nasdaq100_tickers()
if not INVESTMENT_UNIVERSE: # אם הגירוד נכשל, נחזור לרשימה קטנה לדוגמה
    st.warning("גירוד רשימת המניות נכשל או ריק. משתמש ברשימה מצומצמת לדוגמה.")
    INVESTMENT_UNIVERSE = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V", "PG", "JNJ",
        "KO", "PEP", "MCD", "WMT", "HD", "CRM", "ADBE", "NFLX", "CMCSA", "PYPL",
        "QCOM", "INTC", "AMD", "CSCO", "SBUX", "COST", "LLY", "UNH", "XOM", "CVX",
        "ORCL", "BAC", "WFC", "DIS", "NKE", "BA", "SPY", "QQQ"
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
        # st.error(f"שגיאה בדליית נתונים עבור {ticker_symbol}: {e}") # נוריד את זה כדי לא להציף בשגיאות
        return None

@st.cache_data(ttl=3600)
def get_option_chain(ticker_symbol, expiration_date):
    try:
        ticker = yf.Ticker(ticker_symbol)
        option_chain = ticker.option_chain(expiration_date)
        return option_chain.calls, option_chain.puts
    except Exception as e:
        # st.error(f"שגיאה בדליית שרשרת אופציות עבור {ticker_symbol} בתאריך {expiration_date}: {e}") # נוריד את זה כדי לא להציף בשגיאות
        return pd.DataFrame(), pd.DataFrame()

# --- פונקציות לסינון וחישובים לפי "ספר החוקים" ---
def screen_stock(stock_data):
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
    if not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
        return False, f"מחיר מחוץ לטווח ({price:.2f})"
    
    # חוזק עסקי (P/E)
    if not (pe > 0 and pe < MAX_PE_RATIO):
        return False, f"P/E לא מתאים ({pe:.2f})"
        
    # נזילות
    if not (volume >= MIN_AVG_DAILY_VOLUME):
        return False, f"ווליום נמוך ({volume:,})"
        
    # הערה: IV Rank ודוחות רווחים לא נבדקים כאן אוטומטית.
    # IV Rank יטופל ברמת האופציה (impliedVolatility).
    # דוחות רווחים דורשים בדיקה ידנית או API נוסף.

    return True, "עבר סינון מניה"

def find_best_option_strike(options_df, current_price, option_type, target_delta):
    """
    מוצא את הסטרייק הטוב ביותר לפי כלל הדלתא הבטוחה.
    options_df: DataFrame של אופציות (calls או puts)
    current_price: מחיר נכס הבסיס הנוכחי
    option_type: 'call' או 'put'
    target_delta: יעד הדלתא (0.30 או 0.15)
    """
    best_strike_data = None
    min_delta_diff = float('inf')

    # חישוב DTE
    today = datetime.now().date()
    
    for _, row in options_df.iterrows():
        strike = row['strike']
        implied_volatility = row['impliedVolatility']
        
        # המרת תאריך פקיעה
        expiration_date_str = row['expiration']
        try:
            expiration_date = datetime.strptime(expiration_date_str, '%Y-%m-%d').date()
        except ValueError:
            continue # מדלג אם פורמט התאריך לא תקין

        dte = (expiration_date - today).days

        if not (MIN_DTE <= dte <= MAX_DTE):
            continue # מדלג על אופציות מחוץ לטווח DTE

        # נתוני נזילות וביד/אסק
        bid = row['bid']
        ask = row['ask']
        volume = row['volume']
        open_interest = row['openInterest']

        # וודא שיש נזילות מינימלית
        if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0 or volume == 0 or open_interest == 0:
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
            if delta > 0: # דלתא של Put צריכה להיות שלילית
                continue
            if abs(delta) > target_delta: # אם הדלתא (במוחלט) גבוהה מדי, כלומר יותר שלילית מהיעד
                continue
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
                'expiration': expiration_date_str # שמירה כסטרינג
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

# טעינת קובץ ה-CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("שולחן העבודה של מנהל התיק - אוטונומי")
st.markdown("הכלי המרכזי שלך לקבלת החלטות, המבוסס על 'ספר החוקים' שלך.")

st.subheader("הגדרות סריקה")

# מאפשר לבחור את כל המניות כברירת מחדל, אך עדיין מאפשר לבחור תת-קבוצה
selected_tickers = st.multiselect(
    "בחר מניות לסריקה (מומלץ לבחור מספר מצומצם לניסוי ראשוני, או את כל הרשימה):",
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
            is_suitable_stock, reason = screen_stock(stock_data)

            if not is_suitable_stock:
                st.write(f"**{ticker_symbol}:** ❌ לא עבר סינון מניה. סיבה: {reason}")
                continue
            
            st.write(f"**{ticker_symbol}:** ✅ עבר סינון מניה.")

            current_price = stock_data['current_price']
            sma50 = stock_data['sma50']
            
            # חיפוש אופציות מתאימות
            suitable_options_found = False
            if stock_data['options_expirations']: # וודא שיש תאריכי פקיעה
                for expiration_date in stock_data['options_expirations']:
                    calls_df, puts_df = get_option_chain(ticker_symbol, expiration_date)
                    
                    # חישוב DTE
                    today = datetime.now().date()
                    try:
                        exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d').date()
                    except ValueError:
                        continue # מדלג אם פורמט התאריך לא תקין
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
                    best_put_neutral = find_best_option_strike(puts_df, current_price, 'put', TARGET_DELTA_NEUTRAL)
                    best_call_neutral = find_best_option_strike(calls_df, current_price, 'call', TARGET_DELTA_NEUTRAL)

                    if best_put_neutral and best_call_neutral:
                        credit_put_side = best_put_neutral['bid'] - SPREAD_WIDTH
                        credit_call_side = best_call_neutral['bid'] - SPREAD_WIDTH
                        
                        if credit_put_side > 0 and credit_call_side > 0:
                            total_credit = credit_put_side + credit_call_side
                            # הסתברות לרווח עבור Iron Condor היא מורכבת יותר, נשתמש בקירוב
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
                                    'תשואה על סיכון (ROR)': f"{ror_ic:.1f}%" if ror_ic != float('inf') else '∞',
                                    'הסתברות לרווח (POP)': f"{pop_ic:.1%}",
                                    'הוראת GTC (קנה חזרה)': f"${(total_credit / 2):.2f}",
                                    'תאריך יעד לניהול': (exp_dt - timedelta(days=21)).strftime('%Y-%m-%d')
                                })
                                suitable_options_found = True
                
            if not suitable_options_found and is_suitable_stock:
                st.write(f"**{ticker_symbol}:** 🤷 לא נמצאו עסקאות אופציות מתאימות לפי הקריטריונים.")
            elif not stock_data['options_expirations'] and is_suitable_stock:
                st.write(f"**{ticker_symbol}:** 🤷 לא נמצאו תאריכי פקיעה לאופציות.")


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

            # וודא שהמכנה אינו אפס כדי למנוע שגיאות
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
            st.warning("לא נמצאו עסקאות אופציות מתאימות העומדות בכל הקריטריונים שהוגדרו.")


