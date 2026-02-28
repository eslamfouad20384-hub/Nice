import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# ====== المجلدات ======
CACHE = "cache"
MODEL = "model"
TRADE_LOG = "trades.csv"
HISTORICAL = "historical_data"

for folder in [CACHE, MODEL, HISTORICAL]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ===============================
# جلب أفضل العملات مع fallback بين مصادر متعددة
# ===============================
def get_top_symbols(limit=20):
    sources = [
        lambda: requests.get("https://api.coingecko.com/api/v3/coins/markets",
                             params={"vs_currency":"usd","order":"volume_desc","per_page":limit,"page":1}, timeout=10).json(),
        lambda: requests.get("https://api.coinpaprika.com/v1/tickers", timeout=10).json()
    ]
    symbols = []
    for source in sources:
        try:
            data = source()
            if isinstance(data, list):
                for item in data[:limit]:
                    if "symbol" in item:
                        symbols.append(item["symbol"].upper())
                    elif "id" in item:
                        symbols.append(item["id"].upper())
                if symbols:
                    break
        except:
            continue
    return symbols[:limit]

# ===============================
# جلب بيانات OHLCV مع fallback
# ===============================
def fetch_ohlcv(symbol, interval="4h", limit=200):
    df = pd.DataFrame()
    # Primary: CryptoCompare
    try:
        base = "https://min-api.cryptocompare.com/data/v2/"
        url = f"{base}{'histohour' if interval=='4h' else 'histoday'}?fsym={symbol}&tsym=USDT&limit={limit}"
        data = requests.get(url).json()
        if data.get("Response")=="Success":
            df = pd.DataFrame(data["Data"]["Data"])
            hist_file = os.path.join(HISTORICAL, f"{symbol}_{interval}.csv")
            df.to_csv(hist_file,index=False)
            return df
    except:
        pass

    # Fallback: CoinGecko
    if interval=="daily":
        try:
            cg_url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart"
            params = {"vs_currency":"usd","days":limit,"interval":"daily"}
            data = requests.get(cg_url, params=params).json()
            if "prices" in data:
                df = pd.DataFrame(data["prices"], columns=["time","close"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df["high"] = df["close"]
                df["low"] = df["close"]
                df["open"] = df["close"]
                return df
        except:
            pass

    # Fallback: CoinPaprika
    try:
        url = f"https://api.coinpaprika.com/v1/coins/{symbol.lower()}-usd/ohlcv/historical"
        params = {"limit":limit}
        data = requests.get(url, params=params).json()
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df.rename(columns={"close":"close","high":"high","low":"low","open":"open","time_close":"time"}, inplace=True)
            df["time"] = pd.to_datetime(df["time"])
            return df
    except:
        pass

    return df

# ===============================
# إضافة المؤشرات + ATR ودعم/مقاومة
# ===============================
def add_indicators(df):
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["EMA200"] = df["close"].ewm(span=200).mean()
    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = abs(df["high"] - df["prev_close"])
    df["tr3"] = abs(df["low"] - df["prev_close"])
    df["TR"] = df[["tr1","tr2","tr3"]].max(axis=1)
    df["ATR"] = df["TR"].ewm(alpha=1/14, adjust=False).mean()
    df["return"] = df["close"].pct_change()
    # دعم ومقاومة
    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()
    return df.dropna()

# ===============================
# تدريب نموذج AI
# ===============================
def train_ai(df, symbol):
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    df = df.dropna()
    if len(df) < 100:
        return 0
    X = df[["EMA50","EMA200","ATR","return"]]
    y = df["target"]
    model_file = os.path.join(MODEL, f"{symbol}.pkl")
    if os.path.exists(model_file):
        model = pickle.load(open(model_file,"rb"))
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
    try:
        model.fit(X,y)
        pickle.dump(model,open(model_file,"wb"))
        return model.predict_proba(X.iloc[-1:])[0][1]
    except:
        return 0

# ===============================
# حالة السوق لعملة
# ===============================
def market_condition(symbol):
    df = fetch_ohlcv(symbol,"daily",200)
    if df.empty:
        return "غير متاح"
    df = add_indicators(df)
    last = df.iloc[-1]
    if last["EMA50"] > last["EMA200"] and last["close"] > last["EMA50"]:
        return "صاعد"
    elif last["EMA50"] < last["EMA200"] and last["close"] < last["EMA50"]:
        return "هابط"
    else:
        return "عرضي"

# ===============================
# توليد الإشارة مع إشارات تقاطع EMA
# ===============================
def generate_signal(symbol):
    df4h = fetch_ohlcv(symbol,"4h",200)
    if df4h.empty:
        return {"العملة": symbol, "دخول": np.nan, "وقف": np.nan, "هدف": np.nan,
                "احتمال_الصعود": np.nan, "حالة_السوق": np.nan,
                "حالة الإشارة": "مرفوض", "سبب": "بيانات 4H غير متاحة"}

    df = add_indicators(df4h)
    last = df.iloc[-1]

    dfd = fetch_ohlcv(symbol,"daily",200)
    if dfd.empty:
        return {"العملة": symbol, "دخول": np.nan, "وقف": np.nan, "هدف": np.nan,
                "احتمال_الصعود": np.nan, "حالة_السوق": np.nan,
                "حالة الإشارة": "مرفوض", "سبب": "بيانات يومية غير متاحة"}

    dfd = add_indicators(dfd)
    last_daily = dfd.iloc[-1]

    # إشارات تقاطع EMA
    cross_signal = ""
    if dfd["EMA50"].iloc[-2] < dfd["EMA200"].iloc[-2] and dfd["EMA50"].iloc[-1] > dfd["EMA200"].iloc[-1]:
        cross_signal = "صاعد"
    elif dfd["EMA50"].iloc[-2] > dfd["EMA200"].iloc[-2] and dfd["EMA50"].iloc[-1] < dfd["EMA200"].iloc[-1]:
        cross_signal = "هابط"

    prob = train_ai(df,symbol)
    entry = last["close"]
    atr = last["ATR"]
    stop = entry - atr*1.2
    target = entry + atr*1.8

    if prob < 0.55 and not cross_signal:
        trade_status = "مرفوض"
        reason = f"قوة AI ضعيفة ({round(prob*100,2)}%)"
    else:
        trade_status = "مقبول"
        reason = f"إشارة تقاطع EMA" if cross_signal else f"قوة AI قوية ({round(prob*100,2)}%)"

    trade = {"العملة":symbol, "تاريخ":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "دخول":round(entry,4) if trade_status=="مقبول" else np.nan,
             "وقف":round(stop,4) if trade_status=="مقبول" else np.nan,
             "هدف":round(target,4) if trade_status=="مقبول" else np.nan,
             "احتمال_الصعود":round(prob*100,2),
             "حالة_السوق":market_condition(symbol),
             "دعم": round(last_daily["support"],4),
             "مقاومة": round(last_daily["resistance"],4),
             "حالة الإشارة": trade_status,
             "سبب": reason}
    # تسجيل الصفقة
    if not os.path.exists(TRADE_LOG):
        df_log = pd.DataFrame(columns=list(trade.keys()))
        df_log = df_log.append(trade, ignore_index=True)
        df_log.to_csv(TRADE_LOG,index=False)
    else:
        df_log = pd.read_csv(TRADE_LOG)
        df_log = df_log.append(trade, ignore_index=True)
        df_log.to_csv(TRADE_LOG,index=False)

    return trade

# ===============================
# سكان السوق تلقائي
# ===============================
def scan_market():
    symbols = get_top_symbols(20)
    results = []
    for s in symbols:
        try:
            results.append(generate_signal(s))
            time.sleep(0.3)
        except:
            results.append({"العملة": s, "دخول": np.nan, "وقف": np.nan, "هدف": np.nan,
                            "احتمال_الصعود": np.nan, "حالة_السوق": np.nan,
                            "حالة الإشارة": "مرفوض", "سبب": "خطأ عام"})
    df = pd.DataFrame(results)
    df.index = np.arange(1, len(df)+1)
    return df

# ===============================
# Streamlit Interface
# ===============================
st.markdown('<h4 style="font-size:16px;">AI Spot Scanner</h4>', unsafe_allow_html=True)

# فحص السوق أوتوماتيك
df = scan_market()
st.markdown(f"### 🧭 حالة السوق العام: {', '.join(df['حالة_السوق'].dropna().unique())}")

def highlight_rows(row):
    color = 'background-color: #d4f8d4' if row.get('حالة الإشارة')=='مقبول' else 'background-color: #f8d4d4'
    return [color]*len(row)

st.dataframe(df.style.apply(highlight_rows, axis=1))

if (df["حالة الإشارة"]=="مقبول").any():
    st.success("تم تسجيل الصفقات وتحسين ترتيب الإشارات!")
else:
    st.info("لم يتم تسجيل أي صفقة جديدة، لكن تم فحص السوق!")
