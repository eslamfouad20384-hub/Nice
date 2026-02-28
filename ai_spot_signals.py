import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import time
from datetime import datetime, timedelta

# ====== المجلدات ======
CACHE = "cache"
MODEL = "model"
TRADE_LOG = "trades.csv"
HISTORICAL = "historical_data"

for folder in [CACHE, MODEL, HISTORICAL]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ===============================
# جلب أفضل العملات من CoinGecko بما فيها USDC
# ===============================
def get_top_symbols(limit=20):
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency":"usd","order":"volume_desc","per_page":limit,"page":1}
        data = requests.get(url, params=params, timeout=10).json()
        symbols = [item["symbol"].upper() for item in data]
        return symbols
    except:
        return []

# ===============================
# جلب بيانات OHLCV من CryptoCompare مع fallback لCoinGecko
# ===============================
def fetch_ohlcv(symbol, interval="4h", limit=200):
    base = "https://min-api.cryptocompare.com/data/v2/"
    fsym = symbol
    tsym = "USDT"
    url = f"{base}{'histohour' if interval=='4h' else 'histoday'}?fsym={fsym}&tsym={tsym}&limit={limit}"
    try:
        data = requests.get(url).json()
        if data.get("Response") == "Success":
            df = pd.DataFrame(data["Data"]["Data"])
            hist_file = os.path.join(HISTORICAL, f"{symbol}_{interval}.csv")
            df.to_csv(hist_file, index=False)
            return df
    except:
        pass

    # لو فشل CryptoCompare (خصوصاً Daily)، نجرب CoinGecko
    if interval == "daily":
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

    return pd.DataFrame()

# ===============================
# إضافة المؤشرات + ATR حقيقي
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
# إعادة تدريب AI أسبوعيًا
# ===============================
def weekly_retrain():
    if not os.path.exists(TRADE_LOG):
        return
    df = pd.read_csv(TRADE_LOG)
    if df.empty:
        return
    last_train_file = os.path.join(CACHE,"last_train.txt")
    if os.path.exists(last_train_file):
        with open(last_train_file,"r") as f:
            last_train_date = datetime.fromisoformat(f.read().strip())
        if datetime.now() - last_train_date < timedelta(days=7):
            return
    symbols = df["العملة"].unique()
    for sym in symbols:
        hist_file = os.path.join(HISTORICAL, f"{sym}_daily.csv")
        if os.path.exists(hist_file):
            df_hist = pd.read_csv(hist_file)
            df_hist = add_indicators(df_hist)
            train_ai(df_hist, sym)
    with open(last_train_file,"w") as f:
        f.write(datetime.now().isoformat())

# ===============================
# حالة السوق لعملة
# ===============================
def market_condition(symbol):
    df = fetch_ohlcv(symbol,"daily",200)
    if df.empty:
        return "غير متاح"
    df = add_indicators(df)
    last = df.iloc[-1]
    if last["close"] > last["EMA50"] > last["EMA200"]:
        return "صاعد"
    elif last["close"] < last["EMA50"] < last["EMA200"]:
        return "هابط"
    else:
        return "عرضي"

# ===============================
# حالة السوق العام
# ===============================
def overall_market(symbols):
    counts = {"صاعد":0,"هابط":0,"عرضي":0}
    for s in symbols:
        state = market_condition(s)
        if state in counts:
            counts[state] += 1
    total = sum(counts.values())
    if total == 0:
        return "غير متاح"
    best = max(counts, key=lambda k: counts[k])
    return f"{best} ({counts[best]}/{total})"

# ===============================
# تسجيل الصفقة
# ===============================
def log_trade(trade):
    if not os.path.exists(TRADE_LOG):
        df = pd.DataFrame(columns=list(trade.keys()))
        df = df.append(trade, ignore_index=True)
        df.to_csv(TRADE_LOG, index=False)
    else:
        df = pd.read_csv(TRADE_LOG)
        df = df.append(trade, ignore_index=True)
        df.to_csv(TRADE_LOG, index=False)

# ===============================
# توليد الإشارة مع تسجيل كل الإشارات
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
    if last["close"] < dfd["EMA50"].iloc[-1] and last["close"] < dfd["EMA200"].iloc[-1]:
        return {"العملة": symbol, "دخول": np.nan, "وقف": np.nan, "هدف": np.nan,
                "احتمال_الصعود": np.nan, "حالة_السوق": np.nan,
                "حالة الإشارة": "مرفوض", "سبب": "السعر تحت EMA50 و EMA200 يومي"}

    prob = train_ai(df,symbol)
    entry = last["close"]
    atr = last["ATR"]
    stop = entry - atr*1.2
    target = entry + atr*1.8

    if prob < 0.55:
        trade_status = "مرفوض"
        reason = f"قوة AI ضعيفة ({round(prob*100,2)}%)"
    else:
        trade_status = "مقبول"
        reason = ""

    trade = {"العملة":symbol, "تاريخ":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "دخول":round(entry,4) if trade_status=="مقبول" else np.nan,
             "وقف":round(stop,4) if trade_status=="مقبول" else np.nan,
             "هدف":round(target,4) if trade_status=="مقبول" else np.nan,
             "احتمال_الصعود":round(prob*100,2),
             "حالة_السوق":market_condition(symbol),
             "حالة الإشارة": trade_status,
             "سبب": reason}
    log_trade(trade)
    return trade

# ===============================
# سكان السوق مع ترقيم من 1
# ===============================
def scan_market():
    weekly_retrain()
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
# واجهة Streamlit بدون جدول فاضي عند الفتح
# ===============================
st.markdown('<h4 style="font-size:16px;">AI Spot Scanner</h4>', unsafe_allow_html=True)
symbols = get_top_symbols(20)
st.markdown(f"### 🧭 حالة السوق العام: {overall_market(symbols)}")

def highlight_rows(row):
    color = 'background-color: #d4f8d4' if row.get('حالة الإشارة')=='مقبول' else 'background-color: #f8d4d4'
    return [color]*len(row)

# زرار لإعادة الفحص يدويًا
if st.button("🔍 فحص السوق مرة أخرى"):
    df = scan_market()
    st.dataframe(df.style.apply(highlight_rows, axis=1))
    if (df["حالة الإشارة"]=="مقبول").any():
        st.success("تم تسجيل الصفقات وتحسين ترتيب الإشارات!")
    else:
        st.info("لم يتم تسجيل أي صفقة جديدة، لكن تم فحص السوق!")
