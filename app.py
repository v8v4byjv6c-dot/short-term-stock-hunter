
import io
import re
import math
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="短期上昇株ハンター v19.6.3",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

JPX_LIST_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
TPM_ISSUES_PAGE = "https://www.jpx.co.jp/equities/products/tpm/issues/index.html"

st.markdown("""
<style>
.block-container {max-width: 1320px; padding-top: 1rem; padding-bottom: 3rem;}
h1,h2,h3 {letter-spacing:-0.02em;}
.market-note {font-size:.92rem; opacity:.82;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 共通
# ------------------------------------------------------------
def tick_size(price):
    # 簡易呼値。厳密な呼値は銘柄/価格帯で異なるため実注文時に証券会社で確認。
    if price < 3000: return 1
    if price < 5000: return 5
    if price < 10000: return 10
    if price < 30000: return 10
    if price < 50000: return 50
    if price < 100000: return 100
    return 100

def adaptive_limit_buffer(price, atr_yen):
    """逆指値発動後の指値許容幅。固定ティックではなく値動きに合わせて自動計算。"""
    tick = tick_size(price)
    atr_part = 0.15 * atr_yen if np.isfinite(atr_yen) and atr_yen > 0 else price * 0.003
    raw = max(2 * tick, min(atr_part, price * 0.003))
    return math.ceil(raw / tick) * tick


def candidate_limit_buffer(price, atr_yen, policy):
    """逆指値発動後の買い指値許容幅を比較検証する候補式。"""
    tick = tick_size(price)
    floor = 2 * tick

    if policy == "2ティック固定":
        raw = floor
    elif policy == "0.10%":
        raw = max(floor, price * 0.001)
    elif policy == "0.20%":
        raw = max(floor, price * 0.002)
    elif policy == "0.30%":
        raw = max(floor, price * 0.003)
    elif policy == "0.10ATR・上限0.30%":
        atr_part = 0.10 * atr_yen if np.isfinite(atr_yen) and atr_yen > 0 else price * 0.003
        raw = max(floor, min(atr_part, price * 0.003))
    elif policy == "現行 0.15ATR・上限0.30%":
        atr_part = 0.15 * atr_yen if np.isfinite(atr_yen) and atr_yen > 0 else price * 0.003
        raw = max(floor, min(atr_part, price * 0.003))
    elif policy == "0.20ATR・上限0.30%":
        atr_part = 0.20 * atr_yen if np.isfinite(atr_yen) and atr_yen > 0 else price * 0.003
        raw = max(floor, min(atr_part, price * 0.003))
    else:
        raw = adaptive_limit_buffer(price, atr_yen)

    return math.ceil(raw / tick) * tick


def backtest_breakout_limit_buffer(d, slope_days=20, breakout_days=60):
    """
    ブレイク準備中の過去シグナルから、逆指値発動後の指値許容幅を検証。
    日足OHLCによる推定であり、板・瞬間的な価格飛びは完全再現できない。
    """
    if d is None or len(d) < max(120, breakout_days + 80):
        return None

    x = d.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c in x.columns and isinstance(x[c], pd.DataFrame):
            x[c] = x[c].iloc[:,0]
    x = x.dropna(subset=["Open","High","Low","Close","Volume"]).copy()
    if len(x) < max(120, breakout_days + 80):
        return None

    prev_close = x.Close.shift(1)
    tr = pd.concat([
        x.High - x.Low,
        (x.High - prev_close).abs(),
        (x.Low - prev_close).abs()
    ], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()

    x["MA75"] = x.Close.rolling(75).mean()
    x["Slope75"] = (x.MA75 / x.MA75.shift(slope_days) - 1) * 100
    x["V20"] = x.Volume.rolling(20).mean()
    x["VR"] = x.Volume / x.V20
    x["PrevHigh"] = x.High.shift(1).rolling(breakout_days).max()
    x["Ext"] = (x.Close / x.PrevHigh - 1) * 100

    prep = (
        (x.Slope75 > 0) &
        (x.Ext >= -3) & (x.Ext < 0) &
        (x.VR >= 1.1) &
        x.PrevHigh.notna() &
        x.ATR14.notna()
    )

    records = []
    idx_positions = np.where(prep.fillna(False).to_numpy())[0]
    last_signal_pos = -999

    for p in idx_positions:
        if p - last_signal_pos < 5:
            continue
        if p + 1 >= len(x):
            continue

        signal = x.iloc[p]
        nxt = x.iloc[p+1]
        trigger = float(signal.PrevHigh) + tick_size(float(signal.PrevHigh))
        atr = float(signal.ATR14)

        if float(nxt.High) < trigger:
            last_signal_pos = p
            continue

        open_gap = max(0.0, float(nxt.Open) - trigger)
        required_pct = (open_gap / trigger) * 100 if trigger else np.nan

        rec = {
            "trigger": trigger,
            "atr": atr,
            "open": float(nxt.Open),
            "high": float(nxt.High),
            "low": float(nxt.Low),
            "required_buffer_yen": open_gap,
            "required_buffer_pct": required_pct,
        }

        for policy in [
            "2ティック固定","0.10%","0.20%","0.30%",
            "0.10ATR・上限0.30%","現行 0.15ATR・上限0.30%","0.20ATR・上限0.30%"
        ]:
            buf = candidate_limit_buffer(trigger, atr, policy)
            limit = trigger + buf
            if float(nxt.Open) >= trigger:
                immediate = float(nxt.Open) <= limit
                estimated_fill = immediate or (float(nxt.Low) <= limit)
            else:
                immediate = True
                estimated_fill = True

            rec[f"{policy}__buffer"] = buf
            rec[f"{policy}__immediate"] = bool(immediate)
            rec[f"{policy}__fill"] = bool(estimated_fill)

        records.append(rec)
        last_signal_pos = p

    if not records:
        return None
    return pd.DataFrame(records)


def summarize_buffer_backtests(frames):
    """複数銘柄の逆指値バッファ検証を集約。"""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return None, None

    bt = pd.concat(frames, ignore_index=True)
    policies = [
        "2ティック固定","0.10%","0.20%","0.30%",
        "0.10ATR・上限0.30%","現行 0.15ATR・上限0.30%","0.20ATR・上限0.30%"
    ]

    rows = []
    for policy in policies:
        buf = bt[f"{policy}__buffer"]
        imm = bt[f"{policy}__immediate"]
        fill = bt[f"{policy}__fill"]
        rows.append({
            "方式": policy,
            "発動件数": int(len(bt)),
            "寄付き即時約定率%": float(imm.mean()*100),
            "日中推定約定率%": float(fill.mean()*100),
            "平均許容幅円": float(buf.mean()),
            "平均許容幅%": float((buf / bt["trigger"] * 100).mean()),
            "指値超え寄付き率%": float((bt["open"] > (bt["trigger"] + buf)).mean()*100),
        })

    summary = pd.DataFrame(rows)
    eligible = summary[summary["日中推定約定率%"] >= 95].copy()
    if not eligible.empty:
        recommended = eligible.sort_values(["平均許容幅%","寄付き即時約定率%"], ascending=[True,False]).iloc[0]["方式"]
    else:
        recommended = summary.sort_values("日中推定約定率%", ascending=False).iloc[0]["方式"]

    gap_stats = {
        "発動件数": int(len(bt)),
        "必要幅中央値%": float(bt["required_buffer_pct"].median()),
        "必要幅90%点%": float(bt["required_buffer_pct"].quantile(.90)),
        "必要幅95%点%": float(bt["required_buffer_pct"].quantile(.95)),
        "必要幅最大%": float(bt["required_buffer_pct"].max()),
        "参考推奨方式": recommended,
    }
    return summary, gap_stats

def mark(v):
    return "🟢" if bool(v) else "－"

def normalize_code(x):
    m = re.search(r"([0-9A-Z]{4})", str(x).strip().upper())
    return m.group(1) if m else None

# ------------------------------------------------------------
# JPXユニバース
# ------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_jpx_universe():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(JPX_LIST_PAGE, headers=headers, timeout=30)
    r.raise_for_status()
    hrefs = re.findall(r'href=["\']([^"\']+\.(?:xlsx?|XLSX?))["\']', r.text)
    if not hrefs:
        raise RuntimeError("JPXの上場銘柄一覧Excelが見つかりません。")
    preferred = [h for h in hrefs if "data_j" in h.lower()]
    url = urljoin(JPX_LIST_PAGE, (preferred or hrefs)[0])

    x = requests.get(url, headers=headers, timeout=60)
    x.raise_for_status()
    df = pd.read_excel(io.BytesIO(x.content), sheet_name=0)

    code_col = next((c for c in df.columns if "コード" in str(c)), None)
    name_col = next((c for c in df.columns if "銘柄名" in str(c) or "会社名" in str(c)), None)
    market_col = next((c for c in df.columns if "市場・商品区分" in str(c) or "市場区分" in str(c)), None)
    sector_col = next((c for c in df.columns if "33業種" in str(c) or "業種区分" in str(c)), None)

    if code_col is None or name_col is None:
        raise RuntimeError("JPX Excelの列構成を認識できません。")

    u = pd.DataFrame({
        "コード": df[code_col].map(normalize_code),
        "銘柄名": df[name_col].astype(str).str.strip(),
        "市場": df[market_col].astype(str).str.strip() if market_col is not None else "",
        "業種": df[sector_col].astype(str).str.strip() if sector_col is not None else "",
    }).dropna(subset=["コード"]).drop_duplicates("コード")

    # 普通株式系。ETF/REIT等は除外。
    non_common = (
        u["市場"].str.contains(r"ETF|ETN|REIT|投資法人|インフラ|出資証券|優先出資|外国", case=False, regex=True, na=False)
        | u["銘柄名"].str.contains(r"ETF|ETN|REIT|投資法人|インフラファンド|優先出資", case=False, regex=True, na=False)
    )
    u = u[~non_common].copy()

    def market_bucket(m):
        s = str(m)
        if "プライム" in s: return "プライム"
        if "スタンダード" in s: return "スタンダード"
        if "グロース" in s: return "グロース"
        if "PRO" in s.upper() or "プロマーケット" in s or "TOKYO PRO" in s.upper(): return "TOKYO PRO"
        return "その他"

    u["市場区分"] = u["市場"].map(market_bucket)
    u["ticker"] = u["コード"] + ".T"
    u["銘柄"] = u["コード"] + " " + u["銘柄名"]
    u["Yahoo!チャート"] = "https://finance.yahoo.co.jp/quote/" + u["ticker"] + "/chart"
    return u.reset_index(drop=True)

@st.cache_data(ttl=86400, show_spinner=False)
def get_tpm_fallback():
    """メインExcelにTOKYO PROが無い場合の補助。
    JPXのTOKYO PRO Marketページに掲載される銘柄を可能な範囲で取得。"""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        tables = pd.read_html(TPM_ISSUES_PAGE)
    except Exception:
        return pd.DataFrame(columns=["コード","銘柄名","市場","業種","市場区分","ticker","銘柄","Yahoo!チャート"])

    rows = []
    for t in tables:
        flat = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        t = t.copy()
        t.columns = flat
        code_col = next((c for c in t.columns if "コード" in c), None)
        name_col = next((c for c in t.columns if "銘柄名" in c or "会社名" in c), None)
        if code_col and name_col:
            for _, r in t.iterrows():
                code = normalize_code(r[code_col])
                name = str(r[name_col]).strip()
                if code and name and name.lower() != "nan":
                    rows.append((code, name))
    if not rows:
        return pd.DataFrame(columns=["コード","銘柄名","市場","業種","市場区分","ticker","銘柄","Yahoo!チャート"])

    u = pd.DataFrame(rows, columns=["コード","銘柄名"]).drop_duplicates("コード")
    u["市場"] = "TOKYO PRO Market"
    u["業種"] = ""
    u["市場区分"] = "TOKYO PRO"
    u["ticker"] = u["コード"] + ".T"
    u["銘柄"] = u["コード"] + " " + u["銘柄名"]
    u["Yahoo!チャート"] = "https://finance.yahoo.co.jp/quote/" + u["ticker"] + "/chart"
    return u

def select_market_universe(all_u, market):
    if market == "全市場":
        return all_u[all_u["市場区分"].isin(["プライム","スタンダード","グロース"])].copy()
    if market == "TOKYO PRO":
        x = all_u[all_u["市場区分"] == "TOKYO PRO"].copy()
        if x.empty:
            x = get_tpm_fallback()
        return x
    return all_u[all_u["市場区分"] == market].copy()

# ------------------------------------------------------------
# Yahoo Finance 株価
# ------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def download_batch(tickers, period):
    """
    v19.6.3: 全対象を1回の yf.download() で取得。
    100銘柄ずつの逐次16回通信を廃止し、取得フェーズの長時間化を防ぐ。
    repair=False: repair=True は追加の価格修復処理を伴い大規模一括取得では重いため無効。
    """
    tickers=list(dict.fromkeys(tickers))
    out={}
    if not tickers:
        return out
    try:
        d=yf.download(
            tickers,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            repair=False,
            progress=False,
            threads=True,
            timeout=15,
        )
        if d is None or d.empty:
            return out
        if len(tickers)==1 and not isinstance(d.columns,pd.MultiIndex):
            out[tickers[0]]=d.dropna(how="all").copy()
        elif isinstance(d.columns,pd.MultiIndex):
            l0=set(d.columns.get_level_values(0))
            l1=set(d.columns.get_level_values(1))
            if any(t in l0 for t in tickers):
                for t in tickers:
                    if t in l0:
                        x=d[t].dropna(how="all")
                        if not x.empty: out[t]=x
            else:
                for t in tickers:
                    if t in l1:
                        x=d.xs(t,axis=1,level=1).dropna(how="all")
                        if not x.empty: out[t]=x
    except Exception:
        pass
    return out

@st.cache_data(ttl=120, show_spinner=False)
def latest_jp_market_date():
    try:
        d=yf.download("^N225",period="10d",interval="1d",auto_adjust=False,repair=False,progress=False,threads=False,timeout=8)
        if d is None or d.empty: return None
        return pd.Timestamp(d.index[-1]).tz_localize(None).date()
    except Exception: return None

def _frame_last_date(d):
    try: return pd.Timestamp(d.dropna(how="all").index[-1]).tz_localize(None).date()
    except Exception: return None

def refresh_lagging_tickers(data,tickers,market_date):
    """
    v19.6.3: 高速版。
    全銘柄の個別再取得は行わない。一括取得結果の日付だけ監査し、
    市場基準日より古い銘柄を返す。古い銘柄はランキングから除外する。
    """
    if market_date is None:
        return data,[]
    stale=[
        t for t in tickers
        if data.get(t) is None
        or _frame_last_date(data.get(t)) is None
        or _frame_last_date(data.get(t)) < market_date
    ]
    return data,stale

# ------------------------------------------------------------
# A/B/D
# ------------------------------------------------------------
def technical_scan(d, slope_days, max_dev, breakout_days, a_stop_buffer_pct, buy_ticks):
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    need = ["Open","High","Low","Close","Volume"]
    if not all(c in d.columns for c in need):
        return None

    d = d[need].dropna().copy()
    if len(d) < max(100, 75+slope_days+2, breakout_days+2):
        return None

    d["MA25"] = d.Close.rolling(25).mean()
    d["MA75"] = d.Close.rolling(75).mean()
    d["OLD"] = d.MA75.shift(slope_days)
    d["DEV"] = (d.Close/d.MA75 - 1)*100
    d["R5"] = d.Close.pct_change(5)*100
    d["R20"] = d.Close.pct_change(20)*100
    d["R60"] = d.Close.pct_change(60)*100
    d["BREAK_HIGH"] = d.High.shift(1).rolling(breakout_days).max()
    d["V20"] = d.Volume.rolling(20).mean()
    d["VR"] = d.Volume/d.V20
    prev_close = d.Close.shift(1)
    tr = pd.concat([
        d.High - d.Low,
        (d.High - prev_close).abs(),
        (d.Low - prev_close).abs()
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()

    x, p = d.iloc[-1], d.iloc[-2]
    try:
        data_last_date = pd.Timestamp(d.index[-1]).tz_localize(None).date()
    except Exception:
        try:
            data_last_date = pd.Timestamp(d.index[-1]).date()
        except Exception:
            data_last_date = None
    vals = [x.Close,x.High,x.MA25,x.MA75,x.OLD,x.DEV,x.R5,x.R20,x.R60]
    if not all(np.isfinite(float(v)) for v in vals):
        return None

    close=float(x.Close); high=float(x.High); ma25=float(x.MA25); ma75=float(x.MA75)
    prev_close=float(p.Close)
    day_change=close-prev_close
    day_change_pct=(close/prev_close-1)*100 if prev_close else np.nan
    open_today=float(x.Open); high_today=float(x.High); low_today=float(x.Low)
    slope=(ma75/float(x.OLD)-1)*100
    dev=float(x.DEV); r5=float(x.R5); r20=float(x.R20); r60=float(x.R60)
    vr=float(x.VR) if np.isfinite(x.VR) else 0
    bh=float(x.BREAK_HIGH) if np.isfinite(x.BREAK_HIGH) else np.nan
    atr14=float(x.ATR14) if np.isfinite(x.ATR14) else np.nan

    trend = slope > 0
    # Aは参考書どおり「75日線より下」にある銘柄だけを候補にする。
    # そのうえで、75日線とのマイナス乖離が小さいほど高評価。
    below_75 = close < ma75
    # v16 本ベースA：上向き75日線の「すぐ下」だけを対象にする。
    near = -3.0 <= dev < 0.0
    bullish = close > float(p.Close) and close > float(x.Open)

    A = trend and below_75 and near
    # 0%直下ほど高評価。-3%で70点、0%直下で100点。
    dist = max(0, min(100, 100 + dev*10))
    sl = min(100,max(0,slope/5*100))
    As = min(100,max(0,.65*dist+.35*sl)) if A else 0

    # 75日線付近に来ただけでは買わず、前日高値超えで反転確認。
    a_buy = float(p.High) + tick_size(float(p.High))*buy_ticks
    a_stop = ma75*(1-a_stop_buffer_pct/100)

    B = bool(trend and np.isfinite(bh) and close > bh and vr >= 1.3)
    # ブレイク水準から上がりすぎている場合は「追いかけ買い」リスクとして減点。
    breakout_extension = ((close / bh) - 1) * 100 if np.isfinite(bh) and bh > 0 else np.nan
    extension_penalty = 0
    if np.isfinite(breakout_extension):
        if breakout_extension > 10:
            extension_penalty = 30
        elif breakout_extension > 7:
            extension_penalty = 20
        elif breakout_extension > 5:
            extension_penalty = 12
        elif breakout_extension > 3:
            extension_penalty = 5

    # 75日線から大きく上方乖離しているBは、中期的な過熱・高値掴みリスクとして追加減点。
    ma75_penalty = 0
    if dev > 30:
        ma75_penalty = 30
    elif dev > 20:
        ma75_penalty = 20
    elif dev > 10:
        ma75_penalty = 10

    Bs = min(
        100,
        max(
            0,
            min(45,max(0,r20*2))
            + min(30,max(0,(vr-1)*30))
            + (25 if B else 0)
            - extension_penalty
            - ma75_penalty
        )
    )

    recent = float(d.Close.tail(20).max())
    drawdown = (close/recent-1)*100
    D = bool(trend and r60>=10 and drawdown<=-3 and r5>0 and close>ma25)
    Ds = min(100,max(0,min(45,max(0,r60*1.5))+min(25,max(0,r5*4))+(30 if D else 0)))
    bd_buy = high + tick_size(high)
    # v19.1 Bの損切りは「ブレイク水準を支持線として維持できるか」を主根拠にする。
    # ATRは損切り理由そのものではなく、日々のノイズで刈られにくくする補助バッファ。
    b_stop_buffer = 0.5 * atr14 if np.isfinite(atr14) else (bh * 0.015 if np.isfinite(bh) else np.nan)
    b_stop = (bh - b_stop_buffer) if np.isfinite(bh) and np.isfinite(b_stop_buffer) else np.nan

    return {
        "データ最終日":data_last_date,
        "株価":close, "前日終値":prev_close, "前日比":day_change, "前日比%":day_change_pct,
        "始値":open_today, "高値":high_today, "安値":low_today, "75日線":ma75,
        "75日線_比較期間前比%":slope, "75日線_乖離率%":dev,
        "出来高_20日平均比":vr, "20日騰落率%":r20, "60日騰落率%":r60,
        "5日騰落率%":r5,
        "25日線":ma25,
        "株価25日線比%":(close/ma25-1)*100,
        "ATR14%":(atr14/close*100) if np.isfinite(atr14) and close else np.nan,
        "売買代金_億円":(close*float(x.Volume)/100000000) if np.isfinite(float(x.Volume)) else np.nan,
        "ブレイク水準からの上昇率%":breakout_extension,
        "A":A, "Aスコア":As, "A買い価格":a_buy, "A初期損切り":a_stop,
        "B":B, "Bスコア":Bs, "B買い価格":bd_buy,
        "Bブレイク水準":bh,
        "Bブレイク上昇率%":breakout_extension,
        "B出来高倍率":vr,
        "B_75日線過熱減点":ma75_penalty,
        "ATR14":atr14,
        "B初期損切り":b_stop,
        "D":D, "Dスコア":Ds, "D買い価格":bd_buy,
    }


# ------------------------------------------------------------
# v19.6.3 銘柄診断・データ鮮度
# ------------------------------------------------------------
def business_day_age(last_date):
    """今日までの平日ベースの概算経過日数。祝日は考慮しないため警告用の目安。"""
    if last_date is None or pd.isna(last_date):
        return None
    try:
        today = pd.Timestamp.now(tz="Asia/Tokyo").date()
    except Exception:
        today = pd.Timestamp.now().date()
    try:
        a = pd.Timestamp(last_date).date()
        if a >= today:
            return 0
        return max(0, len(pd.bdate_range(pd.Timestamp(a) + pd.Timedelta(days=1), pd.Timestamp(today))))
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_diagnostic_data(ticker, period="1y"):
    """
    指定銘柄を個別再取得。
    yfinance の end は排他的なので、明示的な日付指定を使う場合は翌日まで取る設計にする。
    ここでは period 指定で最新日足を取り直す。
    """
    try:
        d = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=False, repair=True,
            progress=False, threads=False
        )
        if d is None or d.empty:
            return None
        if isinstance(d.columns, pd.MultiIndex):
            # 単一銘柄でもMultiIndexになるyfinance版への対策
            if ticker in set(d.columns.get_level_values(-1)):
                d = d.xs(ticker, axis=1, level=-1)
            else:
                d.columns = d.columns.get_level_values(0)
        return d.dropna(how="all")
    except Exception:
        return None


def diagnose_ai_ticker(ticker, universe, tech, slope_days, max_dev, breakout_days, a_stop_buffer_pct, buy_ticks):
    """
    指定銘柄が独自短期ランキングにいる/いない理由を診断する。
    ランキング本体とは独立して個別再取得し、順位圏外・データ取得・条件を切り分ける。
    """
    ticker = str(ticker).strip().upper()
    if ticker.isdigit():
        ticker += ".T"

    u = universe[universe["ticker"].astype(str).str.upper() == ticker]
    universe_hit = not u.empty
    name = u.iloc[0]["銘柄"] if universe_hit else ticker

    existing = tech[tech["ticker"].astype(str).str.upper() == ticker] if tech is not None and not tech.empty else pd.DataFrame()
    rank = int(existing.iloc[0]["順位"]) if (not existing.empty and "順位" in existing.columns) else None

    d = fetch_ticker_diagnostic_data(ticker)
    if d is None or d.empty:
        return {
            "銘柄": name, "ticker": ticker, "市場対象": universe_hit,
            "個別再取得": False, "現在順位": rank,
            "表示されない主因": "株価日足を個別再取得できませんでした。",
        }

    m = technical_scan(d, slope_days, max_dev, breakout_days, a_stop_buffer_pct, buy_ticks)
    if not m:
        return {
            "銘柄": name, "ticker": ticker, "市場対象": universe_hit,
            "個別再取得": True, "現在順位": rank,
            "表示されない主因": "必要な日足本数またはOHLCVが不足し、technical_scanを通過できませんでした。",
        }

    pre = ai_pre_score(pd.Series(m))
    tmp = pd.Series({**m, "C": False, "Cスコア": 0})
    ai = ai_scores(tmp)

    last_date = m.get("データ最終日")
    age = business_day_age(last_date)
    stale = age is not None and age >= 2

    ext = m.get("Bブレイク上昇率%", np.nan)
    slope = m.get("75日線_比較期間前比%", np.nan)
    vr = m.get("出来高_20日平均比", np.nan)

    if rank is not None:
        reason = f"ランキングには存在します（現在 {rank}位）。" + ("上位100位外なので通常表には出ません。" if rank > 100 else "上位100位内です。")
    elif not universe_hit:
        reason = "現在選択している市場ユニバースの対象外です。"
    elif stale:
        reason = "個別再取得した日足が古いため、ランキング判定を信用できません。データ鮮度警告です。"
    else:
        reason = "市場対象・日足取得とも正常です。現在の全体ランキングDataFrameに存在しないため、一括取得時の欠落/取得失敗を疑います。"

    breakout_price_ok = bool(np.isfinite(ext) and -3 <= ext <= 0)
    breakout_trend_ok = bool(np.isfinite(slope) and slope > 0)
    breakout_volume_ok = bool(np.isfinite(vr) and vr >= 1.1)

    return {
        "銘柄": name, "ticker": ticker,
        "市場対象": universe_hit, "個別再取得": True,
        "データ最終日": last_date, "データ平日経過": age,
        "データ鮮度警告": stale,
        "個別再取得株価": m.get("株価"),
        "現在順位": rank,
        "事前スコア参考": pre,
        "短期総合スコア参考": ai.get("短期総合スコア"),
        "セットアップ参考": ai.get("セットアップ"),
        "直近高値との差%": ext,
        "75日線傾き%": slope,
        "出来高20日平均比": vr,
        "ブレイク距離条件(-3〜0%)": breakout_price_ok,
        "75日線上向き": breakout_trend_ok,
        "出来高1.1倍以上": breakout_volume_ok,
        "ブレイク準備3条件": breakout_price_ok and breakout_trend_ok and breakout_volume_ok,
        "表示されない主因": reason,
    }


# ------------------------------------------------------------
# C
# ------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def financial_momentum(ticker):
    try:
        q = yf.Ticker(ticker).quarterly_financials
        if q is None or q.empty:
            return {}
        def pick(names):
            for n in names:
                if n in q.index:
                    return pd.to_numeric(q.loc[n],errors="coerce").dropna()
            return pd.Series(dtype=float)

        op = pick(["Operating Income"])
        rev = pick(["Total Revenue","Operating Revenue"])
        if len(op) < 5:
            return {}

        op_now, op_yoy = float(op.iloc[0]), float(op.iloc[4])
        og = (op_now/op_yoy-1)*100 if op_yoy > 0 else np.nan
        rg = np.nan
        if len(rev)>=5 and float(rev.iloc[4]) != 0:
            rg = (float(rev.iloc[0])/float(rev.iloc[4])-1)*100

        C = bool(np.isfinite(og) and og>=20 and (not np.isfinite(rg) or rg>=0))
        Cs = min(100,
                 (min(70,max(0,og)) if np.isfinite(og) else 0) +
                 (min(30,max(0,rg*2)) if np.isfinite(rg) else 0))
        return {"C":C,"Cスコア":Cs,"営業利益_前年同期比%":og,"売上高_前年同期比%":rg}
    except Exception:
        return {}


# ------------------------------------------------------------
# 長期・年初来安値モード
# ------------------------------------------------------------
def long_price_metrics(d):
    """現在年の年初来安値と、その安値からの距離を計算。"""
    try:
        if d is None or d.empty:
            return None
        x = d.copy()
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        if not all(c in x.columns for c in ["Close","Low","High"]):
            return None
        x = x.dropna(subset=["Close","Low"])
        if x.empty:
            return None

        idx = pd.DatetimeIndex(x.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        x.index = idx

        current_year = pd.Timestamp.now().year
        ytd = x[x.index >= pd.Timestamp(f"{current_year}-01-01")]
        if ytd.empty:
            return None

        current = float(ytd["Close"].iloc[-1])
        prev = float(ytd["Close"].iloc[-2]) if len(ytd) >= 2 else np.nan
        ytd_low = float(ytd["Low"].min())
        ytd_high = float(ytd["High"].max())
        low_date = ytd["Low"].idxmin().strftime("%Y-%m-%d")
        low_distance = (current / ytd_low - 1) * 100 if ytd_low > 0 else np.nan
        high_drawdown = (current / ytd_high - 1) * 100 if ytd_high > 0 else np.nan

        # 年初来安値に近いほど100点。20%以上離れたら0点。
        low_score = np.clip(100 - max(0, low_distance) * 5, 0, 100)

        return {
            "株価": current,
            "前日終値": prev,
            "前日比": current - prev if np.isfinite(prev) else np.nan,
            "前日比%": (current / prev - 1) * 100 if np.isfinite(prev) and prev else np.nan,
            "年初来安値": ytd_low,
            "年初来安値日": low_date,
            "年初来安値から%": low_distance,
            "年初来高値": ytd_high,
            "年初来高値から%": high_drawdown,
            "安値接近度": float(low_score),
        }
    except Exception:
        return None

@st.cache_data(ttl=21600, show_spinner=False)
def long_term_fundamentals(ticker):
    """
    yfinanceで取得できる範囲のファンダメンタル。
    取得不能項目は欠損のままにし、推測しない。
    """
    try:
        info = yf.Ticker(ticker).info or {}
        def num(key):
            v = info.get(key, np.nan)
            try:
                return float(v) if v is not None else np.nan
            except Exception:
                return np.nan

        market_cap = num("marketCap")
        roe = num("returnOnEquity")
        op_margin = num("operatingMargins")
        profit_margin = num("profitMargins")
        revenue_growth = num("revenueGrowth")
        earnings_growth = num("earningsGrowth")
        dividend_yield = num("dividendYield")
        forward_pe = num("forwardPE")
        trailing_pe = num("trailingPE")
        pbr = num("priceToBook")
        debt_equity = num("debtToEquity")
        payout = num("payoutRatio")

        # yfinanceの dividendYield は環境によって 0.03 / 3.0 の両方があり得るため正規化。
        if np.isfinite(dividend_yield) and dividend_yield <= 1:
            dividend_yield *= 100
        if np.isfinite(roe) and abs(roe) <= 1:
            roe *= 100
        if np.isfinite(op_margin) and abs(op_margin) <= 1:
            op_margin *= 100
        if np.isfinite(profit_margin) and abs(profit_margin) <= 1:
            profit_margin *= 100
        if np.isfinite(revenue_growth) and abs(revenue_growth) <= 1:
            revenue_growth *= 100
        if np.isfinite(earnings_growth) and abs(earnings_growth) <= 1:
            earnings_growth *= 100
        if np.isfinite(payout) and abs(payout) <= 1:
            payout *= 100

        return {
            "時価総額": market_cap,
            "時価総額_億円": market_cap / 1e8 if np.isfinite(market_cap) else np.nan,
            "ROE%": roe,
            "営業利益率%": op_margin,
            "純利益率%": profit_margin,
            "売上成長率%": revenue_growth,
            "利益成長率%": earnings_growth,
            "配当利回り%": dividend_yield,
            "予想PER": forward_pe,
            "実績PER": trailing_pe,
            "PBR": pbr,
            "負債資本倍率": debt_equity,
            "配当性向%": payout,
        }
    except Exception:
        return {}

def long_quality_scores(r):
    """長期用。安値だけでなく企業の質・財務・配当/割安度を分けて採点。"""
    def finite(name):
        v = r.get(name, np.nan)
        return float(v) if np.isfinite(v) else np.nan

    mcap = finite("時価総額_億円")
    roe = finite("ROE%")
    opm = finite("営業利益率%")
    rg = finite("売上成長率%")
    eg = finite("利益成長率%")
    dy = finite("配当利回り%")
    pe = finite("予想PER")
    pbr = finite("PBR")
    de = finite("負債資本倍率")
    proximity = finite("安値接近度")

    # 企業クオリティ
    q_parts = []
    if np.isfinite(mcap):
        q_parts.append(np.clip(35 + np.log10(max(mcap, 10)) * 14, 35, 95))
    if np.isfinite(roe):
        q_parts.append(np.clip(35 + roe * 3, 0, 100))
    if np.isfinite(opm):
        q_parts.append(np.clip(40 + opm * 3, 0, 100))
    if np.isfinite(rg):
        q_parts.append(np.clip(50 + rg * 2, 0, 100))
    if np.isfinite(eg):
        q_parts.append(np.clip(50 + eg * 1.3, 0, 100))
    if np.isfinite(de):
        # YahooのdebtToEquityは%ベースの場合が多い。低いほど加点。
        q_parts.append(np.clip(95 - max(0, de - 30) * 0.35, 20, 95))
    quality = float(np.mean(q_parts)) if q_parts else 45.0

    # 配当・バリュエーション。安いだけを過度に加点しない。
    v_parts = []
    if np.isfinite(dy):
        v_parts.append(np.clip(35 + dy * 12, 20, 100))
    if np.isfinite(pe) and pe > 0:
        v_parts.append(np.clip(105 - pe * 3, 20, 95))
    if np.isfinite(pbr) and pbr > 0:
        v_parts.append(np.clip(95 - max(0, pbr - 0.8) * 22, 20, 95))
    value = float(np.mean(v_parts)) if v_parts else 50.0

    proximity = proximity if np.isfinite(proximity) else 0
    long_score = 0.40 * proximity + 0.45 * quality + 0.15 * value

    low_dist = finite("年初来安値から%")
    earnings_bad = np.isfinite(eg) and eg < -30
    revenue_bad = np.isfinite(rg) and rg < -15
    quality_bad = quality < 42

    if np.isfinite(low_dist) and low_dist <= 5 and quality >= 68 and not earnings_bad:
        judgment = "🟢 長期候補"
    elif np.isfinite(low_dist) and low_dist <= 10 and quality >= 58 and not earnings_bad:
        judgment = "🟡 分割買い候補"
    elif np.isfinite(low_dist) and low_dist <= 5 and (quality_bad or earnings_bad or revenue_bad):
        judgment = "🔴 安い理由を要確認"
    elif quality >= 65:
        judgment = "🟠 良企業・価格待ち"
    else:
        judgment = "⚪ 監視"

    return pd.Series({
        "企業クオリティ": quality,
        "配当・割安度": value,
        "長期総合スコア": float(long_score),
        "長期判定": judgment,
    })

def long_buy_plan(r):
    """長期向けの分割購入参考プラン。短期の逆指値とは別思想。"""
    current = float(r["株価"])
    low = float(r["年初来安値"])
    dist = float(r["年初来安値から%"])
    judgment = str(r["長期判定"])

    if judgment.startswith("🔴") or judgment.startswith("⚪"):
        return pd.Series({
            "長期買い方": "今は買わず要因確認",
            "1回目": "—", "2回目": "—", "3回目": "—",
            "長期前提崩れ": "大幅下方修正・赤字定着・減配・財務急悪化などを再確認",
        })

    if dist <= 3:
        first = f"{current:.0f}円前後で30%"
    else:
        first_price = low * 1.03
        first = f"{first_price:.0f}円以下まで待って30%"

    second = f"{low:.0f}円前後で30%"
    third = f"{low*0.95:.0f}円前後で40%"

    return pd.Series({
        "長期買い方": "一括ではなく3回分割",
        "1回目": first,
        "2回目": second,
        "3回目": third,
        "長期前提崩れ": "大幅下方修正・赤字定着・減配・財務急悪化など",
    })







# ------------------------------------------------------------
# v19 ちょる子式｜大型株逆張り
# ------------------------------------------------------------
def calc_rci(close, period=9):
    s=pd.Series(close).astype(float)
    out=pd.Series(index=s.index,dtype=float)
    if len(s)<period:
        return out
    date_rank=np.arange(1,period+1,dtype=float)
    for i in range(period-1,len(s)):
        w=s.iloc[i-period+1:i+1]
        price_rank=w.rank(method="average").to_numpy(dtype=float)
        d=date_rank-price_rank
        out.iloc[i]=(1-6*np.sum(d*d)/(period*(period*period-1)))*100
    return out

def choruko_metrics(d):
    try:
        x=d.copy()
        if isinstance(x.columns,pd.MultiIndex):
            x.columns=x.columns.get_level_values(0)
        need=["Open","High","Low","Close","Volume"]
        if not all(c in x.columns for c in need):
            return None
        x=x[need].dropna().copy()
        if len(x)<40:
            return None
        x["MA25"]=x.Close.rolling(25).mean()
        sd=x.Close.rolling(25).std(ddof=0)
        x["BB_Z"]=(x.Close-x.MA25)/sd.replace(0,np.nan)
        x["RCI9"]=calc_rci(x.Close,9)
        x["V20"]=x.Volume.rolling(20).mean()

        cur=x.iloc[-1]; prev=x.iloc[-2]
        close=float(cur.Close); prev_close=float(prev.Close)
        day_pct=(close/prev_close-1)*100 if prev_close else np.nan
        ma25=float(cur.MA25) if np.isfinite(cur.MA25) else np.nan
        dev25=(close/ma25-1)*100 if np.isfinite(ma25) and ma25 else np.nan
        bbz=float(cur.BB_Z) if np.isfinite(cur.BB_Z) else np.nan
        rci=float(cur.RCI9) if np.isfinite(cur.RCI9) else np.nan

        s1=np.isfinite(day_pct) and day_pct<=-2.5
        # 「25日線を大きく割り込む」の厳密な％は手元資料では不明。
        # 本ルールとして勝手な乖離閾値を追加せず、25日線より下か＋乖離率を表示。
        s2=np.isfinite(dev25) and dev25<0
        s3=np.isfinite(bbz) and bbz<=-3.0
        s4=np.isfinite(rci) and rci<=-90.0
        count=int(s1)+int(s2)+int(s3)+int(s4)

        # アプリ独自補助：陽線化または前日高値突破
        reversal=(close>float(cur.Open)) or (close>float(prev.High))

        return {
            "株価":close,"現在値":close,"前日終値":prev_close,"前日比%":day_pct,
            "25日線":ma25,"25日線乖離%":dev25,"BBσ":bbz,"RCI9":rci,
            "急落-2.5%":bool(s1),"25日線割れ":bool(s2),"BB-3σ":bool(s3),"RCI-90":bool(s4),
            "底打ち条件数":count,"反転確認_独自":bool(reversal),"急落前水準":prev_close,
        }
    except Exception:
        return None

def choruko_judgment(r, material_status):
    n=int(r.get("底打ち条件数",0))
    rev=bool(r.get("反転確認_独自",False))
    if material_status=="悪材料あり":
        return "🔴 対象外","悪材料あり。『材料なし急落』の前提から外れます。"
    if n==4 and rev and material_status=="悪材料なし確認済":
        return "🟢 最有力候補","4/4条件＋反転確認＋悪材料なし確認済。"
    if n>=3:
        return "🟡 逆張り候補",f"{n}/4条件。材料確認と反転確認を優先。"
    if n>=2:
        return "🟠 監視",f"{n}/4条件。売られすぎ条件がまだ不足。"
    return "⚪ 対象外寄り",f"{n}/4条件のみ。"

def choruko_exit_plan(r):
    close=float(r["現在値"])
    ma25=float(r["25日線"]) if np.isfinite(r.get("25日線",np.nan)) else np.nan
    prev=float(r["急落前水準"]) if np.isfinite(r.get("急落前水準",np.nan)) else np.nan
    candidates=[]
    if np.isfinite(ma25) and ma25>close: candidates.append(("25日線回帰",ma25))
    if np.isfinite(prev) and prev>close: candidates.append(("急落前終値回帰",prev))
    candidates=sorted(candidates,key=lambda x:x[1])
    return {
        "利確参考①":candidates[0][1] if len(candidates)>0 else np.nan,
        "利確参考①根拠":candidates[0][0] if len(candidates)>0 else "—",
        "利確参考②":candidates[1][1] if len(candidates)>1 else np.nan,
        "利確参考②根拠":candidates[1][0] if len(candidates)>1 else "—",
        "損切り方針":"反発せず、買った前提（支持・反転）が崩れたら損切りを再判断。",
    }


# ------------------------------------------------------------
# v18 保有銘柄・個別分析
# ------------------------------------------------------------
def holding_management_analysis(r, buy_price=None, shares=None):
    """
    保有者目線のルールベース診断。
    新規買い評価とは分け、買値・現在値・移動平均・ATRから
    保有継続/注意/利確検討/トレンド崩れを整理する。
    """
    current=float(r["株価"])
    ma25=float(r["25日線"])
    ma75=float(r["75日線"])
    slope=float(r["75日線_比較期間前比%"])
    dev=float(r["75日線_乖離率%"])
    r5=float(r["5日騰落率%"])
    r20=float(r["20日騰落率%"])
    atr=float(r["ATR14"]) if np.isfinite(r.get("ATR14",np.nan)) else current*0.03
    recent_high=float(r["高値"]) if np.isfinite(r.get("高値",np.nan)) else current

    bp=float(buy_price) if buy_price is not None and float(buy_price)>0 else np.nan
    qty=int(shares) if shares is not None and float(shares)>0 else 0

    pnl_pct=(current/bp-1)*100 if np.isfinite(bp) else np.nan
    pnl_yen=(current-bp)*qty if np.isfinite(bp) and qty>0 else np.nan

    # 保有管理用の逆指値参考。
    # 上昇トレンド中は25日線を優先し、弱い場合は75日線を基準にする。
    if current > ma25 and slope > 0:
        trend_stop = ma25 - 0.5*atr
        stop_basis = "25日線 - 0.5ATR"
    else:
        trend_stop = ma75 - 0.5*atr
        stop_basis = "75日線 - 0.5ATR"

    # 現在値より上になることは避ける。
    manage_stop=min(trend_stop, current-tick_size(current))
    if np.isfinite(bp):
        # 含み益がある場合は、買値から極端に遠い損切りになりすぎないよう
        # 2ATR下も候補にして高い方を採用。
        entry_risk_stop=bp-2*atr
        manage_stop=max(manage_stop, entry_risk_stop)
        manage_stop=min(manage_stop, current-tick_size(current))

    # 次の利確参考は現在値からATR基準。予測ではなく管理目安。
    take1=current+1.5*atr
    take2=current+3.0*atr

    if current <= ma75 and slope < 0:
        holding_grade="🔴 トレンド崩れ"
        holding_comment="株価が75日線以下で、75日線も下向き。保有継続の前提を再確認したい状態です。"
    elif current < ma25 and r5 < 0:
        holding_grade="🟠 要注意"
        holding_comment="25日線を下回り、直近5日も弱い状態。売り逆指値の位置を確認したい局面です。"
    elif np.isfinite(pnl_pct) and pnl_pct >= 5 and dev >= 12:
        holding_grade="🟡 一部利確を検討"
        holding_comment="含み益があり、75日線からの上方乖離も大きめ。全部売る判断ではなく、一部利確や逆指値引き上げを検討しやすい状態です。"
    elif slope > 0 and current > ma25:
        holding_grade="🟢 保有継続候補"
        holding_comment="75日線が上向きで株価も25日線より上。短期トレンドは維持していると判定します。"
    else:
        holding_grade="⚪ 中立・監視"
        holding_comment="明確な上昇継続・崩れのどちらにも寄っていません。チャートと次のトリガーを確認します。"

    return {
        "保有評価":holding_grade,
        "保有コメント":holding_comment,
        "含み損益%":pnl_pct,
        "含み損益円":pnl_yen,
        "管理用売り逆指値":manage_stop,
        "管理用逆指値根拠":stop_basis,
        "次の利確参考①":take1,
        "次の利確参考②":take2,
    }

def render_single_holding_analysis(all_u, code_value, buy_price=0.0, shares=0):
    code_norm=normalize_code(code_value)
    if not code_norm:
        st.error("4桁の銘柄コードを入力してください。")
        return None

    hit=all_u[all_u["コード"]==code_norm]
    if hit.empty:
        st.error(f"{code_norm} を東証銘柄一覧で確認できませんでした。")
        return None

    meta=hit.iloc[0]
    ticker=meta["ticker"]
    dmap=download_batch([ticker],"1y")
    d=dmap.get(ticker)
    if d is None or d.empty:
        st.error("株価データを取得できませんでした。")
        return None

    r=technical_scan(d,20,3.0,60,0.35,2)
    if not r:
        st.error("分析に必要な株価データが不足しています。")
        return None

    r.update({
        "ticker":ticker,
        "コード":code_norm,
        "銘柄名":meta["銘柄名"],
        "銘柄":meta["銘柄"],
        "Yahoo!チャート":meta["Yahoo!チャート"],
    })

    # 決算モメンタムと次回決算予定は1銘柄だけ取得。
    fin=financial_momentum(ticker)
    r.update(fin or {})
    if "C" not in r:
        r["C"]=False
    if "Cスコア" not in r:
        r["Cスコア"]=0

    ai=ai_scores(r)
    h=holding_management_analysis(r,buy_price,shares)
    earnings=get_earnings_date_info(ticker)

    st.subheader(f"🔎 {r['銘柄']}｜個別分析")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("現在値",f"{r['株価']:.0f}円",f"{r['前日比']:+.0f}円 / {r['前日比%']:+.2f}%")
    if buy_price and float(buy_price)>0:
        c2.metric("取得単価",f"{float(buy_price):.0f}円",f"{h['含み損益%']:+.2f}%")
    else:
        c2.metric("取得単価","未入力")
    if shares and int(shares)>0 and np.isfinite(h["含み損益円"]):
        c3.metric("保有株数",f"{int(shares):,}株",f"損益 {h['含み損益円']:+,.0f}円")
    else:
        c3.metric("保有株数","未入力")
    c4.metric("保有評価",h["保有評価"])

    st.info(f"**保有者目線：{h['保有評価']}**  {h['保有コメント']}")

    st.markdown("### 🧭 現在のセットアップ判定根拠")
    st.success(f"**{ai['セットアップ']}**")
    for reason in ai.get("セットアップ根拠一覧",[]):
        st.write(f"・{reason}")
    st.caption("この判定はルールベースです。『ブレイク準備中』は高値への接近・75日線の向き・出来高などの条件を満たした状態を表し、将来の上昇を保証するものではありません。")

    st.markdown("### 📌 新規買い評価と保有評価を分けて確認")
    summary=pd.DataFrame([{
        "現在のセットアップ":ai["セットアップ"],
        "新規買いのルール評価":ai["ルール評価"],
        "短期スコア":ai["短期総合スコア"],
        "今の買いやすさ":ai["今の買いやすさ"],
        "保有者としての評価":h["保有評価"],
        "決算警告":earnings.get("warning",""),
    }])
    st.dataframe(summary,use_container_width=True,hide_index=True)

    st.markdown("### 📱 楽天証券｜保有後の管理参考")
    manage=pd.DataFrame([{
        "売り逆指値参考":f"{h['管理用売り逆指値']:.0f}円以下",
        "根拠":h["管理用逆指値根拠"],
        "利確参考①":f"{h['次の利確参考①']:.0f}円",
        "利確参考②":f"{h['次の利確参考②']:.0f}円",
        "現在の新規買い注文":ai["注文種類"],
        "新規買い価格":ai["注文価格表示"],
    }])
    st.dataframe(manage,use_container_width=True,hide_index=True)

    with st.expander("🧮 新規買いの注文価格・損切り・利確の根拠"):
        st.markdown(f"**売買シナリオ：{ai['売買シナリオ']}**")
        st.markdown(f"**注文種類：{ai['注文種類']}**")
        st.write(f"**買いの発動条件**：{ai['買い逆指値発動価格表示']}")
        st.write(f"**発動後の買い指値**：{ai['発動後買い指値表示']}")
        st.caption(f"根拠：{ai['発動後買い指値の根拠']}")
        st.write(f"**損切りの発動条件**：{ai['損切り逆指値発動価格表示']}")
        st.write(f"**発動後の売り指値**：{ai['発動後売り指値表示']}")
        st.caption(f"根拠：{ai['発動後売り指値の根拠']}")
        st.write(f"**注文価格の根拠**：{ai['注文価格の根拠']}")
        st.write(f"**損切り価格の根拠**：{ai['損切り価格の根拠']}")
        st.write(f"**利確価格の根拠**：{ai['利確価格の根拠']}")
    st.caption("売り逆指値・利確参考はルールベースの管理目安です。現在の保有状況や許容損失に応じて調整してください。")

    with st.expander("📊 テクニカル詳細"):
        detail=pd.DataFrame([{
            "現在値":r["株価"],
            "25日線":r["25日線"],
            "75日線":r["75日線"],
            "75日線傾き%":r["75日線_比較期間前比%"],
            "75日線乖離%":r["75日線_乖離率%"],
            "5日騰落率%":r["5日騰落率%"],
            "20日騰落率%":r["20日騰落率%"],
            "60日騰落率%":r["60日騰落率%"],
            "出来高倍率":r["出来高_20日平均比"],
            "ATR14%":r["ATR14%"],
        }])
        st.dataframe(detail,use_container_width=True,hide_index=True)
        st.link_button("Yahoo!チャートを開く",r["Yahoo!チャート"])

    return {"metrics":r,"ai":ai,"holding":h,"earnings":earnings}




# ------------------------------------------------------------
# v19.4.2 表示列の安全化
# ------------------------------------------------------------
def safe_columns(df, columns):
    """表示列が不足していてもKeyErrorでアプリ全体を停止させない。"""
    return df.reindex(columns=columns)


# ------------------------------------------------------------
# v17.6 実戦ランキングのカラム説明
# ------------------------------------------------------------
def practical_ranking_column_config():
    """実戦ランキングの全カラムにヘッダー説明を付ける。"""
    return {
        "順位": st.column_config.NumberColumn(
            "順位",
            help="短期総合スコア・買いやすさなどをもとに並べた現在のランキング順位です。順位そのものが上昇確率を意味するわけではありません。",
            format="%d",
        ),
        "実戦優先度": st.column_config.TextColumn(
            "実戦優先度",
            help="現在のセットアップと買いやすさから、実際に注文候補として扱いやすいかを整理した表示です。🟢注文候補 / 🟡条件待ち / 🟠押し待ち / 🔴見送り・監視。"
        ),
        "銘柄": st.column_config.TextColumn(
            "銘柄",
            help="東証の銘柄コードと銘柄名です。"
        ),
        "Yahoo!チャート": st.column_config.LinkColumn(
            "チャート",
            help="Yahoo!ファイナンスの該当銘柄チャートを開きます。アプリの数値だけで決めず、実際の値動き・出来高・高値安値も確認するためのリンクです。",
            display_text="Yahoo! ↗"
        ),
        "株価": st.column_config.NumberColumn(
            "現在値",
            help="取得できた最新の日足終値です。リアルタイム株価を保証する値ではありません。",
            format="%.0f円"
        ),
        "前日比": st.column_config.NumberColumn(
            "前日比",
            help="最新の日足終値と、その1営業日前の終値との差額です。プラスなら前日終値より上、マイナスなら下です。",
            format="%+.0f円"
        ),
        "前日比%": st.column_config.NumberColumn(
            "前日比%",
            help="前日終値に対する現在値の騰落率です。例：+2.00%なら前日終値から約2%上昇しています。短期の勢いを見る基本情報です。",
            format="%+.2f%%"
        ),
        "短期総合スコア": st.column_config.ProgressColumn(
            "短期スコア",
            help="独自ルールで『上昇力』と『今の買いやすさ』を統合した0〜100点の評価です。上昇確率や勝率ではありません。高いほど、短期上昇候補としての条件が多く揃っています。",
            min_value=0,max_value=100,format="%.1f"
        ),
        "セットアップ": st.column_config.TextColumn(
            "セットアップ",
            help="現在のチャートがどの形に近いかを表します。例：🔥ブレイク準備中、🚀ブレイク直後、🎯75日線押し目、📈モメンタム継続など。注文方法を決める重要な分類です。"
        ),
        "ルール評価": st.column_config.TextColumn(
            "ルール評価",
            help="独自短期ルールの総合評価です。S/A/B/C等はアプリ内の条件に基づく段階評価で、将来の勝率や上昇確率を表す格付けではありません。"
        ),
        "注文種類": st.column_config.TextColumn(
            "注文種類",
            help="現在のセットアップに対して楽天証券で想定する買い注文の種類です。『買い逆指値』は上抜け確認後に買う、『買い指値』は押し目まで待って買う、『注文しない』は条件未達を意味します。"
        ),
        "注文価格表示": st.column_config.TextColumn(
            "注文価格",
            help="買い注文を出す場合の参考価格です。買い逆指値なら『○円以上になったら』の条件価格、買い指値なら『○円付近まで下がったら』の待ち価格です。"
        ),
        "買い逆指値発動価格表示": st.column_config.TextColumn(
            "買い｜発動条件",
            help="買い逆指値で『株価が何円以上になったら注文を発動するか』です。通常の買い指値では『なし』と表示します。"
        ),
        "発動後買い指値表示": st.column_config.TextColumn(
            "買い｜発動後の指値",
            help="逆指値が発動したあと、実際に出す買い指値です。固定ティックではなく、ATRの15%を基本に、発動価格の0.30%を上限、最低2ティックで自動計算します。この価格を超えて飛んだ場合は約定しない可能性があります。"
        ),
        "損切り逆指値発動価格表示": st.column_config.TextColumn(
            "損切り｜発動条件",
            help="保有後、株価が何円以下になったら損切り注文を発動するかです。"
        ),
        "発動後売り指値表示": st.column_config.TextColumn(
            "損切り｜発動後の売り指値",
            help="損切り逆指値が発動したあと、実際に出す売り指値です。固定ティックではなく、ATRの15%を基本に、発動価格の0.30%を上限、最低2ティックで自動計算します。価格がこの指値より下へ飛んだ場合、約定しない可能性があります。"
        ),
        "売買シナリオ": st.column_config.TextColumn(
            "売買シナリオ",
            help="『何を確認して買い、何が起きたら失敗と判断するか』を専門用語をなるべく使わずに要約します。",
            width="large"
        ),
        "注文価格の根拠": st.column_config.TextColumn(
            "注文価格の根拠",
            help="なぜその注文価格になったかを、ブレイク水準・ATR・75日線などの具体的な数値とセットアップの考え方から説明します。",
            width="large"
        ),
        "損切り価格の根拠": st.column_config.TextColumn(
            "損切り価格の根拠",
            help="なぜその損切り価格になったかを、ATR・75日線・ブレイク水準などから説明します。",
            width="large"
        ),
        "利確価格の根拠": st.column_config.TextColumn(
            "利確価格の根拠",
            help="利確①・②の計算根拠です。原則として買値と損切りの差を1Rとし、2R・3Rを参考値として表示します。",
            width="large"
        ),
        "損切り価格表示": st.column_config.TextColumn(
            "損切り",
            help="買い約定後に設定する売り逆指値の参考価格です。ATRや75日線・ブレイク水準などから機械的に計算した目安で、絶対的な正解価格ではありません。"
        ),
        "利確目安①表示": st.column_config.TextColumn(
            "利確①",
            help="買値と損切り価格の差を1R（1単位のリスク）として、原則2R上を第一利確の参考値として表示します。将来到達する価格を予測しているわけではありません。"
        ),
        "RR": st.column_config.NumberColumn(
            "RR",
            help="リスクリワード比です。『想定利益幅 ÷ 想定損失幅』。例：2.00なら、損失幅1に対して利益幅2を狙う設計です。高ければ必ず良いわけではなく、到達可能性も合わせて判断します。",
            format="%.2f"
        ),
        "決算警告": st.column_config.TextColumn(
            "決算警告",
            help="取得できた次回決算予定日が近い場合の注意表示です。決算直前は値動きが急変しやすいため、短期注文の前に確認します。空欄は『決算が遠い』ではなく、予定日未取得の場合もあります。"
        ),
    }

def practical_ranking_explainer():
    with st.expander("❓ 実戦ランキングの各カラムの見方"):
        st.markdown("""
- **順位**：現在の独自短期ランキング順位。上昇確率ではありません。
- **実戦優先度**：今すぐ注文候補か、条件待ち・押し待ち・見送りかを整理した表示。
- **銘柄 / チャート**：銘柄コード・名称とYahoo!チャートへのリンク。
- **現在値**：取得できた最新の日足終値。リアルタイム保証ではありません。
- **前日比 / 前日比%**：1営業日前の終値から、今日どれだけ動いたか。
- **短期スコア**：トレンド・出来高・モメンタム・業績・買い位置・過熱リスクなどを統合した独自スコア。**勝率ではありません**。
- **セットアップ**：ブレイク準備、ブレイク直後、75日線押し目など、現在のチャートの型。
- **セットアップ判定根拠**：詳細画面で、その銘柄がなぜその型になったかを具体的な数値で表示。
  - **🔥 ブレイク準備中**：直近高値の3%以内（まだ未突破）＋75日線上向き＋出来高倍率1.1倍以上。
  - **🚀 ブレイク直後**：直近高値を上抜け後3%以内＋75日線上向き＋出来高倍率1.3倍以上。
  - **🎯 75日線押し目**：75日線上向き＋株価が75日線の-3%〜0%。
  - **📈 モメンタム継続**：60日騰落率+15%以上など、中期上昇を維持しつつ短期過熱が極端でない状態。
- **ルール評価**：S/A/B/C等の独自段階評価。統計的な格付けではありません。
- **注文種類**：楽天証券での参考注文。**買い逆指値＝上抜け確認、買い指値＝押し待ち**。
- **買い｜発動条件**：買い逆指値で「何円以上になったら注文を発動するか」。
- **買い｜発動後の指値**：発動後に実際に出す買い指値。許容幅は自動計算（ATRの15%を基本、発動価格の0.30%を上限、最低2ティック）。
- **損切り｜発動条件**：保有後「何円以下になったら損切り注文を発動するか」。
- **損切り｜発動後の売り指値**：発動後に実際に出す売り指値。許容幅は同じルールで自動計算。
- **注文価格**：従来の要約表示。v19.4では上記4列を実注文向けの主表示にします。
- **注文価格の根拠**：ブレイク準備は「ブレイク水準＋1ティック」の買い逆指値。ブレイク直後は高値を追わず「現在値から0.75ATR程度の押し」を買い指値。75日線押し目は「前日高値＋指定ティック」を上抜けて反転確認する買い逆指値。
- **損切り**：約定後の売り逆指値の参考値。
- **損切り価格の根拠**：まず「買った理由が崩れる水準」を決めます。75日線押し目なら75日線の明確割れ、ブレイク系ならブレイク水準の支持失敗。ATRはブレイク系で日々のノイズを避ける補助バッファとしてのみ使用します。
- **利確①**：基本的に損切り幅の2倍（2R）を狙う第一利確参考値。
- **利確価格の根拠**：買値と損切りの差を1Rとして、利確①=2R、利確②=3Rとするリスク管理上の考え方。

#### 📝 用語をかんたんに
- **ブレイク**：これまで超えられなかった高値を上に抜けること。
- **ブレイク水準**：その「これまで超えられなかった高値」。
- **1ティック**：株価が動く最小単位。ここでは「高値に到達しただけ」でなく「高値を超えた」と確認するために使います。
- **買い逆指値**：株価が指定価格**以上**になったら買い注文を発動する方法。上昇を確認してから買う用途。
- **買い指値**：指定価格**以下**まで下がったら買う方法。安くなるのを待つ用途。
- **支持線**：株価が下がってきたときに、下げ止まりを期待する価格帯。
- **ATR**：その銘柄が普段1日にどの程度動くかを見る値動き幅の目安。独自短期では主に「少しの値動きで損切りされないための余裕」の計算に使います。
- **0.5ATR / 0.75ATR**：ATRの50% / 75%という意味です。
- **1R**：買値から損切り価格までの損失幅。2Rならその2倍の値幅です。
- **RR**：利益幅 ÷ 損失幅。2.00ならリスク1に対して利益2を狙う設計。
- **決算警告**：決算予定が近い場合の注意。空欄でも予定日を取得できていない場合があります。

**見る順番のおすすめ**：  
`実戦優先度 → 前日比 → 短期スコア → セットアップ → 注文種類 → 注文価格 → 損切り → RR → チャート確認`
""")


# ------------------------------------------------------------
# v17.4 ネットワーク取得の並列化
# ------------------------------------------------------------
def parallel_fetch(tickers, func, max_workers=6):
    """
    yfinanceの銘柄別取得を少数スレッドで並列化。
    過剰アクセスを避けるため既定6並列。失敗銘柄は空dictで返す。
    """
    tickers = list(tickers)
    results = {}
    if not tickers:
        return results

    workers = max(1, min(max_workers, len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(func, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                results[t] = fut.result() or {}
            except Exception:
                results[t] = {}
    return results


# ------------------------------------------------------------
# v17.3 モード別セッション保持
# ------------------------------------------------------------
def mode_cache_key(mode_name, selected_markets):
    """モード＋市場の組み合わせごとに結果を分離して保持する。"""
    markets = tuple(sorted(selected_markets or []))
    return f"{mode_name}::{markets}"

def get_mode_cache(mode_name, selected_markets):
    key = mode_cache_key(mode_name, selected_markets)
    return st.session_state.get("scan_cache_v1963", {}).get(key)

def set_mode_cache(mode_name, selected_markets, payload):
    if "scan_cache_v1963" not in st.session_state:
        st.session_state.scan_cache_v1963 = {}
    key = mode_cache_key(mode_name, selected_markets)
    st.session_state.scan_cache_v1963[key] = payload

def cache_age_text(ts):
    if ts is None:
        return ""
    try:
        now = pd.Timestamp.now()
        t = pd.Timestamp(ts)
        sec = max(0, int((now - t).total_seconds()))
        if sec < 60:
            return f"{sec}秒前"
        mins = sec // 60
        if mins < 60:
            return f"{mins}分前"
        return f"{mins//60}時間{mins%60}分前"
    except Exception:
        return ""

def cached_result_banner(payload):
    if not payload:
        return
    ts = payload.get("timestamp")
    age = cache_age_text(ts)
    label = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts is not None else "不明"
    if ts is not None:
        mins = max(0, int((pd.Timestamp.now() - pd.Timestamp(ts)).total_seconds() // 60))
    else:
        mins = 999
    if mins >= 10:
        st.warning(f"前回取得：{label}（{age}）。10分以上経過しているため、必要なら再スキャンしてください。")
    else:
        st.info(f"前回取得：{label}（{age}）。モード切替後もこの結果を保持しています。")


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_regime():
    """TOPIXと日経平均の移動平均線・直近騰落から、市場トレンドを簡易判定する。"""
    out = []
    for ticker, name in [("^TOPX", "TOPIX"), ("^N225", "日経平均")]:
        try:
            d = yf.download(ticker, period="6mo", auto_adjust=True, progress=False, threads=False)
            if d is None or d.empty or len(d) < 80:
                continue
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            ma20 = close.rolling(20).mean()
            ma75 = close.rolling(75).mean()
            now = float(close.iloc[-1])
            m20 = float(ma20.iloc[-1])
            m75 = float(ma75.iloc[-1])
            slope75 = (m75 / float(ma75.iloc[-11]) - 1) * 100 if len(ma75.dropna()) >= 11 else 0
            score = 0
            score += 35 if now > m20 else 0
            score += 30 if m20 > m75 else 0
            score += 25 if slope75 > 0 else 0
            score += 10 if float(close.iloc[-1] / close.iloc[-6] - 1) > 0 else 0
            out.append({"指数":name, "現在値":now, "20日線":m20, "75日線":m75, "75日線傾き%":slope75, "地合い点":score})
        except Exception:
            pass

    if not out:
        return {"label":"⚪ 地合い不明", "score":50, "comment":"指数データを取得できませんでした。", "details":[]}

    score = sum(x["地合い点"] for x in out) / len(out)
    if score >= 80:
        label, comment = "🟢 強気", "日経平均・TOPIXとも上昇トレンドが強い状態です。"
    elif score >= 60:
        label, comment = "🟡 やや強気", "主要指数は概ね上向きですが、一部条件は未達です。"
    elif score >= 40:
        label, comment = "🟠 中立", "主要指数のトレンドが混在しています。"
    else:
        label, comment = "🔴 弱気", "日経平均・TOPIXのトレンド条件が弱い状態です。"
    return {"label":label, "score":score, "comment":comment, "details":out}

@st.cache_data(ttl=21600, show_spinner=False)
def get_earnings_date_info(ticker):
    """yfinanceで取得できる場合だけ次回決算日を返す。取得不能時は無理に推定しない。"""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = []
        if isinstance(cal, dict):
            v = cal.get("Earnings Date") or cal.get("EarningsDate")
            if isinstance(v, (list, tuple)):
                dates = list(v)
            elif v is not None:
                dates = [v]
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            for key in ["Earnings Date", "EarningsDate"]:
                if key in cal.index:
                    vals = cal.loc[key]
                    dates = vals.tolist() if hasattr(vals, "tolist") else [vals]
                    break
        parsed = [pd.Timestamp(x).tz_localize(None) if pd.Timestamp(x).tzinfo else pd.Timestamp(x) for x in dates if pd.notna(x)]
        today = pd.Timestamp.now().normalize()
        future = sorted([x.normalize() for x in parsed if x.normalize() >= today])
        if not future:
            return {"date":None, "days":None, "warning":""}
        dt = future[0]
        days = int((dt - today).days)
        warn = "🔴 決算直前" if days <= 3 else ("🟠 決算1週間以内" if days <= 7 else "")
        return {"date":dt.strftime("%Y-%m-%d"), "days":days, "warning":warn}
    except Exception:
        return {"date":None, "days":None, "warning":""}

def backtest_current_ai_logic(d, slope_days=20, breakout_days=60, horizon=5, target_pct=5.0, stop_pct=3.0):
    """
    過去時点で得られた株価・出来高だけを使う簡易ウォークフォワード検証。
    独自短期の技術面に近い条件で候補日を抽出し、その後horizon営業日の+target/-stop到達を集計。
    財務(C)は将来情報混入を避けるため、この簡易版バックテストでは使わない。
    """
    if d is None or len(d) < max(120, breakout_days + 80):
        return None
    x = d.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c in x.columns and isinstance(x[c], pd.DataFrame):
            x[c] = x[c].iloc[:,0]
    x = x.dropna(subset=["Close","High","Low","Volume"]).copy()
    x["MA20"] = x.Close.rolling(20).mean()
    x["MA75"] = x.Close.rolling(75).mean()
    x["V20"] = x.Volume.rolling(20).mean()
    x["R5"] = x.Close.pct_change(5)*100
    x["R20"] = x.Close.pct_change(20)*100
    x["R60"] = x.Close.pct_change(60)*100
    x["Slope75"] = (x.MA75/x.MA75.shift(slope_days)-1)*100
    x["Dev75"] = (x.Close/x.MA75-1)*100
    x["VR"] = x.Volume/x.V20
    x["PrevHigh"] = x.High.shift(1).rolling(breakout_days).max()
    x["Ext"] = (x.Close/x.PrevHigh-1)*100

    signals = (
        (x.Slope75 > 0) &
        (x.R60 > 5) &
        (x.VR >= 1.0) &
        (x.Dev75 > -5) & (x.Dev75 < 20) &
        (
            ((x.Ext >= -3) & (x.Ext <= 3)) |
            ((x.Dev75 >= -3) & (x.Dev75 <= 8))
        )
    )
    idxs = list(x.index[signals])
    # Avoid counting clusters of nearly identical consecutive signals.
    chosen = []
    last_pos = -99
    posmap = {idx:i for i,idx in enumerate(x.index)}
    for idx in idxs:
        p = posmap[idx]
        if p - last_pos >= horizon:
            chosen.append(idx)
            last_pos = p

    rows = []
    for idx in chosen:
        p = posmap[idx]
        if p + horizon >= len(x):
            continue
        entry = float(x.Close.iloc[p])
        fut = x.iloc[p+1:p+1+horizon]
        max_up = (float(fut.High.max())/entry-1)*100
        max_down = (float(fut.Low.min())/entry-1)*100
        ret = (float(fut.Close.iloc[-1])/entry-1)*100
        target_hit = max_up >= target_pct
        stop_hit = max_down <= -stop_pct
        rows.append((ret,max_up,max_down,target_hit,stop_hit))

    if not rows:
        return None
    bt = pd.DataFrame(rows, columns=["ret","max_up","max_down","target_hit","stop_hit"])
    return {
        "件数":len(bt),
        f"{horizon}日平均リターン%":float(bt.ret.mean()),
        f"{horizon}日中央値%":float(bt.ret.median()),
        f"+{target_pct:.0f}%到達率":float(bt.target_hit.mean()*100),
        f"-{stop_pct:.0f}%到達率":float(bt.stop_hit.mean()*100),
        "平均最大上昇%":float(bt.max_up.mean()),
        "平均最大下落%":float(bt.max_down.mean()),
    }

# ------------------------------------------------------------
# v18.0.1 独自短期スコア関数（UIより先に定義）
# ------------------------------------------------------------
def ai_pre_score(r):
    slope=float(r["75日線_比較期間前比%"])
    r20=float(r["20日騰落率%"])
    r60=float(r["60日騰落率%"])
    vr=float(r["出来高_20日平均比"])
    dev=float(r["75日線_乖離率%"])
    ext=float(r["Bブレイク上昇率%"]) if np.isfinite(r["Bブレイク上昇率%"]) else -99

    trend = np.clip(45 + slope*10 + (10 if r["株価25日線比%"]>0 else -10),0,100)
    volume = np.clip(35 + (vr-1)*45,0,100)
    momentum = np.clip(40 + r20*2 + r60*.6,0,100)
    entry = 35
    if -3 <= dev < 0 and slope>0: entry=90
    elif -3 <= ext <= 0 and vr>=1.1 and slope>0: entry=92
    elif 0 < ext <= 3 and vr>=1.3: entry=85
    elif 0 <= dev <= 10 and slope>0: entry=65
    risk = 90
    if dev>10: risk-=15
    if dev>20: risk-=25
    if dev>30: risk-=25
    if ext>5: risk-=15
    if ext>10: risk-=25
    if np.isfinite(r["ATR14%"]) and r["ATR14%"]>5: risk-=15
    risk=np.clip(risk,0,100)
    return .30*trend+.25*volume+.20*momentum+.15*entry+.10*risk

def ai_scores(r):
    slope=float(r["75日線_比較期間前比%"])
    r5=float(r["5日騰落率%"])
    r20=float(r["20日騰落率%"])
    r60=float(r["60日騰落率%"])
    vr=float(r["出来高_20日平均比"])
    dev=float(r["75日線_乖離率%"])
    ext=float(r["Bブレイク上昇率%"]) if np.isfinite(r["Bブレイク上昇率%"]) else np.nan
    atrp=float(r["ATR14%"]) if np.isfinite(r["ATR14%"]) else np.nan
    trading=float(r["売買代金_億円"]) if np.isfinite(r["売買代金_億円"]) else 0
    cscore=float(r.get("Cスコア",0) or 0)

    trend=np.clip(40+slope*10+(15 if r["株価25日線比%"]>0 else -10)+(10 if r60>10 else 0),0,100)
    volume=np.clip(30+(vr-1)*40+min(20,np.log10(max(trading,0.1))*8),0,100)
    momentum=np.clip(40+r20*2+r60*.7+r5*1.5,0,100)
    earnings=np.clip(cscore,0,100)

    # Entry quality
    entry=30
    setup="監視"
    if slope>0 and -3<=dev<0:
        entry=92; setup="🎯 押し目・75日線接近"
    if np.isfinite(ext) and -3<=ext<=0 and slope>0 and vr>=1.1:
        entry=max(entry,95); setup="🔥 ブレイク準備中"
    elif np.isfinite(ext) and 0<ext<=3 and vr>=1.3 and slope>0:
        entry=max(entry,88); setup="🚀 ブレイク直後"
    elif r60>=15 and -3<=r5<=5 and 0<=dev<=12 and slope>0:
        entry=max(entry,72); setup="📈 モメンタム継続"
    if bool(r.get("C",False)) and r20>0:
        entry=max(entry,75)
        if setup=="監視": setup="💹 決算加速"

    signals=[]
    if slope>0 and -3<=dev<0:
        signals.append("🎯 押し目")
    if np.isfinite(ext) and -3<=ext<=0 and slope>0 and vr>=1.1:
        signals.append("🔥 ブレイク準備中")
    if np.isfinite(ext) and 0<ext<=3 and vr>=1.3 and slope>0:
        signals.append("🚀 ブレイク直後")
    if bool(r.get("C",False)) and r20>0:
        signals.append("💹 決算加速")
    if r60>=15 and -3<=r5<=5 and 0<=dev<=12 and slope>0:
        signals.append("📈 モメンタム継続")
    if not signals:
        signals=["監視"]

    # v18.0.2 セットアップ判定根拠
    setup_reasons=[]
    if setup=="🔥 ブレイク準備中":
        setup_reasons=[
            f"直近高値まで {abs(ext):.2f}%" if np.isfinite(ext) else "直近高値へ接近",
            f"75日線は上向き（比較期間前比 {slope:+.2f}%）",
            f"出来高は20日平均の {vr:.2f}倍",
            "高値をまだ明確に上抜けていないため『ブレイク直後』ではなく『準備中』",
        ]
    elif setup=="🚀 ブレイク直後":
        setup_reasons=[
            f"直近高値を {ext:+.2f}% 上抜け" if np.isfinite(ext) else "直近高値を上抜け",
            f"75日線は上向き（比較期間前比 {slope:+.2f}%）",
            f"出来高は20日平均の {vr:.2f}倍",
            "高値突破後3%以内のため『ブレイク直後』",
        ]
    elif setup=="🎯 押し目・75日線接近":
        setup_reasons=[
            f"株価は75日線の {dev:+.2f}% 位置",
            f"75日線は上向き（比較期間前比 {slope:+.2f}%）",
            "75日線のすぐ下まで押しているため押し目候補",
        ]
    elif setup=="📈 モメンタム継続":
        setup_reasons=[
            f"60日騰落率 {r60:+.2f}%",
            f"5日騰落率 {r5:+.2f}%",
            f"75日線乖離 {dev:+.2f}%",
            "中期上昇トレンドを維持しつつ、短期の過熱が極端ではない",
        ]
    elif setup=="💹 決算加速":
        setup_reasons=[
            "決算モメンタムCに該当",
            f"20日騰落率 {r20:+.2f}%",
            "決算面の加速を確認し、株価も20日ベースで上向き",
        ]
    else:
        setup_reasons=[
            "押し目・ブレイク・決算加速・モメンタム継続の主要条件が未達",
            "現在は監視対象として扱う",
        ]
    setup_reason_text=" / ".join(setup_reasons)

    # Risk / overheat; higher score = safer
    risk=95
    if dev>10: risk-=15
    if dev>20: risk-=25
    if dev>30: risk-=30
    if np.isfinite(ext) and ext>3: risk-=8
    if np.isfinite(ext) and ext>7: risk-=20
    if np.isfinite(ext) and ext>10: risk-=25
    if np.isfinite(atrp) and atrp>4: risk-=10
    if np.isfinite(atrp) and atrp>6: risk-=20
    risk=np.clip(risk,0,100)

    strength=.30*trend+.25*volume+.25*momentum+.20*earnings
    ease=.55*entry+.45*risk
    total=.60*strength+.40*ease

    # Trigger / stop references
    if setup.startswith("🎯"):
        trigger=float(r["A買い価格"])
        stop=float(r["A初期損切り"])
    elif np.isfinite(r["Bブレイク水準"]):
        trigger=float(r["Bブレイク水準"])+tick_size(float(r["Bブレイク水準"]))
        if np.isfinite(r["B初期損切り"]):
            stop=float(r["B初期損切り"])
        else:
            stop=trigger-(float(r["ATR14%"])/100*float(r["株価"]) if np.isfinite(r["ATR14%"]) else trigger*.03)
    else:
        trigger=float(r["株価"])+tick_size(float(r["株価"]))
        stop=trigger-(float(r["ATR14%"])/100*float(r["株価"]) if np.isfinite(r["ATR14%"]) else trigger*.03)

    if total>=80 and ease>=75:
        grade="🟢 S｜最有力"
    elif total>=70 and ease>=65:
        grade="🟢 A｜良好"
    elif total>=60:
        grade="🟡 B｜候補"
    elif strength>=70 and ease<55:
        grade="🟠 C｜強いが今は待ち"
    elif total>=50:
        grade="🟡 C｜慎重"
    else:
        grade="⚪ 見送り"

    # 楽天証券向け注文ナビ v14
    # 「買い逆指値」と「買い指値」と「約定後の売り逆指値」を明確に分離する。
    current=float(r["株価"])
    atr_yen=(float(r["ATR14%"])/100*current) if np.isfinite(r["ATR14%"]) else current*.03
    chase = (dev > 20) or (np.isfinite(ext) and ext > 7) or ease < 45

    buy_order_type="注文しない"
    buy_price=np.nan
    buy_price_text="—"
    buy_condition="監視"
    stop_order_type="—"
    stop_price=np.nan
    order_reason=""
    buy_price_reason=""
    stop_price_reason=""
    take_profit_reason=""

    # v19.4 逆指値注文を「発動条件」と「発動後の実際の指値」に分解
    buy_trigger_price=np.nan
    buy_limit_price=np.nan
    stop_trigger_price=np.nan
    stop_limit_price=np.nan
    buy_trigger_text="—"
    buy_limit_text="—"
    stop_trigger_text="—"
    stop_limit_text="—"
    buy_limit_buffer=np.nan
    stop_limit_buffer=np.nan
    buy_limit_reason="—"
    stop_limit_reason="—"

    if chase:
        order_reason="過熱または買いやすさ不足。現在値を追いかけず、押し目形成後に再判定。"
        buy_price_reason="注文を出さないため買い価格は設定しません。75日線乖離・ブレイク後上昇率・買いやすさのいずれかが過熱側です。"
        stop_price_reason="未約定のため損切り価格は設定しません。"

    elif setup.startswith("🔥"):
        # ブレイク前：上抜けを確認してから買うため「買い逆指値」
        buy_order_type="買い逆指値"
        buy_price=float(trigger)
        buy_trigger_price=buy_price
        buy_limit_buffer=adaptive_limit_buffer(buy_trigger_price, atr_yen)
        buy_limit_price=buy_trigger_price + buy_limit_buffer
        buy_trigger_text=f"{buy_trigger_price:.0f}円以上"
        buy_limit_text=f"{buy_limit_price:.0f}円"
        buy_price_text=f"発動 {buy_trigger_text} → 指値 {buy_limit_text}"
        buy_condition=f"株価が{buy_trigger_price:.0f}円以上になったら、{buy_limit_price:.0f}円の買い指値を発注"
        stop_order_type="売り逆指値"
        stop_price=float(stop)
        order_reason="まだブレイク前。上抜けを確認してから入る。"
        buy_price_reason=f"直近高値（過去{breakout_days}営業日で超えられなかった高値）{float(r['Bブレイク水準']):.0f}円より、最小の値幅1つ分だけ上の {buy_price:.0f}円を条件価格にしています。つまり『{buy_price:.0f}円以上になったら買いを検討』です。安く買うための価格ではなく、直近高値を実際に突破して上昇の強さを確認するための価格です。注文種類は買い逆指値（株価が指定価格以上になったら買い注文を発動）です。発動後の指値は固定ティックではなく自動計算し、{buy_trigger_price:.0f}円＋許容幅{buy_limit_buffer:.0f}円＝{buy_limit_price:.0f}円です。許容幅は現行方式としてATRの15%を基本に、発動価格の0.30%を上限、最低2ティックとしています。これは最適値ではなく、v19.5の過去検証で比較対象にしています。"
        blevel=float(r["Bブレイク水準"])
        atr_abs=float(r["ATR14"]) if np.isfinite(r["ATR14"]) else np.nan
        buffer_abs=(blevel-stop_price)
        stop_price_reason=f"買った理由は『直近高値 {blevel:.0f}円を突破したこと』です。突破後は、この {blevel:.0f}円付近が下値を支える価格（支持線）になってほしいと考えます。損切り参考は {blevel:.0f}円 － 値動きの余裕幅 {buffer_abs:.0f}円 ＝ {stop_price:.0f}円です。余裕幅は原則0.5ATR（ATR＝その銘柄が普段1日にどれくらい動くかの目安）の半分です。ATR自体を理由に損切るのではなく、一瞬の小さな値下がりで損切りされにくくするためだけに使います。"

    elif setup.startswith("🚀"):
        # ブレイク済み：すでに上抜けているため、さらに上で買う逆指値は使わず押し待ちの指値。
        breakout=float(r["Bブレイク水準"]) if np.isfinite(r["Bブレイク水準"]) else current
        pullback_low=max(breakout, current-0.75*atr_yen)
        pullback_high=current-tick_size(current)
        if pullback_low > pullback_high:
            pullback_low=pullback_high
        buy_order_type="買い指値"
        buy_price=float(pullback_high)
        buy_limit_price=buy_price
        buy_trigger_text="なし（通常の買い指値）"
        buy_limit_text=f"{buy_price:.0f}円"
        buy_price_text=f"{pullback_low:.0f}〜{pullback_high:.0f}円"
        buy_condition=f"逆指値の発動条件はなし。{buy_price_text}への押しを待ち、上に飛んだ場合は追いかけない"
        stop_order_type="売り逆指値"
        stop_price=float(stop)
        order_reason="ブレイク済み。新規の買い逆指値ではなく、押しを待つ指値買い。"
        buy_price_reason=f"すでに直近高値 {breakout:.0f}円を突破済みなので、上がった価格を追いかけません。現在値 {current:.0f}円から少し下がるのを待ちます。下げ幅の目安は0.75ATR（ATR＝その銘柄が普段1日にどれくらい動くかの目安の75%）です。ただし、突破した {breakout:.0f}円より下では買わない範囲として {buy_price_text} を買い指値（指定した価格以下で買う注文）の候補にしています。"
        buffer_abs=(breakout-stop_price)
        stop_price_reason=f"突破した直近高値 {breakout:.0f}円付近が、その後は下値を支える価格（支持線）になることを期待しています。損切り参考は {breakout:.0f}円 － 値動きの余裕幅 {buffer_abs:.0f}円 ＝ {stop_price:.0f}円です。余裕幅は原則0.5ATR（普段の1日の値動き幅の目安の半分）。少し割れただけではなく、突破が失敗した可能性が高まったところで撤退する考え方です。"

    elif setup.startswith("🎯"):
        # v19.1 75日線付近まで押した後、「前日高値超え」で反転を確認して入る。
        # 条件価格は現在値より上に置くトリガーなので、買い指値ではなく買い逆指値。
        buy_order_type="買い逆指値"
        buy_price=float(r["A買い価格"])
        buy_trigger_price=buy_price
        buy_limit_buffer=adaptive_limit_buffer(buy_trigger_price, atr_yen)
        buy_limit_price=buy_trigger_price + buy_limit_buffer
        buy_trigger_text=f"{buy_trigger_price:.0f}円以上"
        buy_limit_text=f"{buy_limit_price:.0f}円"
        buy_price_text=f"発動 {buy_trigger_text} → 指値 {buy_limit_text}"
        buy_condition=f"75日線付近の押し目形成後、株価が{buy_trigger_price:.0f}円以上になったら、{buy_limit_price:.0f}円の買い指値を発注"
        stop_order_type="売り逆指値"
        stop_price=float(r["A初期損切り"])
        order_reason="75日線付近まで押しただけでは買わず、前日高値超えで反転を確認してから入る。"
        buy_price_reason=f"75日移動平均線（過去75営業日の平均株価）付近まで下がった後、本当に上向きへ戻り始めたかを確認してから買う考え方です。前日の高値より最小の値幅分だけ上の {buy_price:.0f}円を超えたら買いを検討します。注文種類は買い逆指値（株価が指定価格以上になったら買い注文を発動）です。発動後の指値は{buy_trigger_price:.0f}円＋自動許容幅{buy_limit_buffer:.0f}円＝{buy_limit_price:.0f}円です。許容幅は現行方式としてATRの15%を基本に、発動価格の0.30%を上限、最低2ティックとしています。これは最適値ではなく、v19.5の過去検証で比較対象にしています。安い価格で待つ『買い指値』ではありません。"
        stop_price_reason=f"買った理由は『上向きの75日移動平均線（過去75営業日の平均株価）が下値を支える』と考えたためです。その75日線から設定した余裕幅だけ下の {stop_price:.0f}円を割ったら、買った前提が崩れたと判断する損切り参考です。このセットアップではATRを損切りの主な理由にはしていません。"

    elif setup.startswith("💹"):
        # 決算加速だけでは注文方法を決め打ちしない。
        buy_order_type="条件確認後"
        buy_price=np.nan
        buy_price_text="—"
        buy_condition="決算加速だけで注文せず、押し目またはブレイク条件が追加で出るまで待つ"
        stop_order_type="—"
        stop_price=np.nan
        order_reason="決算の良さだけを理由に注文種類を決めない。"
        buy_price_reason="決算加速だけではエントリー価格の根拠が不足するため、押し目またはブレイク条件が追加で出るまで価格を設定しません。"
        stop_price_reason="未約定のため損切り価格は設定しません。"

    elif setup.startswith("📈"):
        # モメンタム継続は高値追いを避け、明確な新トリガーが出るまで待つ。
        buy_order_type="注文しない"
        buy_price=np.nan
        buy_price_text="—"
        buy_condition="押し目または新しいブレイク準備シグナルを待つ"
        stop_order_type="—"
        stop_price=np.nan
        order_reason="モメンタムだけで高値を追わない。"
        buy_price_reason="モメンタム継続だけでは高値追いになる可能性があるため、押し目または新しいブレイク準備シグナルが出るまで価格を設定しません。"
        stop_price_reason="未約定のため損切り価格は設定しません。"

    else:
        buy_order_type="注文しない"
        buy_condition="監視継続"
        order_reason="注文方法を一意に決められるセットアップではない。"
        buy_price_reason="主要セットアップの条件が不足しているため、買い価格は設定しません。"
        stop_price_reason="未約定のため損切り価格は設定しません。"

    # 約定後の損切り逆指値も、発動価格と発動後の売り指値を分ける。
    if stop_order_type=="売り逆指値" and np.isfinite(stop_price):
        stop_trigger_price=float(stop_price)
        stop_limit_buffer=adaptive_limit_buffer(stop_trigger_price, atr_yen)
        stop_limit_price=stop_trigger_price - stop_limit_buffer
        stop_trigger_text=f"{stop_trigger_price:.0f}円以下"
        stop_limit_text=f"{stop_limit_price:.0f}円"
    else:
        stop_trigger_text="—"
        stop_limit_text="—"

    if np.isfinite(buy_trigger_price) and np.isfinite(buy_limit_price):
        buy_limit_reason=f"発動価格 {buy_trigger_price:.0f}円 ＋ 自動許容幅 {buy_limit_buffer:.0f}円 ＝ {buy_limit_price:.0f}円。許容幅はATR（普段1日の値動き幅）の15%を基本に、発動価格の0.30%を上限、最低2ティックとして呼値単位に丸めます。"
    elif buy_order_type=="買い指値" and np.isfinite(buy_limit_price):
        buy_limit_reason="通常の買い指値なので、逆指値発動後の許容幅は使いません。"
    if np.isfinite(stop_trigger_price) and np.isfinite(stop_limit_price):
        stop_limit_reason=f"損切り発動価格 {stop_trigger_price:.0f}円 － 自動許容幅 {stop_limit_buffer:.0f}円 ＝ {stop_limit_price:.0f}円。許容幅はATR（普段1日の値動き幅）の15%を基本に、発動価格の0.30%を上限、最低2ティックとして呼値単位に丸めます。"

    # リスク計算は、逆指値買いでは発動後に許容する買い指値、
    # 損切り側では発動後の売り指値を使い、予定上の最悪寄りで計算。
    entry_reference = (
        buy_limit_price if buy_order_type=="買い逆指値" and np.isfinite(buy_limit_price)
        else buy_price
    )
    exit_reference = (
        stop_limit_price if stop_order_type=="売り逆指値" and np.isfinite(stop_limit_price)
        else stop_price
    )
    risk_base = entry_reference if np.isfinite(entry_reference) else np.nan
    risk_pct=((exit_reference/risk_base)-1)*100 if np.isfinite(risk_base) and np.isfinite(exit_reference) and risk_base else np.nan
    stop_price_text=f"発動 {stop_trigger_text} → 指値 {stop_limit_text}" if np.isfinite(stop_trigger_price) else "—"
    order_summary=f"{buy_order_type}｜{buy_price_text}｜約定後 {stop_order_type} {stop_price_text}"
    take_profit1=take_profit2=rr=reward_pct=np.nan
    if np.isfinite(entry_reference) and np.isfinite(exit_reference) and entry_reference > exit_reference:
        risk_yen=entry_reference-exit_reference
        take_profit1=entry_reference+2*risk_yen
        take_profit2=entry_reference+3*risk_yen
        rr=2.0
        reward_pct=(take_profit1/entry_reference-1)*100
    take_profit1_text=f"{take_profit1:.0f}円" if np.isfinite(take_profit1) else "—"
    take_profit2_text=f"{take_profit2:.0f}円" if np.isfinite(take_profit2) else "—"
    if np.isfinite(take_profit1) and np.isfinite(entry_reference) and np.isfinite(exit_reference):
        risk_yen=entry_reference-exit_reference
        take_profit_reason=f"実際の注文で許容する買い価格 {entry_reference:.0f}円 と、損切り発動後の売り指値 {exit_reference:.0f}円 の差 {risk_yen:.0f}円を1Rとし、利確①は2R、利確②は3R上に置く参考値です。価格到達を予測するものではありません。"
    else:
        take_profit_reason="買値または損切り価格が未設定のため、利確価格も設定していません。"
    if buy_order_type in ["買い逆指値","買い指値"]:
        practical_priority="🟢 注文候補" if ease>=70 else ("🟡 条件待ち" if ease>=55 else "🟠 押し待ち")
    else:
        practical_priority="🔴 見送り/監視"

    reasons=[]
    if trend>=75: reasons.append("上昇トレンドが強い")
    if volume>=70: reasons.append(f"出来高{vr:.2f}倍で資金流入")
    if momentum>=75: reasons.append("短中期モメンタムが強い")
    if earnings>=60: reasons.append("業績モメンタムも良好")
    if entry>=85: reasons.append(setup.replace("🎯 ","").replace("🔥 ","").replace("🚀 ",""))
    if risk<45: reasons.append("過熱・値動きリスクが大きい")
    comment="／".join(reasons[:4]) if reasons else "決定的な優位性はまだ弱い"

    return pd.Series({
        "短期総合スコア":total,
        "上昇力":strength,
        "今の買いやすさ":ease,
        "トレンド":trend,
        "需給・出来高":volume,
        "モメンタム":momentum,
        "業績":earnings,
        "エントリー":entry,
        "リスク":risk,
        "セットアップ":setup,
        "セットアップ判定根拠":setup_reason_text,
        "セットアップ根拠一覧":setup_reasons,
        "追加シグナル":"・".join(signals),
        "ルール評価":grade,
        "評価コメント":comment,
        "注文種類":buy_order_type,
        "注文価格":buy_price,
        "注文価格表示":buy_price_text,
        "買い逆指値発動価格":buy_trigger_price,
        "買い逆指値発動価格表示":buy_trigger_text,
        "発動後買い指値":buy_limit_price,
        "発動後買い指値表示":buy_limit_text,
        "発動後買い指値の根拠":buy_limit_reason,
        "注文条件":buy_condition,
        "損切り注文":stop_order_type,
        "損切り価格":stop_price,
        "損切り価格表示":stop_price_text,
        "損切り逆指値発動価格":stop_trigger_price,
        "損切り逆指値発動価格表示":stop_trigger_text,
        "発動後売り指値":stop_limit_price,
        "発動後売り指値表示":stop_limit_text,
        "発動後売り指値の根拠":stop_limit_reason,
        "想定初期リスク%":risk_pct,
        "注文理由":order_reason,
        "売買シナリオ":(
            "75日線が支持線→前日高値超えで反転確認→75日線の明確割れで撤退" if setup.startswith("🎯") else
            "直近高値突破→突破水準が支持線化→支持失敗で撤退" if setup.startswith("🔥") else
            "ブレイク後の押しを待つ→突破水準を維持→支持失敗で撤退" if setup.startswith("🚀") else
            "明確なエントリーシナリオ待ち"
        ),
        "注文価格の根拠":buy_price_reason,
        "損切り価格の根拠":stop_price_reason,
        "利確価格の根拠":take_profit_reason,
        "注文サマリー":order_summary,
        "利確目安①":take_profit1,"利確目安②":take_profit2,
        "利確目安①表示":take_profit1_text,"利確目安②表示":take_profit2_text,
        "想定利益%":reward_pct,"RR":rr,"実戦優先度":practical_priority,
    })


st.title("🎯 短期上昇株ハンター v19.6.3")
st.write("同じURLで、短期・長期ランキングに加えて **🔎 保有銘柄の個別分析と管理** まで行えます。")

mode = st.radio(
    "分析モード",
    ["📘 本ベース A/B/C/D", "🧪 独自短期・独自統合スクリーナー", "🏦 長期・年初来安値", "📕 ちょる子式｜大型株逆張り", "🔎 保有銘柄・個別分析"],
    horizontal=True,
    help="ランキング3モードに加えて、買った銘柄を保有者目線で個別分析・管理できます。"
)

with st.expander("ℹ️ 5つのモードの違い"):
    st.markdown("""
**📘 本ベース A/B/C/D**  
これまで育ててきたロジックです。Aは参考書の「75日線が上向き・株価は75日線より下・75日線上抜けで買う」を中心にしています。B/C/Dは補助戦略です。

**🧪 独自短期・独自統合スクリーナー**  
本のルールには縛られず、短期上昇候補を6要素で採点します。

- トレンド 25%
- 需給・出来高 20%
- モメンタム 15%
- 業績 15%
- エントリー位置 15%
- リスク・過熱 10%

さらに **「上昇力」** と **「今の買いやすさ」** を別々に表示します。  
外部AI APIは使用しないため追加料金はかかりません。

**🏦 長期・年初来安値**  
年初来安値に近い銘柄を探しつつ、**安いだけでは買わない**長期モードです。  
時価総額・ROE・利益率・成長率・配当・PER/PBRなど、取得できるファンダメンタルを合わせて「長期候補 / 分割買い候補 / 安い理由を要確認」を判定します。  
短期売買とは混ぜず、長期の分割購入を前提に表示します。

**📕 ちょる子式｜大型株逆張り**  
読書メモと確認できた書評を基礎にした大型株逆張りモードです。  
前日比-2.5%以上、25日線割れ、BB-3σ、RCI-90以下を個別に表示します。  
「材料なし」は株価だけで断定しないため、未確認/確認済/悪材料ありを分けます。反転確認は🤖アプリ独自補助です。

**🔎 保有銘柄・個別分析**  
銘柄コード・取得単価・保有株数を入れて、**新規で買う判断とは別に、保有者として継続・注意・一部利確・トレンド崩れを判定**します。  
保有銘柄を登録して一覧管理することもできます。
""")

with st.sidebar:
    st.header("設定")

    # まず内部既定値を定義。表示しないモードでも技術計算関数が安全に動くようにする。
    period = "1y"
    slope_days = 20
    max_dev = 3.0
    buy_ticks = 2
    a_stop_buffer_pct = 0.35
    breakout_days = 60
    c_check_count = 100
    bt_top_n = 10
    bt_horizon = 5
    bt_target = 5.0
    bt_stop = 3.0
    long_fund_n = 80
    long_mcap_filter = "5,000億円以上"
    long_max_low_dist = 10
    choruko_mcap = "5,000億円以上"
    choruko_material_default = "未確認"

    if mode.startswith("📘"):
        st.subheader("📘 本ベース設定")
        pl = st.selectbox(
            "株価をさかのぼる期間", ["6か月","1年","2年"], index=1,
            help="何を変える？：スキャンに使う過去の株価データ量です。長くすると過去の値動きを多く参照できますが、取得データ量が増えて処理が重くなる場合があります。"
        )
        period = {"6か月":"6mo","1年":"1y","2年":"2y"}[pl]

        slope_days = st.slider(
            "75日線が上向きかを判断する期間", 5, 40, 20, 1,
            help="何を変える？：75日移動平均線が上向きかを、何営業日前の75日線と比較して判定するかです。小さくすると最近の変化に敏感になり、大きくするとより中期的な傾向を重視します。例：20なら現在と20営業日前の75日線を比較します。"
        )

        st.info("Aの75日線乖離条件は **-3%〜0%で固定** です。v16以降、本の考え方に寄せて深い押しはA対象から除外しています。")

        buy_ticks = st.slider(
            "A：反転確認を何ティック上抜けで判定するか", 1, 5, 2, 1,
            help="何を変える？：本ベースAで『押し目から反転した』と判断する買いトリガーの厳しさです。前日高値を何ティック上抜けたら反転確認とするかを指定します。大きくすると慎重になり、小さくすると早めに反応します。"
        )
        a_stop_buffer_pct = st.slider(
            "A：75日線割れの損切りバッファ",0.10,2.00,0.35,0.05,format="%.2f%%",
            help="何を変える？：本ベースAの損切り参考値を75日線から何％下に置くかです。大きくすると値動きへの余裕は増えますが、損切り時の損失幅も広がります。例：0.35%なら75日線の0.35%下を目安にします。"
        )
        breakout_days = st.slider(
            "B：高値更新を見る期間",20,120,60,10,
            help="何を変える？：本ベースBで、過去何営業日の高値をブレイク基準にするかです。小さくすると短期的な高値更新を拾いやすく、大きくするとより大きな節目の突破を重視します。例：60なら過去60営業日の高値を基準にします。"
        )
        c_check_count = st.slider(
            "C：決算を詳しく確認する上位銘柄数",30,200,100,10,
            help="何を変える？：事前ランキング上位のうち、決算データを追加取得してC判定まで行う銘柄数です。増やすほど広く確認できますが、個別データ取得が増えるためスキャン時間も長くなります。"
        )

    elif mode.startswith("🧪"):
        st.subheader("🧪 独自短期設定")
        pl = st.selectbox(
            "株価をさかのぼる期間", ["6か月","1年","2年"], index=1,
            help="何を変える？：スキャンに使う過去の株価データ量です。長くすると過去の値動きを多く参照できますが、取得データ量が増えて処理が重くなる場合があります。"
        )
        period = {"6か月":"6mo","1年":"1y","2年":"2y"}[pl]

        slope_days = st.slider(
            "75日線トレンド判定期間", 5, 40, 20, 1,
            help="何を変える？：独自短期で75日線の方向を見る比較期間です。小さくすると最近のトレンド変化に敏感になり、大きくすると中期的な方向を重視します。例：20なら現在の75日線と20営業日前を比較します。"
        )
        buy_ticks = st.slider(
            "押し目反転確認ティック数", 1, 5, 2, 1,
            help="何を変える？：独自短期の押し目候補で、前日高値を何ティック上抜けたら『反転確認』とするかです。大きくするとダマシを避けやすい反面、買い判定は遅くなります。小さくすると早めに反応します。"
        )
        a_stop_buffer_pct = st.slider(
            "押し目の損切りバッファ",0.10,2.00,0.35,0.05,format="%.2f%%",
            help="何を変える？：独自短期の押し目候補で、75日線から何％下を損切り参考値にするかです。大きくすると値動きへの許容幅が広がりますが、損失幅も大きくなります。"
        )
        breakout_days = st.slider(
            "ブレイク判定期間",20,120,60,10,
            help="何を変える？：独自短期で過去何営業日の高値をブレイク基準にするかです。小さくすると短期の高値更新を拾いやすく、大きくするとより強い節目の突破を重視します。例：60なら過去60営業日の高値を基準にします。"
        )
        c_check_count = st.slider(
            "決算を詳しく確認する上位銘柄数",30,200,100,10,
            help="何を変える？：独自短期の上位候補のうち、決算情報を詳しく取得する銘柄数です。増やすほど決算面を広く確認できますが、Yahoo Financeへの個別取得が増えるためスキャン時間も長くなります。"
        )
        st.caption("逆指値発動後の指値幅は自動計算（固定ティックは廃止）。ATR＝普段1日の値動き幅の15%を基本に、発動価格の0.30%を上限、最低2ティックとして呼値単位に丸めます。")

        st.divider()
        st.subheader("過去類似シグナル検証")
        st.caption("v19.5では通常の過去類似シグナル成績に加え、ブレイク時の『発動後指値の許容幅』も過去データで比較します。")
        bt_top_n = st.slider(
            "検証する上位銘柄数",5,30,10,5,
            help="何を変える？：独自短期ランキングの上位何銘柄まで過去類似シグナルを検証するかです。増やすほど検証対象は広がりますが、バックテスト処理も重くなります。"
        )
        bt_horizon = st.selectbox(
            "何営業日先まで検証するか",[5,10,20],index=0,
            help="何を変える？：過去に類似シグナルが出た後、何営業日先までの値動きを検証対象にするかです。5日は約1週間、10日は約2週間、20日は約1か月のイメージです。"
        )
        bt_target = st.selectbox(
            "上昇目標",[3.0,5.0,8.0,10.0],index=1,format_func=lambda x:f"+{x:.0f}%",
            help="何を変える？：過去類似シグナル検証で『上昇成功』と見る目標率です。例：+5%なら、設定した検証期間内にシグナル価格から5%以上上昇したかを確認します。高くすると成功条件が厳しくなります。"
        )
        bt_stop = st.selectbox(
            "下落警戒ライン",[2.0,3.0,5.0],index=1,format_func=lambda x:f"-{x:.0f}%",
            help="何を変える？：過去類似シグナル検証で『下落警戒』と見る下落率です。例：-3%ならシグナル価格から3%下落したかを確認します。小さい値ほど下落に厳しい判定になります。"
        )

    elif mode.startswith("🏦"):
        st.subheader("🏦 長期・年初来安値設定")
        st.caption("短期売買用の75日線・ブレイク・バックテスト設定は長期モードでは表示しません。")

        # 年初来安値を正しく見るため、少なくとも1年分を取得。
        period = "1y"
        st.info("株価取得期間は **1年固定**。年初来安値を1月から確認するためです。")

        long_fund_n = st.slider(
            "ファンダメンタルを確認する上位銘柄数",
            20, 150, 80, 10,
            help="何を変える？：年初来安値に近い候補のうち、時価総額・ROE・配当・PER・PBRなどを詳しく取得する上位銘柄数です。増やすほど候補を広く調べられますが、長期モードでは特に処理時間へ影響します。"

        )
        long_mcap_filter = st.selectbox(
            "大手企業フィルター",
            ["制限なし","1,000億円以上","5,000億円以上","1兆円以上"],
            index=2,
            help="何を変える？：長期候補として残す企業の最低時価総額です。大きくすると大型株中心に絞り込みます。「制限なし」では時価総額による除外を行いません。"
        )
        long_max_low_dist = st.slider(
            "年初来安値から何%以内を重点表示するか",
            1, 30, 10, 1,
            help="何を変える？：現在値が年初来安値から何％以内なら『安値に近い』候補として重点表示するかです。小さくすると年初来安値ギリギリの銘柄に厳しく絞り、大きくすると候補範囲を広げます。例：10%なら年初来安値から10%以内です。"
        )

    elif mode.startswith("📕"):
        st.subheader("📕 ちょる子式設定")
        st.info("本由来の4条件は固定表示し、閾値を勝手に変更しない設計です。")
        choruko_mcap = st.selectbox(
            "大型株フィルター",["1,000億円以上","5,000億円以上","1兆円以上"],index=1,
            help="対象とする最低時価総額です。読書メモの『大型株中心』をアプリ側で具体化した補助条件です。"
        )
        choruko_material_default = st.selectbox(
            "材料確認の初期状態",["未確認","悪材料なし確認済","悪材料あり"],index=0,
            help="『材料なし』は株価だけでは判定できません。ニュース・適時開示等を確認した場合のみ変更してください。"
        )

    elif mode.startswith("🔎"):
        st.subheader("🔎 個別分析設定")
        st.info("個別分析では左側の細かいスキャン設定は不要です。銘柄コード・取得単価・保有株数はメイン画面で入力します。分析には1年分の日足を使用します。")

try:
    all_u = get_jpx_universe()
except Exception as e:
    st.error(f"東証銘柄一覧を取得できませんでした：{e}")
    st.stop()

if "selected_markets_v12" not in st.session_state:
    st.session_state.selected_markets_v12 = ["プライム"]
if "run_scan_v10" not in st.session_state:
    st.session_state.run_scan_v10 = False
if "scan_cache_v1963" not in st.session_state:
    st.session_state.scan_cache_v1963 = {}
if "holdings_v18" not in st.session_state:
    st.session_state.holdings_v18 = []
if "last_individual_v18" not in st.session_state:
    st.session_state.last_individual_v18 = None

# ------------------------------------------------------------
# v18 保有銘柄・個別分析モード
# ------------------------------------------------------------
if mode.startswith("🔎"):
    st.subheader("🔎 保有銘柄・個別分析")
    st.caption("ランキングとは別に、買った後の銘柄を『保有者目線』で分析します。外部AI APIは使わず、株価・移動平均・出来高・ATR・決算データをルールベースで文章化します。")

    tab1,tab2=st.tabs(["🔎 1銘柄を分析","📋 保有銘柄一覧"])

    with tab1:
        with st.form("individual_analysis_form"):
            c1,c2,c3=st.columns(3)
            code_input=c1.text_input("銘柄コード",value=(st.session_state.last_individual_v18 or {}).get("code","7267"),help="東証の4桁銘柄コードを入力します。例：本田技研工業は7267。")
            buy_input=c2.number_input("取得単価（円）",min_value=0.0,value=float((st.session_state.last_individual_v18 or {}).get("buy_price",0.0)),step=1.0,help="実際に買った平均取得単価です。未保有・新規分析なら0円のままでも分析できます。")
            shares_input=c3.number_input("保有株数",min_value=0,value=int((st.session_state.last_individual_v18 or {}).get("shares",0)),step=100,help="保有している株数です。入力すると含み損益金額を計算します。")
            add_holding=st.checkbox("分析後、この銘柄を保有一覧に登録/更新する",value=False)
            submitted=st.form_submit_button("🔍 個別分析する",type="primary",use_container_width=True)

        if submitted:
            st.session_state.last_individual_v18={"code":code_input,"buy_price":buy_input,"shares":shares_input}
            if add_holding and normalize_code(code_input):
                norm=normalize_code(code_input)
                new_item={"code":norm,"buy_price":float(buy_input),"shares":int(shares_input)}
                existing=[x for x in st.session_state.holdings_v18 if x["code"]!=norm]
                existing.append(new_item)
                st.session_state.holdings_v18=existing

        last=st.session_state.last_individual_v18
        if last:
            render_single_holding_analysis(all_u,last["code"],last["buy_price"],last["shares"])
        else:
            st.info("銘柄コードを入力して「個別分析する」を押してください。")

    with tab2:
        st.markdown("### 📋 登録済み保有銘柄")
        if not st.session_state.holdings_v18:
            st.info("まだ保有銘柄が登録されていません。『1銘柄を分析』から登録できます。")
        else:
            basic=[]
            for h in st.session_state.holdings_v18:
                hit=all_u[all_u["コード"]==h["code"]]
                name=hit.iloc[0]["銘柄名"] if not hit.empty else ""
                basic.append({"コード":h["code"],"銘柄名":name,"取得単価":h["buy_price"],"保有株数":h["shares"]})
            st.dataframe(pd.DataFrame(basic),use_container_width=True,hide_index=True)

            selected_code=st.selectbox(
                "詳しく分析する保有銘柄",
                [x["code"] for x in st.session_state.holdings_v18],
                format_func=lambda c: next((f"{c} {r['銘柄名']}" for _,r in all_u[all_u["コード"]==c].iterrows()),c)
            )
            selected=next(x for x in st.session_state.holdings_v18 if x["code"]==selected_code)
            render_single_holding_analysis(all_u,selected["code"],selected["buy_price"],selected["shares"])

            st.divider()
            del_code=st.selectbox("保有一覧から削除する銘柄",[x["code"] for x in st.session_state.holdings_v18],key="delete_holding_code")
            if st.button("🗑️ 選択した保有銘柄を削除"):
                st.session_state.holdings_v18=[x for x in st.session_state.holdings_v18 if x["code"]!=del_code]
                st.rerun()

    st.warning("個別分析は売買を自動執行する機能ではありません。『新規買い評価』と『保有者としての評価』を意図的に分けています。")
    st.stop()

if mode.startswith("📕"):
    st.subheader("📕 ちょる子式｜大型株逆張り")
    st.caption("📕＝読書メモ＋確認済み書評 / 🤖＝アプリ独自補助。両者を混ぜずに表示します。")
    with st.expander("📚 判定条件と注意点"):
        st.markdown("""
**📕 4つの売られすぎ条件**
1. 材料なしで前日終値比 **-2.5%以上**
2. **25日線を下回る**（『大きく割り込む』の厳密％は資料で確定できないため乖離率を併記）
3. ボリンジャーバンド **-3σ以下**
4. RCI **-90以下**

**🤖 アプリ独自補助**
- 前日高値突破または陽線化を反転確認として表示
- 時価総額で大型株を絞り込み
- 「材料なし」は自動断定しない

**出口参考**
- 25日線への回帰
- 急落前終値への回帰
""")

st.subheader("🏢 スキャンする市場")
st.caption("複数選択できます。例：プライム＋スタンダード。初期設定はプライムのみです。")
with st.expander("⚡ スキャン速度について"):
    st.markdown("""
v17.4では次を高速化しています。

- 株価データは**30分キャッシュ**し、同じ市場・取得期間なら再利用
- 決算データ、決算予定日、長期ファンダメンタルは**最大6銘柄ずつ並列取得**
- 長期モードでは不要な短期テクニカル計算を省略
- モード別の前回スキャン結果はセッション内で保持

それでも時間がかかる主因は、Yahoo Financeへ銘柄ごとに取得する**決算・企業情報**です。  
速度優先ならサイドバーの「決算を詳しく確認する上位銘柄数」「ファンダメンタルを確認する上位銘柄数」を減らしてください。
""")

market_options = ["プライム", "スタンダード", "グロース", "TOKYO PRO"]
selected_markets = st.multiselect(
    "対象市場",
    market_options,
    default=st.session_state.selected_markets_v12,
    help="選択した市場をまとめて1つのランキングにします。"
)
st.session_state.selected_markets_v12 = selected_markets

parts = []
for m in selected_markets:
    x = select_market_universe(all_u, m)
    if x is not None and not x.empty:
        parts.append(x)

if parts:
    universe = pd.concat(parts, ignore_index=True).drop_duplicates("コード")
else:
    universe = pd.DataFrame(columns=all_u.columns)

market_label = "＋".join(selected_markets) if selected_markets else "未選択"

m1,m2,m3 = st.columns(3)
m1.metric("選択中", market_label)
m2.metric("対象", f"{len(universe):,}銘柄")
m3.metric("モード", "本ベース" if mode.startswith("📘") else ("長期・年初来安値" if mode.startswith("🏦") else "独自短期"))

regime = get_market_regime()
if mode.startswith("🏦"):
    st.info(f"**市場トレンド（長期では参考）：{regime['label']}**  長期モードでは市場トレンドより、企業の質と年初来安値への接近を重視します。")
else:
    st.info(f"**市場トレンド：{regime['label']}（{regime['score']:.0f}/100）**  {regime['comment']}")

with st.expander("📐 市場トレンド判定の仕組み"):
    st.markdown("""
**TOPIXと日経平均をそれぞれ100点満点で採点し、2指数の平均を表示します。**

- 現在値 ＞ 20日移動平均線：**+35点**
- 20日移動平均線 ＞ 75日移動平均線：**+30点**
- 75日移動平均線が10営業日前より上昇：**+25点**
- 直近5営業日の騰落率がプラス：**+10点**

**総合判定**
- 80点以上：🟢 強気
- 60〜79点：🟡 やや強気
- 40〜59点：🟠 中立
- 40点未満：🔴 弱気

これは日経平均・TOPIXの**トレンド状態**を見る独自ルールです。市場心理や騰落銘柄数などを含む広義の「地合い」全体を表す指標ではありません。
""")
    if regime["details"]:
        st.dataframe(pd.DataFrame(regime["details"]), use_container_width=True, hide_index=True)
    else:
        st.write("指数データを取得できませんでした。")

if not selected_markets:
    st.warning("市場を1つ以上選択してください。")

cache_payload = get_mode_cache(mode, selected_markets)

btn1, btn2 = st.columns([4,1])
with btn1:
    if st.button(f"🚀 {market_label}をスキャン", type="primary", use_container_width=True, disabled=not selected_markets):
        download_batch.clear()
        latest_jp_market_date.clear()
        fetch_ticker_diagnostic_data.clear()
        st.session_state.run_scan_v10=True
with btn2:
    if st.button("🗑️ 保持結果を削除", use_container_width=True, disabled=cache_payload is None):
        key = mode_cache_key(mode, selected_markets)
        st.session_state.scan_cache_v1963.pop(key, None)
        cache_payload = None

if cache_payload is not None and not st.session_state.run_scan_v10:
    cached_result_banner(cache_payload)

if not st.session_state.run_scan_v10 and cache_payload is not None:
    # 前回の結果を復元
    cached_mode = cache_payload.get("mode")
    if cached_mode == "book":
        ranked = cache_payload["ranked"].copy()
        tech = cache_payload["tech"].copy()
        st.subheader("📘 本ベース｜総合ランキング")
        st.caption("前回スキャン結果を表示しています。必要なときだけ再スキャンしてください。")
        st.dataframe(
            ranked[["順位","銘柄","Yahoo!チャート","株価","前日比","前日比%","75日線","75日線_乖離率%","A","B","C","D","該当戦略数","買い価格目安","A初期損切り目安","B初期損切り目安","総合スコア","総合診断"]],
            use_container_width=True,hide_index=True,
            column_config={
                "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗"),
                "株価":st.column_config.NumberColumn("株価",format="%.0f円"),
                "前日比":st.column_config.NumberColumn("前日比",format="%+.0f円"),
                "前日比%":st.column_config.NumberColumn("前日比%",format="%+.2f%%"),
                "75日線":st.column_config.NumberColumn("75日線",format="%.0f円"),
                "75日線_乖離率%":st.column_config.NumberColumn("75日線乖離",format="%+.2f%%"),
                "買い価格目安":st.column_config.NumberColumn("買い目安",format="%.0f円"),
                "A初期損切り目安":st.column_config.NumberColumn("A損切り",format="%.0f円"),
                "B初期損切り目安":st.column_config.NumberColumn("B損切り",format="%.0f円"),
                "総合スコア":st.column_config.NumberColumn("総合",format="%.1f"),
            }
        )
        st.stop()

    elif cached_mode == "ai":
        tech = cache_payload["tech"].copy()
        st.subheader("📊 実戦ランキング")
        st.warning("⏸️ 前回スキャン結果を表示中です。株価は自動更新されません。最新日足で判定するには「スキャン」を押してください。")
        st.dataframe(
            safe_columns(tech.head(100), ["順位","実戦優先度","銘柄","Yahoo!チャート","株価","前日比","前日比%","短期総合スコア","セットアップ","ルール評価","注文種類","買い逆指値発動価格表示","発動後買い指値表示","損切り逆指値発動価格表示","発動後売り指値表示","利確目安①表示","RR","決算警告"]),
            use_container_width=True,hide_index=True,
            column_config=practical_ranking_column_config()
        )
        practical_ranking_explainer()
        with st.expander("🔎 詳細を見る"):
            st.dataframe(
                safe_columns(tech.head(100), ["順位","銘柄","始値","高値","安値","前日終値","前日比%","上昇力","今の買いやすさ","出来高_20日平均比","75日線","75日線_比較期間前比%","75日線_乖離率%","追加シグナル","セットアップ判定根拠","売買シナリオ","注文条件","注文理由","買い逆指値発動価格表示","発動後買い指値表示","発動後買い指値の根拠","注文価格の根拠","損切り逆指値発動価格表示","発動後売り指値表示","発動後売り指値の根拠","損切り価格の根拠","利確価格の根拠","損切り注文","損切り価格表示","利確目安①表示","利確目安②表示","想定初期リスク%","想定利益%","RR","次回決算日","決算警告","評価コメント"]),
                use_container_width=True,hide_index=True
            )
        st.stop()

    elif cached_mode == "long":
        long_df = cache_payload["long_df"].copy()
        st.subheader("🏦 長期・年初来安値｜総合候補")
        st.warning("⏸️ 前回スキャン結果を表示中です。株価は自動更新されません。最新日足で判定するには「スキャン」を押してください。")
        st.dataframe(
            long_df.head(100)[["順位","長期判定","銘柄","Yahoo!チャート","株価","前日比%","年初来安値","年初来安値日","年初来安値から%","安値接近度","企業クオリティ","配当・割安度","長期総合スコア","時価総額_億円","配当利回り%","ROE%","予想PER","PBR","長期買い方","1回目","2回目","3回目"]],
            use_container_width=True,hide_index=True,
            column_config={
                "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗"),
                "株価":st.column_config.NumberColumn("現在値",format="%.0f円"),
                "前日比%":st.column_config.NumberColumn("前日比%",format="%+.2f%%"),
                "年初来安値":st.column_config.NumberColumn("年初来安値",format="%.0f円"),
                "年初来安値から%":st.column_config.NumberColumn("安値から",format="%+.2f%%"),
                "長期総合スコア":st.column_config.ProgressColumn("長期総合",min_value=0,max_value=100,format="%.1f"),
            }
        )
        st.stop()

if st.session_state.run_scan_v10:
    st.session_state.run_scan_v10=False
    if universe.empty:
        st.warning("対象銘柄を取得できませんでした。")
        st.stop()

    status=st.empty()
    progress=st.progress(0)
    status.info(f"① {market_label}の株価データを取得しています…")
    st.caption("⚡ v19.6.3：全銘柄の個別再取得を廃止。一括取得＋鮮度監査で高速化しています。")
    data=download_batch(universe.ticker.tolist(),period)
    market_data_date=latest_jp_market_date()
    data,stale_tickers=refresh_lagging_tickers(data,universe.ticker.tolist(),market_data_date)
    stale_set=set(stale_tickers)
    batch_requested=len(universe); batch_received=len(data)
    if market_data_date is not None:
        st.info(f"📅 今回のランキング基準日：{market_data_date}（日経平均の日足最終日）")
    else:
        st.warning("⚠️ 日本市場の最新日を確認できません。ランキング基準日の保証ができません。")
    if stale_tickers:
        st.error(f"⚠️ {len(stale_tickers)}銘柄の日足が市場基準日 {market_data_date} より古いため、古い価格で順位付けせず今回のランキングから除外します。全銘柄の個別再取得は行いません。")
    rows=[]; total=len(universe)

    for i,row in universe.reset_index(drop=True).iterrows():
        if row.ticker in stale_set: continue
        d=data.get(row.ticker)
        if d is not None:
            if mode.startswith("🏦"):
                r=long_price_metrics(d)
            elif mode.startswith("📕"):
                r=choruko_metrics(d)
            else:
                r=technical_scan(d,slope_days,max_dev,breakout_days,a_stop_buffer_pct,buy_ticks)

            if r:
                r.update({
                    "ticker":row.ticker,"コード":row["コード"],"銘柄名":row["銘柄名"],
                    "銘柄":row["銘柄"],"市場":row["市場"],"業種":row["業種"],
                    "Yahoo!チャート":row["Yahoo!チャート"]
                })
                rows.append(r)
        if i%20==0:
            progress.progress(min(.68,(i+1)/max(total,1)*.68))

    if not rows:
        status.error("株価データを取得できませんでした。")
        st.stop()

    if batch_received < batch_requested:
        missing_count = batch_requested - batch_received
        st.warning(f"⚠️ 一括株価取得：{batch_received:,}/{batch_requested:,}銘柄。{missing_count:,}銘柄は今回の一括取得でデータを受け取れませんでした。ランキング外＝条件外とは限りません。")

    tech=pd.DataFrame(rows)

    # モード別の事前順位。
    if mode.startswith("📘"):
        tech["事前スコア"]=tech["Aスコア"]*.375+tech["Bスコア"]*.3125+tech["Dスコア"]*.3125
    elif mode.startswith("🏦"):
        tech["事前スコア"]=tech["安値接近度"].fillna(0)
    elif mode.startswith("📕"):
        tech["事前スコア"]=tech["底打ち条件数"]*25
    else:
        tech["事前スコア"]=tech.apply(ai_pre_score,axis=1)

    tech=tech.sort_values("事前スコア",ascending=False).reset_index(drop=True)

    # 長期モードは短期C判定をせず、上位候補のファンダメンタルを取得してここで完結。
    if mode.startswith("🏦"):
        status.info("② 年初来安値に近い候補の企業情報を確認しています…")
        fund_n=min(long_fund_n, len(tech))
        fund_tickers=tech.head(fund_n).ticker.tolist()
        status.info(f"② 年初来安値に近い上位{fund_n}銘柄の企業情報を並列取得しています…")
        f_map=parallel_fetch(fund_tickers, long_term_fundamentals, max_workers=6)
        progress.progress(1.0)
        progress.empty()

        fund_rows=[]
        for _,rr in tech.head(fund_n).iterrows():
            x=rr.to_dict()
            x.update(f_map.get(rr["ticker"],{}))
            fund_rows.append(x)
        long_df=pd.DataFrame(fund_rows)

        scores=long_df.apply(long_quality_scores,axis=1)
        long_df=pd.concat([long_df,scores],axis=1)
        plans=long_df.apply(long_buy_plan,axis=1)
        long_df=pd.concat([long_df,plans],axis=1)

        # 大手基準
        mcap_threshold={
            "制限なし":0,
            "1,000億円以上":1000,
            "5,000億円以上":5000,
            "1兆円以上":10000,
        }[long_mcap_filter]
        long_df["大手条件"] = long_df["時価総額_億円"].fillna(-1) >= mcap_threshold if mcap_threshold>0 else True
        long_df["重点安値圏"] = long_df["年初来安値から%"] <= long_max_low_dist

        long_df=long_df.sort_values(
            ["長期総合スコア","安値接近度","企業クオリティ"],
            ascending=False
        ).reset_index(drop=True)
        long_df.insert(0,"順位",np.arange(1,len(long_df)+1))
        status.success("長期・年初来安値ランキングを作成しました。")
        set_mode_cache(mode, selected_markets, {
            "mode":"long",
            "timestamp":pd.Timestamp.now(),
            "long_df":long_df.copy(),
        })

        st.subheader("🏦 長期・年初来安値｜総合候補")
        st.caption("『安い＝買い』ではありません。年初来安値への近さと、取得できた企業クオリティ・配当/バリュエーションを分けて評価します。ファンダメンタルは年初来安値接近上位だけ取得するため、全上場銘柄を完全比較するものではありません。")

        focus=long_df[long_df["重点安値圏"]].copy()
        if mcap_threshold>0:
            focus=focus[focus["大手条件"]]

        if focus.empty:
            st.info("現在の設定では総合候補がありません。年初来安値からの距離または大手フィルターを緩めてください。")
        else:
            st.dataframe(
                focus[[
                    "順位","長期判定","銘柄","Yahoo!チャート","株価","前日比%",
                    "年初来安値","年初来安値日","年初来安値から%",
                    "安値接近度","企業クオリティ","配当・割安度","長期総合スコア",
                    "時価総額_億円","配当利回り%","ROE%","予想PER","PBR",
                    "長期買い方","1回目","2回目","3回目"
                ]],
                use_container_width=True,hide_index=True,
                column_config={
                    "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗"),
                    "株価":st.column_config.NumberColumn("現在値",format="%.0f円"),
                    "前日比%":st.column_config.NumberColumn("前日比%",format="%+.2f%%"),
                    "年初来安値":st.column_config.NumberColumn("年初来安値",format="%.0f円"),
                    "年初来安値から%":st.column_config.NumberColumn("安値から",format="%+.2f%%"),
                    "安値接近度":st.column_config.ProgressColumn("安値接近度",min_value=0,max_value=100,format="%.1f"),
                    "企業クオリティ":st.column_config.ProgressColumn("企業クオリティ",min_value=0,max_value=100,format="%.1f"),
                    "配当・割安度":st.column_config.ProgressColumn("配当・割安度",min_value=0,max_value=100,format="%.1f"),
                    "長期総合スコア":st.column_config.ProgressColumn("長期総合",min_value=0,max_value=100,format="%.1f"),
                    "時価総額_億円":st.column_config.NumberColumn("時価総額",format="%.0f億円"),
                    "配当利回り%":st.column_config.NumberColumn("配当利回り",format="%.2f%%"),
                    "ROE%":st.column_config.NumberColumn("ROE",format="%.1f%%"),
                    "予想PER":st.column_config.NumberColumn("予想PER",format="%.1f倍"),
                    "PBR":st.column_config.NumberColumn("PBR",format="%.2f倍"),
                }
            )

        tabs=st.tabs(["📉 年初来安値接近","🏢 大手だけ","💰 高配当","⭐ 総合長期候補","🔎 詳細"])
        with tabs[0]:
            x=long_df.sort_values("年初来安値から%",ascending=True)
            st.dataframe(
                x[["順位","銘柄","Yahoo!チャート","株価","年初来安値","年初来安値から%","長期判定","企業クオリティ","時価総額_億円"]],
                use_container_width=True,hide_index=True,
                column_config={"Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗")}
            )
        with tabs[1]:
            x=long_df[long_df["大手条件"]].sort_values(["年初来安値から%","企業クオリティ"],ascending=[True,False])
            if x.empty:
                st.info("大手条件に該当する取得済み候補がありません。")
            else:
                st.dataframe(x[["順位","銘柄","Yahoo!チャート","株価","年初来安値から%","時価総額_億円","企業クオリティ","長期判定"]],use_container_width=True,hide_index=True,
                    column_config={"Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗")})
        with tabs[2]:
            x=long_df[long_df["配当利回り%"].fillna(0)>=3.0].sort_values(["配当利回り%","企業クオリティ"],ascending=False)
            if x.empty:
                st.info("配当利回り3%以上の取得済み候補はありません。")
            else:
                st.dataframe(x[["順位","銘柄","Yahoo!チャート","株価","年初来安値から%","配当利回り%","ROE%","長期判定"]],use_container_width=True,hide_index=True,
                    column_config={"Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗")})
        with tabs[3]:
            x=long_df[long_df["長期判定"].isin(["🟢 長期候補","🟡 分割買い候補"])].sort_values("長期総合スコア",ascending=False)
            if x.empty:
                st.info("現在の取得済み候補に長期候補はありません。")
            else:
                st.dataframe(x[["順位","長期判定","銘柄","Yahoo!チャート","株価","年初来安値から%","企業クオリティ","配当・割安度","長期総合スコア","1回目","2回目","3回目"]],use_container_width=True,hide_index=True,
                    column_config={"Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗")})
        with tabs[4]:
            st.dataframe(
                long_df[[
                    "順位","銘柄","年初来安値","年初来安値から%","年初来高値","年初来高値から%",
                    "時価総額_億円","ROE%","営業利益率%","売上成長率%","利益成長率%",
                    "配当利回り%","予想PER","実績PER","PBR","負債資本倍率","配当性向%",
                    "長期判定","長期前提崩れ"
                ]],
                use_container_width=True,hide_index=True
            )

        csv=long_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 長期・年初来安値ランキングをCSV保存",
            csv,
            f"長期_年初来安値_{market_label}.csv",
            "text/csv",
            use_container_width=True
        )

        st.warning("長期モードは『年初来安値だから買う』機能ではありません。業績悪化・減配・構造的な競争力低下などで安値になっている可能性があります。分割購入案は参考値で、損失を限定するものではありません。")
        st.stop()

    if mode.startswith("📕"):
        status.info("② 大型株候補の時価総額を確認しています…")
        threshold={"1,000億円以上":1000,"5,000億円以上":5000,"1兆円以上":10000}[choruko_mcap]
        candidates=tech.sort_values(["底打ち条件数","前日比%"],ascending=[False,True]).head(min(120,len(tech))).copy()
        fmap=parallel_fetch(candidates.ticker.tolist(),long_term_fundamentals,max_workers=6)
        candidates["時価総額_億円"]=[fmap.get(t,{}).get("時価総額_億円",np.nan) for t in candidates.ticker]
        candidates=candidates[candidates["時価総額_億円"].fillna(-1)>=threshold].copy()
        candidates["材料確認"]=choruko_material_default

        vals=[]
        for _,rr in candidates.iterrows():
            j,reason=choruko_judgment(rr,choruko_material_default)
            p=choruko_exit_plan(rr)
            x={"判定":j,"判定理由":reason,**p}
            vals.append(x)
        extra=pd.DataFrame(vals,index=candidates.index)
        candidates=pd.concat([candidates,extra],axis=1)
        candidates=candidates.sort_values(["底打ち条件数","反転確認_独自","前日比%"],ascending=[False,False,True]).reset_index(drop=True)
        candidates.insert(0,"順位",np.arange(1,len(candidates)+1))
        progress.progress(1.0); progress.empty()
        status.success("ちょる子式候補を作成しました。")

        st.subheader("📕 ちょる子式｜大型株逆張りランキング")
        st.caption("4条件を個別表示します。『材料なし』は自動断定せず、既定は未確認です。4/4でも即買い判定ではありません。")
        if candidates.empty:
            st.info("現在の大型株フィルターでは候補がありません。")
            st.stop()

        st.dataframe(
            candidates[["順位","判定","銘柄","Yahoo!チャート","現在値","前日比%","急落-2.5%","25日線乖離%","25日線割れ","BBσ","BB-3σ","RCI9","RCI-90","底打ち条件数","材料確認","反転確認_独自","時価総額_億円","利確参考①","利確参考①根拠","利確参考②","利確参考②根拠"]],
            use_container_width=True,hide_index=True,
            column_config={
                "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗"),
                "現在値":st.column_config.NumberColumn("現在値",format="%.0f円"),
                "前日比%":st.column_config.NumberColumn("前日比%",format="%+.2f%%"),
                "25日線乖離%":st.column_config.NumberColumn("25日線乖離",format="%+.2f%%"),
                "BBσ":st.column_config.NumberColumn("BB位置",format="%.2fσ"),
                "RCI9":st.column_config.NumberColumn("RCI(9)",format="%.1f"),
                "時価総額_億円":st.column_config.NumberColumn("時価総額",format="%.0f億円"),
                "利確参考①":st.column_config.NumberColumn("利確①",format="%.0f円"),
                "利確参考②":st.column_config.NumberColumn("利確②",format="%.0f円"),
            }
        )
        with st.expander("❓ 各列の意味"):
            st.markdown("""
- **急落-2.5%**：前日終値比-2.5%以上なら○
- **25日線乖離 / 25日線割れ**：25日線より下かと、どの程度離れたか
- **BB位置 / BB-3σ**：ボリンジャーバンド上の標準偏差位置
- **RCI(9) / RCI-90**：9日RCI。-90以下なら○
- **底打ち条件数**：4条件中の該当数。4/4でも即買いではありません
- **材料確認**：自動ニュース判定ではありません
- **反転確認_独自**：🤖陽線化または前日高値突破
- **利確①/②**：📕25日線・急落前終値への平均回帰を参考
""")
        st.warning("『-2.5%下がったら買い』機能ではありません。大型株・売られすぎ・材料・反転を分けて確認します。")
        st.stop()

    n=min(c_check_count,len(tech))
    c_tickers=tech.head(n).ticker.tolist()
    status.info(f"② 上位{n}銘柄の決算データを並列取得しています…")
    cmap=parallel_fetch(c_tickers, financial_momentum, max_workers=6)
    progress.progress(1.0)
    progress.empty()

    tech["C"]=[cmap.get(t,{}).get("C",False) for t in tech.ticker]
    tech["Cスコア"]=[cmap.get(t,{}).get("Cスコア",0) for t in tech.ticker]
    tech["営業利益_前年同期比%"]=[cmap.get(t,{}).get("営業利益_前年同期比%",np.nan) for t in tech.ticker]
    tech["売上高_前年同期比%"]=[cmap.get(t,{}).get("売上高_前年同期比%",np.nan) for t in tech.ticker]
    tech["C確認済み"]=[t in cmap for t in tech.ticker]
    tech["Cデータ取得成功"]=[
        bool(cmap.get(t)) and (
            np.isfinite(cmap.get(t,{}).get("営業利益_前年同期比%", np.nan))
            or np.isfinite(cmap.get(t,{}).get("売上高_前年同期比%", np.nan))
        )
        for t in tech.ticker
    ]

    c_checked = int(tech["C確認済み"].sum())
    c_success = int(tech["Cデータ取得成功"].sum())
    c_hits = int(tech["C"].sum())

    st.subheader("🧾 決算モメンタム（C）の取得状況")
    c1,c2,c3 = st.columns(3)
    c1.metric("Cを確認した銘柄", f"{c_checked:,} / {len(tech):,}")
    c2.metric("決算データ取得成功", f"{c_success:,}")
    c3.metric("C該当", f"{c_hits:,}")
    st.caption("Cは処理時間短縮のため上位候補だけ詳しく取得します。Cが付かない＝決算が悪い、とは限りません。未確認・データ不足の銘柄もあります。")

    if mode.startswith("📘"):
        status.success("本ベースランキングを作成しました。")
        tech["該当戦略数"]=tech[["A","B","C","D"]].sum(axis=1)
        tech["総合スコア"]=(tech.Aスコア*.30+tech.Bスコア*.25+tech.Cスコア*.20+tech.Dスコア*.25+tech["該当戦略数"]*5).clip(upper=120)

        def buy_book(r):
            if r.A:return r["A買い価格"]
            if r.B:return r["B買い価格"]
            if r.D:return r["D買い価格"]
            return np.nan
        tech["買い価格目安"]=tech.apply(buy_book,axis=1)
        tech["A初期損切り目安"]=np.where(tech.A,tech["A初期損切り"],np.nan)
        tech["B初期損切り目安"]=np.where(tech.B,tech["B初期損切り"],np.nan)
        tech["該当戦略"]=tech.apply(lambda r:"・".join([s for s in ["A","B","C","D"] if bool(r[s])]) or "－",axis=1)

        # Reuse v9-style diagnosis with MA75 overheat.
        def book_diag(r):
            hits=int(r["該当戦略数"]); score=float(r["総合スコア"])
            dev=float(r["75日線_乖離率%"]); ext=float(r["Bブレイク上昇率%"]) if np.isfinite(r["Bブレイク上昇率%"]) else -99
            if bool(r["B"]) and (dev>30 or ext>10): return "🔴 E｜追いかけ注意"
            if bool(r["B"]) and (dev>20 or ext>7): return "🟠 D｜押し待ち"
            if bool(r["A"]) and score>=70: return "🟢 S｜かなり良い形"
            if hits>=2 and score>=65: return "🟢 A｜良好"
            if bool(r["B"]) and ext<=3 and dev<=10 and r["B出来高倍率"]>=1.5 and score>=55: return "🟢 A｜良好"
            if hits>=1 and score>=50: return "🟡 B｜候補"
            if hits>=1: return "🟡 C｜慎重"
            return "⚪ 対象外"
        tech["総合診断"]=tech.apply(book_diag,axis=1)
        tech=tech.sort_values(["総合スコア","該当戦略数"],ascending=False).reset_index(drop=True)
        tech.insert(0,"順位",np.arange(1,len(tech)+1))
        for s in ["A","B","C","D"]:
            tech[s]=tech[s].map(mark)
        ranked=tech[tech["該当戦略数"]>=1]

        set_mode_cache(mode, selected_markets, {
            "mode":"book",
            "timestamp":pd.Timestamp.now(),
            "tech":tech.copy(),
            "ranked":ranked.copy(),
        })

        st.subheader("📘 本ベース｜総合ランキング")
        st.caption("Aは『75日線が上向き』『株価が75日線の下』『乖離率 -3%〜0%』に限定。0%に近いほど高評価です。Aの買い目安は前日高値＋指定ティック超えによる反転確認価格です。")
        st.dataframe(
            ranked[["順位","銘柄","Yahoo!チャート","株価","前日比","前日比%","75日線","75日線_乖離率%","A","B","C","D","該当戦略数","買い価格目安","A初期損切り目安","B初期損切り目安","総合スコア","総合診断"]],
            use_container_width=True,hide_index=True,
            column_config={
                "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗"),
                "株価":st.column_config.NumberColumn("株価",format="%.0f円"),
                "前日比":st.column_config.NumberColumn("前日比",format="%+.0f円"),
                "前日比%":st.column_config.NumberColumn("前日比%",format="%+.2f%%"),
                "75日線":st.column_config.NumberColumn("75日線",format="%.0f円"),
                "75日線_乖離率%":st.column_config.NumberColumn("75日線乖離",format="%+.2f%%"),
                "買い価格目安":st.column_config.NumberColumn("買い目安",format="%.0f円"),
                "A初期損切り目安":st.column_config.NumberColumn("A損切り",format="%.0f円"),
                "B初期損切り目安":st.column_config.NumberColumn("B損切り",format="%.0f円"),
                "総合スコア":st.column_config.NumberColumn("総合",format="%.1f"),
            }
        )
        st.caption("Aは上向き75日線の直下（乖離 -3%〜0%）だけを候補にし、75日線に近いほど高評価。実際の買いは前日高値＋指定ティック超えの反転確認を待つ設計です。")

    else:
        status.success("短期スクリーニングランキングを作成しました。")
        ai = tech.apply(ai_scores,axis=1)
        tech=pd.concat([tech,ai],axis=1)
        tech=tech.sort_values(["短期総合スコア","今の買いやすさ","上昇力"],ascending=False).reset_index(drop=True)
        tech.insert(0,"順位",np.arange(1,len(tech)+1))

        # 決算日警告：上位候補だけ取得して速度を維持
        earnings_tickers=tech.head(min(30,len(tech))).ticker.tolist()
        earnings_infos=parallel_fetch(earnings_tickers, get_earnings_date_info, max_workers=6)
        tech["次回決算日"] = [earnings_infos.get(t,{}).get("date") for t in tech.ticker]
        tech["決算まで日数"] = [earnings_infos.get(t,{}).get("days") for t in tech.ticker]
        tech["決算警告"] = [earnings_infos.get(t,{}).get("warning","") for t in tech.ticker]

        # 地合いを診断コメントへ反映
        if regime["score"] < 40:
            tech["評価コメント"] = tech["評価コメント"].astype(str) + "／市場トレンド弱気のため新規買いは厳選"
        elif regime["score"] < 60:
            tech["評価コメント"] = tech["評価コメント"].astype(str) + "／市場トレンド中立"

        set_mode_cache(mode, selected_markets, {
            "mode":"ai",
            "timestamp":pd.Timestamp.now(),
            "tech":tech.copy(),
        })

        st.subheader("📊 実戦ランキング")
        st.caption("現在値 → 前日比 → 評価 → 楽天証券の買い注文 → 損切り → 利確 → RR。前日比は最新日足と1本前の日足の比較で、リアルタイム配信値ではありません。")

        # v19.6.3 データ鮮度監査
        if "データ最終日" in tech.columns:
            age_series = tech["データ最終日"].apply(business_day_age)
            stale_count = int((age_series.fillna(999) >= 2).sum())
            latest_dates = pd.to_datetime(tech["データ最終日"], errors="coerce").dropna()
            if not latest_dates.empty:
                newest = latest_dates.max().date()
                oldest = latest_dates.min().date()
                st.caption(f"📅 スキャン使用日足：最新 {newest} / 最古 {oldest} ｜ 2平日以上古い銘柄 {stale_count}件")
            if stale_count:
                st.warning(f"⚠️ {stale_count}銘柄で日足が2平日以上古い可能性があります。ランキングだけで判断せず、下の『銘柄診断』で個別再取得してください。")

        with st.expander("🩺 銘柄診断｜なぜランキングにいる／消えた？", expanded=False):
            st.caption("銘柄コードを入力すると、一括スキャンとは別にその銘柄の日足を再取得し、順位圏外・データ欠落・ブレイク条件を切り分けます。例：7752（リコー）")
            diag_code = st.text_input("診断する銘柄コード", value="7752", key="ai_diag_ticker_v196")
            if st.button("🔍 この銘柄を診断", key="run_ai_diag_v196"):
                diag = diagnose_ai_ticker(
                    diag_code, universe, tech,
                    slope_days, max_dev, breakout_days, a_stop_buffer_pct, buy_ticks
                )
                st.markdown(f"### {diag.get('銘柄', diag_code)}")
                st.write(f"**診断結論**：{diag.get('表示されない主因','—')}")
                d1,d2,d3,d4 = st.columns(4)
                d1.metric("現在順位", f"{diag['現在順位']}位" if diag.get("現在順位") else "ランキング外")
                d2.metric("個別再取得株価", f"{diag['個別再取得株価']:.1f}円" if isinstance(diag.get("個別再取得株価"), (int,float,np.integer,np.floating)) and np.isfinite(diag.get("個別再取得株価")) else "取得不可")
                d3.metric("データ最終日", str(diag.get("データ最終日") or "不明"))
                d4.metric("セットアップ", str(diag.get("セットアップ参考") or "—"))
                detail = pd.DataFrame([
                    ["市場対象", "○" if diag.get("市場対象") else "×"],
                    ["個別日足再取得", "○" if diag.get("個別再取得") else "×"],
                    ["データ鮮度警告", "⚠️ あり" if diag.get("データ鮮度警告") else "なし"],
                    ["直近高値との差", f"{diag.get('直近高値との差%',np.nan):+.2f}%" if np.isfinite(diag.get("直近高値との差%",np.nan)) else "—"],
                    ["ブレイク距離 -3〜0%", "○" if diag.get("ブレイク距離条件(-3〜0%)") else "×"],
                    ["75日線上向き", "○" if diag.get("75日線上向き") else "×"],
                    ["出来高 20日平均1.1倍以上", "○" if diag.get("出来高1.1倍以上") else "×"],
                    ["🔥ブレイク準備3条件", "○" if diag.get("ブレイク準備3条件") else "×"],
                ], columns=["確認項目","結果"])
                st.dataframe(detail, use_container_width=True, hide_index=True)
                st.caption("個別再取得株価も日足終値ベースで、リアルタイム株価ではありません。日中の現在値と一致しない場合があります。")
        st.dataframe(
            safe_columns(tech.head(100), ["順位","実戦優先度","銘柄","Yahoo!チャート","株価","前日比","前日比%","短期総合スコア","セットアップ","ルール評価","注文種類","買い逆指値発動価格表示","発動後買い指値表示","損切り逆指値発動価格表示","発動後売り指値表示","利確目安①表示","RR","決算警告"]),
            use_container_width=True,
            hide_index=True,
            column_config=practical_ranking_column_config()
        )
        practical_ranking_explainer()
        st.warning("逆指値は『発動条件』と『発動後の指値』を別々に表示しています。指値付き逆指値は価格が指値を飛び越えると約定しない場合があります。発動後指値の自動許容幅は経験則ベースです。v19.5では過去のブレイク発動日を使って、2ティック固定・0.10/0.20/0.30%・ATR方式を比較表示します。現行方式が最適とは限りません。呼値はアプリ内の簡易計算なので、最終入力時は楽天証券の注文画面でも確認してください。株価データはリアルタイム保証ではありません。")
        with st.expander("🔎 詳細を見る"):
            st.dataframe(
                safe_columns(tech.head(100), ["順位","銘柄","始値","高値","安値","前日終値","前日比%","上昇力","今の買いやすさ","出来高_20日平均比","75日線","75日線_比較期間前比%","75日線_乖離率%","追加シグナル","セットアップ判定根拠","売買シナリオ","注文条件","注文理由","買い逆指値発動価格表示","発動後買い指値表示","発動後買い指値の根拠","注文価格の根拠","損切り逆指値発動価格表示","発動後売り指値表示","発動後売り指値の根拠","損切り価格の根拠","利確価格の根拠","損切り注文","損切り価格表示","利確目安①表示","利確目安②表示","想定初期リスク%","想定利益%","RR","次回決算日","決算警告","評価コメント"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "出来高_20日平均比":st.column_config.NumberColumn("出来高倍率",format="%.2f倍"),
                    "75日線_比較期間前比%":st.column_config.NumberColumn("75日線傾き",format="%+.2f%%"),
                    "75日線_乖離率%":st.column_config.NumberColumn("75日線乖離",format="%+.2f%%"),
                }
            )

        st.subheader("🧪 v15｜過去類似シグナル成績")
        st.caption("短期スクリーニングランキング上位銘柄について、過去時点の株価・出来高だけで類似シグナルを再現します。財務(C)は将来情報混入を避けるため、この簡易検証には含めません。")
        bt_rows=[]
        for _, rr in tech.head(min(bt_top_n,len(tech))).iterrows():
            d_bt = data.get(rr["ticker"])
            res = backtest_current_ai_logic(
                d_bt, slope_days=slope_days, breakout_days=breakout_days,
                horizon=bt_horizon, target_pct=bt_target, stop_pct=bt_stop
            )
            if res:
                res.update({"順位":int(rr["順位"]), "銘柄":rr["銘柄"], "短期総合スコア":float(rr["短期総合スコア"])})
                bt_rows.append(res)
        if bt_rows:
            bt_df=pd.DataFrame(bt_rows).sort_values(f"+{bt_target:.0f}%到達率",ascending=False)
            st.dataframe(bt_df,use_container_width=True,hide_index=True)
            st.caption("これは銘柄ごとの過去類似シグナル集計で、将来の勝率ではありません。サンプル件数が少ない結果は特に慎重に見てください。")
        else:
            st.info("現在の取得期間では十分な過去シグナルがありません。株価をさかのぼる期間を1年または2年にすると検証件数が増えます。")

        st.subheader("🧪 v19.5｜逆指値の発動後指値バッファ検証")
        st.caption("『0.15×ATR・上限0.30%・最低2ティック』が本当に妥当かを、ランキング上位銘柄の過去ブレイク局面で比較します。固定の正解値ではなく、約定しやすさと買値悪化のトレードオフを見る検証です。")
        buffer_frames=[]
        for _, rr in tech.head(min(bt_top_n,len(tech))).iterrows():
            d_buf=data.get(rr["ticker"])
            bres=backtest_breakout_limit_buffer(d_buf,slope_days=slope_days,breakout_days=breakout_days)
            if bres is not None and not bres.empty:
                bres=bres.copy()
                bres["銘柄"]=rr["銘柄"]
                buffer_frames.append(bres)

        buffer_summary, gap_stats=summarize_buffer_backtests(buffer_frames)
        if buffer_summary is not None and gap_stats is not None:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("過去の発動件数",f"{gap_stats['発動件数']}件")
            c2.metric("必要幅 90%点",f"{gap_stats['必要幅90%点%']:.3f}%")
            c3.metric("必要幅 95%点",f"{gap_stats['必要幅95%点%']:.3f}%")
            c4.metric("参考推奨",str(gap_stats["参考推奨方式"]))

            display_buffer=buffer_summary.copy()
            display_buffer["現行"] = display_buffer["方式"].eq("現行 0.15ATR・上限0.30%").map({True:"← 現在使用",False:""})
            st.dataframe(
                display_buffer[["方式","現行","発動件数","寄付き即時約定率%","日中推定約定率%","平均許容幅円","平均許容幅%","指値超え寄付き率%"]],
                use_container_width=True,hide_index=True,
                column_config={
                    "寄付き即時約定率%":st.column_config.NumberColumn("寄付き即時約定率",format="%.1f%%"),
                    "日中推定約定率%":st.column_config.NumberColumn("日中推定約定率",format="%.1f%%"),
                    "平均許容幅円":st.column_config.NumberColumn("平均許容幅",format="%.1f円"),
                    "平均許容幅%":st.column_config.NumberColumn("平均許容幅%",format="%.3f%%"),
                    "指値超え寄付き率%":st.column_config.NumberColumn("指値を超えて寄付いた率",format="%.1f%%"),
                }
            )
            with st.expander("❓ この検証の読み方"):
                st.markdown(f"""
- **寄付き即時約定率**：発動日に寄付きから逆指値が発動した場合、設定した買い指値以内で始まった割合。
- **日中推定約定率**：寄付きで指値を飛び越えても、その日の安値が指値まで戻ったケースを含む推定値。
- **平均許容幅**：発動価格からどこまで高い買値を許す設定か。広いほど約定しやすい一方、高値掴みの許容も大きくなります。
- **指値を超えて寄付いた率**：発動日に株価が買い指値より高く始まった割合。
- **必要幅90%点 / 95%点**：過去の発動日の寄付きギャップの90% / 95%が収まった幅です。

**現在方式**：0.15×ATR（ATR＝普段1日の値動き幅）を基本に、発動価格の0.30%を上限、最低2ティック。  
**今回の参考推奨**：**{gap_stats["参考推奨方式"]}**

これは**日足OHLCからの推定**です。板情報や発動後の細かな約定順序は再現できないため、楽天証券での実際の約定率を保証するものではありません。
""")
            if gap_stats["発動件数"] < 20:
                st.warning("検証サンプルが20件未満です。参考値として見てください。株価取得期間を2年にするとサンプルが増えやすくなります。")
            elif str(gap_stats["参考推奨方式"]) != "現行 0.15ATR・上限0.30%":
                st.warning("今回の過去データでは、現在の0.15ATR方式より小さい許容幅でも95%以上の推定約定率を満たす方式があります。現行設定が最適とは限りません。")
            else:
                st.info("今回の過去データでは、現行0.15ATR方式が『推定約定率95%以上の中で許容幅が小さい方式』として残りました。ただし将来の最適性を保証しません。")
        else:
            st.info("バッファ検証に必要な過去のブレイク発動件数を確保できませんでした。取得期間を1年または2年にしてください。")

        tabs=st.tabs(["🔥 ブレイク準備中","🎯 押し目","🚀 ブレイク直後","💹 決算加速","📊 6要素の内訳"])

        tab_specs = [
            ("🔥 ブレイク準備中", lambda df: df[df["追加シグナル"].str.contains("🔥 ブレイク準備中", na=False)]),
            ("🎯 押し目", lambda df: df[df["追加シグナル"].str.contains("🎯 押し目", na=False)]),
            ("🚀 ブレイク直後", lambda df: df[df["追加シグナル"].str.contains("🚀 ブレイク直後", na=False)]),
            ("💹 決算加速", lambda df: df[(df["C"] == True) & (df["20日騰落率%"] > 0)]),
        ]
        for tab,(label,fn) in zip(tabs[:4],tab_specs):
            with tab:
                x=fn(tech)
                if x.empty:
                    st.info("該当なし")
                else:
                    st.dataframe(
                        x.head(50)[["順位","銘柄","Yahoo!チャート","短期総合スコア","上昇力","今の買いやすさ","セットアップ","追加シグナル","注文種類","注文価格表示","損切り価格表示","ルール評価"]],
                        use_container_width=True,hide_index=True,
                        column_config={"Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗")}
                    )
        with tabs[4]:
            st.dataframe(
                tech.head(100)[["順位","銘柄","トレンド","需給・出来高","モメンタム","業績","エントリー","リスク","75日線_乖離率%","出来高_20日平均比","20日騰落率%","60日騰落率%"]],
                use_container_width=True,hide_index=True
            )

        csv=tech.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 短期スクリーニングランキングをCSV保存",csv,f"短期スクリーニングランキング_{market_label}.csv","text/csv",use_container_width=True)

st.divider()
st.caption("v17は短期2モードに加え、🏦長期・年初来安値モードを追加しました。長期モードは安値接近度と企業クオリティを分けて評価し、年初来安値だからという理由だけでは買い判定しません。")