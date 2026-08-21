
import io
import re
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="短期上昇株ハンター v17",
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
@st.cache_data(ttl=900, show_spinner=False)
def download_batch(tickers, period):
    out = {}
    tickers = list(tickers)
    for s in range(0, len(tickers), 100):
        chunk = tickers[s:s+100]
        try:
            d = yf.download(
                chunk,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if d is None or d.empty:
                continue
            if len(chunk) == 1 and not isinstance(d.columns, pd.MultiIndex):
                out[chunk[0]] = d.copy()
            elif isinstance(d.columns, pd.MultiIndex):
                l0 = set(d.columns.get_level_values(0))
                l1 = set(d.columns.get_level_values(1))
                if any(t in l0 for t in chunk):
                    for t in chunk:
                        if t in l0:
                            x = d[t].dropna(how="all")
                            if not x.empty: out[t] = x
                else:
                    for t in chunk:
                        if t in l1:
                            x = d.xs(t, axis=1, level=1).dropna(how="all")
                            if not x.empty: out[t] = x
        except Exception:
            pass
    return out

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
    # Bの初期損切り参考値：ブレイク水準 - 1ATR(14)
    # 値動きが大きい銘柄ほど余裕を持たせる。
    b_stop = (bh - atr14) if np.isfinite(bh) and np.isfinite(atr14) else np.nan

    return {
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
# UI
# ------------------------------------------------------------


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_regime():
    """TOPIXと日経平均のトレンドから、短期売買の地合いを簡易判定する。"""
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
        label, comment = "🟢 強気", "個別株の買いシグナルを通常どおり評価しやすい地合いです。"
    elif score >= 60:
        label, comment = "🟡 やや強気", "買い候補は選別しつつ、通常サイズを検討できる地合いです。"
    elif score >= 40:
        label, comment = "🟠 中立", "新規買いはトリガー確認を重視し、追いかけ買いを避けたい地合いです。"
    else:
        label, comment = "🔴 弱気", "新規買いを厳選し、ポジションを小さめにすることを検討したい地合いです。"
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

st.title("🎯 短期上昇株ハンター v17")
st.write("同じURLの中で、**📘 本ベース A/B/C/D** と **🧪 独自短期・独自統合スクリーナー**を切り替えられます。")

mode = st.radio(
    "分析モード",
    ["📘 本ベース A/B/C/D", "🧪 独自短期・独自統合スクリーナー", "🏦 長期・年初来安値"],
    horizontal=True,
    help="短期2モードと、年初来安値付近の優良株を探す長期モードを同じURLで切り替えます。"
)

with st.expander("ℹ️ 3つのモードの違い"):
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
""")

with st.sidebar:
    st.header("共通設定")
    pl = st.selectbox("株価をさかのぼる期間",["6か月","1年","2年"],index=1)
    period = {"6か月":"6mo","1年":"1y","2年":"2y"}[pl]
    slope_days = st.slider(
        "75日線が上向きかを判断する期間", 5, 40, 20, 1,
        help="20なら、現在の75日線が20営業日前より高いかを判定します。"
    )
    max_dev = st.slider("A：75日線から下に離れてよい範囲",1.0,10.0,5.0,0.5)
    buy_ticks = st.slider("A：75日線を何ティック上抜けたら買い候補か",1,5,2,1)
    a_stop_buffer_pct = st.slider("A：75日線割れの損切りバッファ",0.10,2.00,0.35,0.05,format="%.2f%%")
    breakout_days = st.slider("高値更新を見る期間",20,120,60,10)
    c_check_count = st.slider("決算を詳しく確認する上位銘柄数",30,200,100,10)
    st.divider()
    st.subheader("v13 検証設定")
    bt_top_n = st.slider("バックテストする上位銘柄数",5,30,10,5, help="処理時間を抑えるため、短期スクリーニングランキング上位だけを検証します。")
    bt_horizon = st.selectbox("何営業日先まで検証するか",[5,10,20],index=0)
    bt_target = st.selectbox("上昇目標",[3.0,5.0,8.0,10.0],index=1,format_func=lambda x:f"+{x:.0f}%")
    bt_stop = st.selectbox("下落警戒ライン",[2.0,3.0,5.0],index=1,format_func=lambda x:f"-{x:.0f}%")

    if mode.startswith("🏦"):
        st.divider()
        st.subheader("長期・年初来安値設定")
        long_fund_n = st.slider(
            "ファンダメンタルを確認する上位銘柄数",
            20, 150, 80, 10,
            help="年初来安値に近い順に候補を絞り、上位だけ時価総額・ROE・配当等を取得します。増やすほど時間がかかります。"
        )
        long_mcap_filter = st.selectbox(
            "大手企業フィルター",
            ["制限なし","1,000億円以上","5,000億円以上","1兆円以上"],
            index=2,
            help="長期モードの『大手』タブや総合候補に使います。"
        )
        long_max_low_dist = st.slider(
            "年初来安値から何%以内を重点表示するか",
            1, 30, 10, 1,
            help="10%なら、現在値が年初来安値から+10%以内の銘柄を重点候補にします。"
        )

try:
    all_u = get_jpx_universe()
except Exception as e:
    st.error(f"東証銘柄一覧を取得できませんでした：{e}")
    st.stop()

if "selected_markets_v12" not in st.session_state:
    st.session_state.selected_markets_v12 = ["プライム"]
if "run_scan_v10" not in st.session_state:
    st.session_state.run_scan_v10 = False

st.subheader("🏢 スキャンする市場")
st.caption("複数選択できます。例：プライム＋スタンダード。初期設定はプライムのみです。")

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
    st.info(f"**短期地合い（長期では参考）：{regime['label']}**  長期モードでは地合い点より、企業の質と年初来安値への接近を重視します。")
else:
    st.info(f"**市場地合い：{regime['label']}（{regime['score']:.0f}/100）**  {regime['comment']}")
with st.expander("地合い判定の内訳"):
    if regime["details"]:
        st.dataframe(pd.DataFrame(regime["details"]), use_container_width=True, hide_index=True)
    else:
        st.write("指数データを取得できませんでした。")

if not selected_markets:
    st.warning("市場を1つ以上選択してください。")

if st.button(f"🚀 {market_label}をスキャン", type="primary", use_container_width=True, disabled=not selected_markets):
    st.session_state.run_scan_v10=True

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

    if chase:
        order_reason="過熱または買いやすさ不足。現在値を追いかけず、押し目形成後に再判定。"

    elif setup.startswith("🔥"):
        # ブレイク前：上抜けを確認してから買うため「買い逆指値」
        buy_order_type="買い逆指値"
        buy_price=float(trigger)
        buy_price_text=f"{buy_price:.0f}円以上"
        buy_condition=f"株価が{buy_price:.0f}円以上になったら買い条件発動"
        stop_order_type="売り逆指値"
        stop_price=float(stop)
        order_reason="まだブレイク前。上抜けを確認してから入る。"

    elif setup.startswith("🚀"):
        # ブレイク済み：すでに上抜けているため、さらに上で買う逆指値は使わず押し待ちの指値。
        breakout=float(r["Bブレイク水準"]) if np.isfinite(r["Bブレイク水準"]) else current
        pullback_low=max(breakout, current-0.75*atr_yen)
        pullback_high=current-tick_size(current)
        if pullback_low > pullback_high:
            pullback_low=pullback_high
        buy_order_type="買い指値"
        buy_price=float(pullback_high)
        buy_price_text=f"{pullback_low:.0f}〜{pullback_high:.0f}円"
        buy_condition=f"{buy_price_text}への押しを待つ。上に飛んだ場合は追いかけない"
        stop_order_type="売り逆指値"
        stop_price=float(stop)
        order_reason="ブレイク済み。新規の買い逆指値ではなく、押しを待つ指値買い。"

    elif setup.startswith("🎯"):
        # 75日線押し目：下で待つので買い指値
        buy_order_type="買い指値"
        buy_price=float(r["A買い価格"])
        buy_price_text=f"{buy_price:.0f}円"
        buy_condition=f"{buy_price:.0f}円前後への押しを待つ"
        stop_order_type="売り逆指値"
        stop_price=float(r["A初期損切り"])
        order_reason="75日線付近の押し目を待って買う。"

    elif setup.startswith("💹"):
        # 決算加速だけでは注文方法を決め打ちしない。
        buy_order_type="条件確認後"
        buy_price=np.nan
        buy_price_text="—"
        buy_condition="決算加速だけで注文せず、押し目またはブレイク条件が追加で出るまで待つ"
        stop_order_type="—"
        stop_price=np.nan
        order_reason="決算の良さだけを理由に注文種類を決めない。"

    elif setup.startswith("📈"):
        # モメンタム継続は高値追いを避け、明確な新トリガーが出るまで待つ。
        buy_order_type="注文しない"
        buy_price=np.nan
        buy_price_text="—"
        buy_condition="押し目または新しいブレイク準備シグナルを待つ"
        stop_order_type="—"
        stop_price=np.nan
        order_reason="モメンタムだけで高値を追わない。"

    else:
        buy_order_type="注文しない"
        buy_condition="監視継続"
        order_reason="注文方法を一意に決められるセットアップではない。"

    risk_base = buy_price if np.isfinite(buy_price) else np.nan
    risk_pct=((stop_price/risk_base)-1)*100 if np.isfinite(risk_base) and np.isfinite(stop_price) and risk_base else np.nan
    stop_price_text=f"{stop_price:.0f}円以下" if np.isfinite(stop_price) else "—"
    order_summary=f"{buy_order_type}｜{buy_price_text}｜約定後 {stop_order_type} {stop_price_text}"
    take_profit1=take_profit2=rr=reward_pct=np.nan
    if np.isfinite(buy_price) and np.isfinite(stop_price) and buy_price > stop_price:
        risk_yen=buy_price-stop_price
        take_profit1=buy_price+2*risk_yen
        take_profit2=buy_price+3*risk_yen
        rr=2.0
        reward_pct=(take_profit1/buy_price-1)*100
    take_profit1_text=f"{take_profit1:.0f}円" if np.isfinite(take_profit1) else "—"
    take_profit2_text=f"{take_profit2:.0f}円" if np.isfinite(take_profit2) else "—"
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
        "追加シグナル":"・".join(signals),
        "ルール評価":grade,
        "評価コメント":comment,
        "注文種類":buy_order_type,
        "注文価格":buy_price,
        "注文価格表示":buy_price_text,
        "注文条件":buy_condition,
        "損切り注文":stop_order_type,
        "損切り価格":stop_price,
        "損切り価格表示":stop_price_text,
        "想定初期リスク%":risk_pct,
        "注文理由":order_reason,
        "注文サマリー":order_summary,
        "利確目安①":take_profit1,"利確目安②":take_profit2,
        "利確目安①表示":take_profit1_text,"利確目安②表示":take_profit2_text,
        "想定利益%":reward_pct,"RR":rr,"実戦優先度":practical_priority,
    })

if st.session_state.run_scan_v10:
    st.session_state.run_scan_v10=False
    if universe.empty:
        st.warning("対象銘柄を取得できませんでした。")
        st.stop()

    status=st.empty()
    progress=st.progress(0)
    status.info(f"① {market_label}の株価データを取得しています…")
    data=download_batch(universe.ticker.tolist(),period)
    rows=[]; total=len(universe)

    for i,row in universe.reset_index(drop=True).iterrows():
        d=data.get(row.ticker)
        if d is not None:
            r=technical_scan(d,slope_days,max_dev,breakout_days,a_stop_buffer_pct,buy_ticks)
            if r:
                r.update({
                    "ticker":row.ticker,"コード":row["コード"],"銘柄名":row["銘柄名"],
                    "銘柄":row["銘柄"],"市場":row["市場"],"業種":row["業種"],
                    "Yahoo!チャート":row["Yahoo!チャート"]
                })
                lm = long_price_metrics(d)
                if lm:
                    r.update(lm)
                rows.append(r)
        if i%20==0:
            progress.progress(min(.68,(i+1)/max(total,1)*.68))

    if not rows:
        status.error("株価データを取得できませんでした。")
        st.stop()

    tech=pd.DataFrame(rows)

    # モード別の事前順位。
    if mode.startswith("📘"):
        tech["事前スコア"]=tech["Aスコア"]*.375+tech["Bスコア"]*.3125+tech["Dスコア"]*.3125
    elif mode.startswith("🏦"):
        # 長期ではまず年初来安値への接近を優先。
        tech["事前スコア"]=tech["安値接近度"].fillna(0)
    else:
        tech["事前スコア"]=tech.apply(ai_pre_score,axis=1)

    tech=tech.sort_values("事前スコア",ascending=False).reset_index(drop=True)

    # 長期モードは短期C判定をせず、上位候補のファンダメンタルを取得してここで完結。
    if mode.startswith("🏦"):
        status.info("② 年初来安値に近い候補の企業情報を確認しています…")
        fund_n=min(long_fund_n, len(tech))
        f_map={}
        for j,t in enumerate(tech.head(fund_n).ticker):
            f_map[t]=long_term_fundamentals(t)
            progress.progress(.68+(j+1)/max(fund_n,1)*.32)
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

    status.info("② 上位候補の決算データを確認しています…")
    n=min(c_check_count,len(tech)); cmap={}
    for j,t in enumerate(tech.head(n).ticker):
        cmap[t]=financial_momentum(t)
        progress.progress(.68+(j+1)/max(n,1)*.32)
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
        earnings_infos = {}
        for t in tech.head(min(30,len(tech))).ticker:
            earnings_infos[t] = get_earnings_date_info(t)
        tech["次回決算日"] = [earnings_infos.get(t,{}).get("date") for t in tech.ticker]
        tech["決算まで日数"] = [earnings_infos.get(t,{}).get("days") for t in tech.ticker]
        tech["決算警告"] = [earnings_infos.get(t,{}).get("warning","") for t in tech.ticker]

        # 地合いを診断コメントへ反映
        if regime["score"] < 40:
            tech["評価コメント"] = tech["評価コメント"].astype(str) + "／地合い弱気のため新規買いは厳選"
        elif regime["score"] < 60:
            tech["評価コメント"] = tech["評価コメント"].astype(str) + "／地合い中立"

        st.subheader("📊 実戦ランキング")
        st.caption("現在値 → 前日比 → 評価 → 楽天証券の買い注文 → 損切り → 利確 → RR。前日比は最新日足と1本前の日足の比較で、リアルタイム配信値ではありません。")
        st.dataframe(tech.head(100)[["順位","実戦優先度","銘柄","Yahoo!チャート","株価","前日比","前日比%","短期総合スコア","セットアップ","ルール評価","注文種類","注文価格表示","損切り価格表示","利確目安①表示","RR","決算警告"]],use_container_width=True,hide_index=True,
            column_config={"Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗"),"株価":st.column_config.NumberColumn("現在値",format="%.0f円"),"前日比":st.column_config.NumberColumn("前日比",format="%+.0f円"),"前日比%":st.column_config.NumberColumn("前日比%",format="%+.2f%%"),"短期総合スコア":st.column_config.ProgressColumn("短期スコア",min_value=0,max_value=100,format="%.1f"),"RR":st.column_config.NumberColumn("RR",format="%.2f")})
        st.warning("買い逆指値＝上抜け確認、買い指値＝押し待ち、売り逆指値＝約定後の損切り。株価データはリアルタイム保証ではありません。")
        with st.expander("🔎 詳細を見る"):
            st.dataframe(tech.head(100)[["順位","銘柄","始値","高値","安値","前日終値","前日比%","上昇力","今の買いやすさ","出来高倍率","75日線","75日線傾き%","75日線乖離%","追加シグナル","注文条件","注文理由","損切り注文","損切り価格表示","利確目安①表示","利確目安②表示","想定初期リスク%","想定利益%","RR","次回決算日","決算警告","評価コメント"]],use_container_width=True,hide_index=True)

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