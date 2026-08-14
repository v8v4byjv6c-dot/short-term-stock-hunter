
import io
import re
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="短期上昇株ハンター v11",
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
    slope=(ma75/float(x.OLD)-1)*100
    dev=float(x.DEV); r5=float(x.R5); r20=float(x.R20); r60=float(x.R60)
    vr=float(x.VR) if np.isfinite(x.VR) else 0
    bh=float(x.BREAK_HIGH) if np.isfinite(x.BREAK_HIGH) else np.nan
    atr14=float(x.ATR14) if np.isfinite(x.ATR14) else np.nan

    trend = slope > 0
    # Aは参考書どおり「75日線より下」にある銘柄だけを候補にする。
    # そのうえで、75日線とのマイナス乖離が小さいほど高評価。
    below_75 = close < ma75
    near = -max_dev <= dev < 0.0
    crossed = np.isfinite(float(p.MA75)) and float(p.Close) <= float(p.MA75) and close > ma75
    bullish = close > float(p.Close) and close > float(x.Open)

    A = trend and below_75 and near
    dist = max(0,100-abs(dev)/max(max_dev,.1)*100)
    sl = min(100,max(0,slope/5*100))
    As = min(100,max(0,.50*dist+.35*sl+15*crossed+8*bullish))
    a_buy = ma75 + tick_size(ma75)*buy_ticks
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
        "株価":close, "75日線":ma75,
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
# UI
# ------------------------------------------------------------

st.title("🎯 短期上昇株ハンター v11")
st.write("同じURLの中で、**📘 本ベース A/B/C/D** と **🧪 AI式・独自統合スクリーナー**を切り替えられます。")

mode = st.radio(
    "分析モード",
    ["📘 本ベース A/B/C/D", "🧪 AI式・独自統合スクリーナー"],
    horizontal=True,
    help="AI式モードも外部AI APIは使いません。トレンド・出来高・モメンタム・業績・買い位置・リスクを統合する独自ルールです。"
)

with st.expander("ℹ️ 2つのモードの違い"):
    st.markdown("""
**📘 本ベース A/B/C/D**  
これまで育ててきたロジックです。Aは参考書の「75日線が上向き・株価は75日線より下・75日線上抜けで買う」を中心にしています。B/C/Dは補助戦略です。

**🧪 AI式・独自統合スクリーナー**  
本のルールには縛られず、短期上昇候補を6要素で採点します。

- トレンド 25%
- 需給・出来高 20%
- モメンタム 15%
- 業績 15%
- エントリー位置 15%
- リスク・過熱 10%

さらに **「上昇力」** と **「今の買いやすさ」** を別々に表示します。  
外部AI APIは使用しないため追加料金はかかりません。
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

try:
    all_u = get_jpx_universe()
except Exception as e:
    st.error(f"東証銘柄一覧を取得できませんでした：{e}")
    st.stop()

if "selected_market" not in st.session_state:
    st.session_state.selected_market = "プライム"
if "run_scan_v10" not in st.session_state:
    st.session_state.run_scan_v10 = False

st.subheader("🏢 スキャンする市場")
b1,b2,b3,b4,b5 = st.columns(5)
buttons = [
    (b1,"⭐ プライム","プライム"),
    (b2,"スタンダード","スタンダード"),
    (b3,"グロース","グロース"),
    (b4,"TOKYO PRO","TOKYO PRO"),
    (b5,"全市場","全市場"),
]
for col,label,value in buttons:
    with col:
        if st.button(label, type="primary" if st.session_state.selected_market==value else "secondary", use_container_width=True):
            st.session_state.selected_market=value
            st.session_state.run_scan_v10=True

market = st.session_state.selected_market
universe = select_market_universe(all_u, market)

m1,m2,m3 = st.columns(3)
m1.metric("選択中", market)
m2.metric("対象", f"{len(universe):,}銘柄")
m3.metric("モード", "本ベース" if mode.startswith("📘") else "AI式")

if st.button(f"🚀 {market}をスキャン", type="primary", use_container_width=True):
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

    # 楽天証券向けの注文コメント（AI式）
    current=float(r["株価"])
    risk_pct=((stop/trigger)-1)*100 if trigger and np.isfinite(stop) else np.nan
    chase = (dev > 20) or (np.isfinite(ext) and ext > 7) or ease < 45

    if chase:
        rakuten_action="🟠 今は注文を置かず押し待ち"
        rakuten_buy="現在値で追いかけず、買いやすさが改善するまで監視"
        rakuten_after="押し目形成後に買いトリガーを再計算"
    elif setup.startswith("🔥"):
        rakuten_action="🟢 逆指値・買い（ブレイク確認）"
        rakuten_buy=f"株価が{trigger:.0f}円以上になったら買い。条件到達後は指値を基本候補にする"
        rakuten_after=f"約定後は{stop:.0f}円以下を損切り逆指値の参考にする"
    elif setup.startswith("🚀"):
        rakuten_action="🟡 逆指値・買い（高値追いに注意）"
        rakuten_buy=f"株価が{trigger:.0f}円以上を維持する場合のみ買い候補。寄り付き急騰なら見送る"
        rakuten_after=f"約定後は{stop:.0f}円以下を損切り逆指値の参考にする"
    elif setup.startswith("🎯"):
        rakuten_action="🟢 逆指値・買い（反発確認）"
        rakuten_buy=f"75日線付近から反発し、株価が{trigger:.0f}円以上になったら買い候補"
        rakuten_after=f"約定後は{stop:.0f}円以下を損切り逆指値の参考にする"
    elif setup.startswith("💹"):
        rakuten_action="🟡 決算後の値動きを確認して逆指値"
        rakuten_buy=f"決算だけで成行買いせず、株価が{trigger:.0f}円以上で強さを確認して買い候補"
        rakuten_after=f"約定後は{stop:.0f}円以下を損切り逆指値の参考にする"
    else:
        rakuten_action="🟡 条件確認後に逆指値"
        rakuten_buy=f"株価が{trigger:.0f}円以上になり、出来高・トレンドが崩れていなければ買い候補"
        rakuten_after=f"約定後は{stop:.0f}円以下を損切り逆指値の参考にする"

    rakuten_risk = f"{risk_pct:.1f}%" if np.isfinite(risk_pct) else "算出不可"
    rakuten_comment = f"{rakuten_action}｜{rakuten_buy}｜{rakuten_after}｜想定初期リスク {rakuten_risk}"

    reasons=[]
    if trend>=75: reasons.append("上昇トレンドが強い")
    if volume>=70: reasons.append(f"出来高{vr:.2f}倍で資金流入")
    if momentum>=75: reasons.append("短中期モメンタムが強い")
    if earnings>=60: reasons.append("業績モメンタムも良好")
    if entry>=85: reasons.append(setup.replace("🎯 ","").replace("🔥 ","").replace("🚀 ",""))
    if risk<45: reasons.append("過熱・値動きリスクが大きい")
    comment="／".join(reasons[:4]) if reasons else "決定的な優位性はまだ弱い"

    return pd.Series({
        "AI式総合":total,
        "上昇力":strength,
        "今の買いやすさ":ease,
        "トレンド":trend,
        "需給・出来高":volume,
        "モメンタム":momentum,
        "業績":earnings,
        "エントリー":entry,
        "リスク":risk,
        "セットアップ":setup,
        "AI式診断":grade,
        "AI式コメント":comment,
        "買いトリガー":trigger,
        "損切り参考":stop,
        "楽天証券｜注文方針":rakuten_action,
        "楽天証券｜買い方":rakuten_buy,
        "楽天証券｜約定後":rakuten_after,
        "想定初期リスク%":risk_pct,
        "楽天証券｜注文コメント":rakuten_comment,
    })

if st.session_state.run_scan_v10:
    st.session_state.run_scan_v10=False
    if universe.empty:
        st.warning("対象銘柄を取得できませんでした。")
        st.stop()

    status=st.empty()
    progress=st.progress(0)
    status.info(f"① {market}の株価データを取得しています…")
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
                rows.append(r)
        if i%20==0:
            progress.progress(min(.68,(i+1)/max(total,1)*.68))

    if not rows:
        status.error("株価データを取得できませんでした。")
        st.stop()

    tech=pd.DataFrame(rows)

    # Decide which names get detailed financial lookup.
    if mode.startswith("📘"):
        tech["事前スコア"]=tech["Aスコア"]*.375+tech["Bスコア"]*.3125+tech["Dスコア"]*.3125
    else:
        tech["事前スコア"]=tech.apply(ai_pre_score,axis=1)

    tech=tech.sort_values("事前スコア",ascending=False).reset_index(drop=True)
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
        st.dataframe(
            ranked[["順位","銘柄","Yahoo!チャート","株価","75日線","75日線_乖離率%","A","B","C","D","該当戦略数","買い価格目安","A初期損切り目安","B初期損切り目安","総合スコア","総合診断"]],
            use_container_width=True,hide_index=True,
            column_config={
                "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗"),
                "株価":st.column_config.NumberColumn("株価",format="%.0f円"),
                "75日線":st.column_config.NumberColumn("75日線",format="%.0f円"),
                "75日線_乖離率%":st.column_config.NumberColumn("75日線乖離",format="%+.2f%%"),
                "買い価格目安":st.column_config.NumberColumn("買い目安",format="%.0f円"),
                "A初期損切り目安":st.column_config.NumberColumn("A損切り",format="%.0f円"),
                "B初期損切り目安":st.column_config.NumberColumn("B損切り",format="%.0f円"),
                "総合スコア":st.column_config.NumberColumn("総合",format="%.1f"),
            }
        )
        st.caption("A/B/C/Dの考え方はv9を維持しています。Aは75日線より下の銘柄だけが候補です。")

    else:
        status.success("AI式ランキングを作成しました。")
        ai = tech.apply(ai_scores,axis=1)
        tech=pd.concat([tech,ai],axis=1)
        tech=tech.sort_values(["AI式総合","今の買いやすさ","上昇力"],ascending=False).reset_index(drop=True)
        tech.insert(0,"順位",np.arange(1,len(tech)+1))

        st.subheader("🧪 AI式｜総合ランキング")
        st.caption("『上昇力』が高くても『今の買いやすさ』が低ければ、強いけれど高値追いすべきではない銘柄として分離します。")
        st.dataframe(
            tech.head(100)[["順位","銘柄","Yahoo!チャート","株価","AI式総合","上昇力","今の買いやすさ","セットアップ","AI式診断","買いトリガー","損切り参考","想定初期リスク%","楽天証券｜注文方針","AI式コメント"]],
            use_container_width=True,hide_index=True,
            column_config={
                "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗"),
                "株価":st.column_config.NumberColumn("株価",format="%.0f円"),
                "AI式総合":st.column_config.ProgressColumn("総合",min_value=0,max_value=100,format="%.1f"),
                "上昇力":st.column_config.ProgressColumn("上昇力",min_value=0,max_value=100,format="%.1f"),
                "今の買いやすさ":st.column_config.ProgressColumn("買いやすさ",min_value=0,max_value=100,format="%.1f"),
                "買いトリガー":st.column_config.NumberColumn("買いトリガー",format="%.0f円"),
                "損切り参考":st.column_config.NumberColumn("損切り参考",format="%.0f円"),
                "想定初期リスク%":st.column_config.NumberColumn("初期リスク",format="%.1f%%"),
                "楽天証券｜注文方針":st.column_config.TextColumn("楽天証券｜注文方針",width="large"),
                "AI式コメント":st.column_config.TextColumn("コメント",width="large"),
            }
        )

        st.subheader("📱 AI式｜楽天証券での買い方")
        st.caption("セットアップごとに、逆指値の使い方・買いトリガー・約定後の損切り参考値を文章化します。これは自動発注ではありません。")
        st.dataframe(
            tech.head(100)[["順位","銘柄","セットアップ","AI式診断","楽天証券｜注文方針","楽天証券｜買い方","楽天証券｜約定後","買いトリガー","損切り参考","想定初期リスク%","Yahoo!チャート"]],
            use_container_width=True, hide_index=True,
            column_config={
                "楽天証券｜注文方針":st.column_config.TextColumn("注文方針",width="large"),
                "楽天証券｜買い方":st.column_config.TextColumn("買い方",width="large"),
                "楽天証券｜約定後":st.column_config.TextColumn("約定後",width="large"),
                "買いトリガー":st.column_config.NumberColumn("買いトリガー",format="%.0f円"),
                "損切り参考":st.column_config.NumberColumn("損切り参考",format="%.0f円"),
                "想定初期リスク%":st.column_config.NumberColumn("初期リスク",format="%.1f%%"),
                "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗"),
            }
        )
        st.warning("逆指値はトリガー到達＝約定保証ではありません。指値は急騰時に約定しない可能性、成行は想定より不利な価格で約定する可能性があります。")

        tabs=st.tabs(["🔥 ブレイク準備中","🎯 押し目","🚀 ブレイク直後","💹 決算加速","📊 6要素の内訳"])
        setup_filters=[
            "🔥 ブレイク準備中","🎯 押し目・75日線接近","🚀 ブレイク直後","💹 決算加速"
        ]
        for tab,name in zip(tabs[:4],setup_filters):
            with tab:
                x=tech[tech["セットアップ"]==name]
                if x.empty: st.info("該当なし")
                else:
                    st.dataframe(x.head(50)[["順位","銘柄","Yahoo!チャート","AI式総合","上昇力","今の買いやすさ","買いトリガー","損切り参考","AI式診断"]],use_container_width=True,hide_index=True,
                        column_config={"Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo! ↗")})
        with tabs[4]:
            st.dataframe(
                tech.head(100)[["順位","銘柄","トレンド","需給・出来高","モメンタム","業績","エントリー","リスク","75日線_乖離率%","出来高_20日平均比","20日騰落率%","60日騰落率%"]],
                use_container_width=True,hide_index=True
            )

        csv=tech.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 AI式ランキングをCSV保存",csv,f"AI式ランキング_{market}.csv","text/csv",use_container_width=True)

st.divider()
st.caption("v11は1つのStreamlit URL・1つのGitHubリポジトリで2モードを切り替え、AI式にも楽天証券向け注文コメントを表示します。AI式モードは生成AIではなく独自統合ルールです。投資判断・将来の値上がりを保証しません。")
