
import io
import re
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="短期上昇株ハンター v6",
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

    x, p = d.iloc[-1], d.iloc[-2]
    vals = [x.Close,x.High,x.MA25,x.MA75,x.OLD,x.DEV,x.R5,x.R20,x.R60]
    if not all(np.isfinite(float(v)) for v in vals):
        return None

    close=float(x.Close); high=float(x.High); ma25=float(x.MA25); ma75=float(x.MA75)
    slope=(ma75/float(x.OLD)-1)*100
    dev=float(x.DEV); r5=float(x.R5); r20=float(x.R20); r60=float(x.R60)
    vr=float(x.VR) if np.isfinite(x.VR) else 0
    bh=float(x.BREAK_HIGH) if np.isfinite(x.BREAK_HIGH) else np.nan

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

    Bs = min(
        100,
        max(
            0,
            min(45,max(0,r20*2))
            + min(30,max(0,(vr-1)*30))
            + (25 if B else 0)
            - extension_penalty
        )
    )

    recent = float(d.Close.tail(20).max())
    drawdown = (close/recent-1)*100
    D = bool(trend and r60>=10 and drawdown<=-3 and r5>0 and close>ma25)
    Ds = min(100,max(0,min(45,max(0,r60*1.5))+min(25,max(0,r5*4))+(30 if D else 0)))
    bd_buy = high + tick_size(high)

    return {
        "株価":close, "75日線":ma75,
        "75日線_比較期間前比%":slope, "75日線_乖離率%":dev,
        "出来高_20日平均比":vr, "20日騰落率%":r20, "60日騰落率%":r60,
        "ブレイク水準からの上昇率%":breakout_extension,
        "A":A, "Aスコア":As, "A買い価格":a_buy, "A初期損切り":a_stop,
        "B":B, "Bスコア":Bs, "B買い価格":bd_buy,
        "Bブレイク水準":bh,
        "Bブレイク上昇率%":breakout_extension,
        "B出来高倍率":vr,
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
st.title("🎯 短期上昇株ハンター v6")
st.write("市場を選ぶだけで対象銘柄を自動取得し、A〜Dで短期上昇候補をランキングします。**普段使いはプライム推奨**です。")

with st.expander("📘 A・B・C・Dとは？ ランキングの仕組み"):
    st.markdown("""
**A｜75日線押し目**  
75日線が上向いており、**現在の株価が75日線より下**にある銘柄だけを候補にします。その中で75日線とのマイナス乖離が小さい銘柄を高く評価します。まだ買わず、75日線を上抜けた価格を買いトリガーにし、買った後に75日線を再び明確に割る水準を初期損切り目安とします。

**B｜ブレイクアウト**  
75日線が上向きで、過去一定期間の高値を更新し、出来高も増えている銘柄を評価します。ただし、ブレイク水準からすでに大きく上がりすぎている場合は「追いかけ買い」リスクとしてスコアを減点します。

**C｜決算モメンタム**  
四半期営業利益が前年同期比+20%以上で、売上高が取得できる場合は減収でない銘柄を評価。Yahoo Financeで取得可能な四半期財務による暫定ロジックです。

**D｜モメンタム＋押し目**  
直近60日で強く上昇したあと一度調整し、短期的に再上昇している銘柄を評価します。

**🏆 総合ランキング**  
A 30%・B 25%・C 20%・D 25%を合成し、複数戦略に同時該当するほど加点します。
""")

with st.sidebar:
    st.header("設定")
    pl = st.selectbox("株価をさかのぼる期間",["6か月","1年","2年"],index=1)
    period = {"6か月":"6mo","1年":"1y","2年":"2y"}[pl]
    slope_days = st.slider("75日線が上向きかを判断する期間",5,40,20,1,
                           help="20なら現在の75日線が20営業日前より高いかを判定します。")
    max_dev = st.slider("A：75日線から下に離れてよい範囲",1.0,10.0,5.0,0.5)
    buy_ticks = st.slider("A：75日線を何ティック上抜けたら買い候補か",1,5,2,1)
    a_stop_buffer_pct = st.slider("A：75日線割れの損切りバッファ",0.10,2.00,0.35,0.05,format="%.2f%%")
    breakout_days = st.slider("B：何日間の高値更新を見るか",20,120,60,10)
    c_check_count = st.slider("C：決算を詳しく確認する上位銘柄数",30,200,100,10)

try:
    all_u = get_jpx_universe()
except Exception as e:
    st.error(f"東証銘柄一覧を取得できませんでした：{e}")
    st.stop()

if "selected_market" not in st.session_state:
    st.session_state.selected_market = "プライム"
if "run_scan" not in st.session_state:
    st.session_state.run_scan = False

st.subheader("🏢 スキャンする市場")
st.caption("ボタンを押すと、その市場だけを取得してすぐランキングを開始します。全市場はプライム＋スタンダード＋グロースです。")

b1,b2,b3,b4,b5 = st.columns(5)
with b1:
    if st.button("⭐ プライム", type="primary" if st.session_state.selected_market=="プライム" else "secondary", use_container_width=True):
        st.session_state.selected_market="プライム"; st.session_state.run_scan=True
with b2:
    if st.button("スタンダード", type="primary" if st.session_state.selected_market=="スタンダード" else "secondary", use_container_width=True):
        st.session_state.selected_market="スタンダード"; st.session_state.run_scan=True
with b3:
    if st.button("グロース", type="primary" if st.session_state.selected_market=="グロース" else "secondary", use_container_width=True):
        st.session_state.selected_market="グロース"; st.session_state.run_scan=True
with b4:
    if st.button("TOKYO PRO", type="primary" if st.session_state.selected_market=="TOKYO PRO" else "secondary", use_container_width=True):
        st.session_state.selected_market="TOKYO PRO"; st.session_state.run_scan=True
with b5:
    if st.button("全市場", type="primary" if st.session_state.selected_market=="全市場" else "secondary", use_container_width=True):
        st.session_state.selected_market="全市場"; st.session_state.run_scan=True

market = st.session_state.selected_market
universe = select_market_universe(all_u, market)

m1,m2,m3 = st.columns(3)
m1.metric("選択中", market)
m2.metric("自動スキャン対象", f"{len(universe):,}銘柄")
m3.metric("銘柄入力", "不要")

if market == "TOKYO PRO":
    st.warning("TOKYO PRO MarketはYahoo Financeで株価・財務データを取得できない銘柄が多い可能性があります。その場合はランキング対象から自動的に外れます。")
elif market == "プライム":
    st.success("プライムだけを対象にするため、全市場スキャンより取得・計算量を減らせます。")

# もう一度同じ市場を走らせたい場合
if st.button(f"🔄 {market}を再スキャン", use_container_width=True):
    st.session_state.run_scan = True

if st.session_state.run_scan:
    st.session_state.run_scan = False

    if universe.empty:
        st.warning(f"{market}の対象銘柄を取得できませんでした。")
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    status.info(f"① {market}の株価データ（{len(universe):,}銘柄）を取得しています…")
    data = download_batch(universe.ticker.tolist(), period)

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
        if i % 20 == 0:
            progress.progress(min(.70,(i+1)/max(total,1)*.70))

    if not rows:
        status.error("株価データを取得できませんでした。Yahoo Finance側で取得できない市場・銘柄、または一時的な制限の可能性があります。")
        st.stop()

    tech=pd.DataFrame(rows)
    tech["テクニカル仮スコア"]=tech.Aスコア*.375+tech.Bスコア*.3125+tech.Dスコア*.3125
    tech=tech.sort_values("テクニカル仮スコア",ascending=False).reset_index(drop=True)

    status.info("② 上位候補の決算モメンタム（C）を確認しています…")
    n=min(c_check_count,len(tech)); cmap={}
    for j,t in enumerate(tech.head(n).ticker):
        cmap[t]=financial_momentum(t)
        progress.progress(.70+(j+1)/max(n,1)*.30)

    progress.empty()
    status.success(f"{market}ランキングを作成しました。")

    tech["C"]=[cmap.get(t,{}).get("C",False) for t in tech.ticker]
    tech["Cスコア"]=[cmap.get(t,{}).get("Cスコア",0) for t in tech.ticker]
    tech["営業利益_前年同期比%"]=[cmap.get(t,{}).get("営業利益_前年同期比%",np.nan) for t in tech.ticker]
    tech["売上高_前年同期比%"]=[cmap.get(t,{}).get("売上高_前年同期比%",np.nan) for t in tech.ticker]
    tech["該当戦略数"]=tech[["A","B","C","D"]].sum(axis=1)
    tech["総合スコア"]=(tech.Aスコア*.30+tech.Bスコア*.25+tech.Cスコア*.20+tech.Dスコア*.25+tech["該当戦略数"]*5).clip(upper=120)

    def buy(r):
        if r.A:return r["A買い価格"]
        if r.B:return r["B買い価格"]
        if r.D:return r["D買い価格"]
        return np.nan

    tech["買い価格目安"]=tech.apply(buy,axis=1)
    tech["A初期損切り目安"]=np.where(tech.A,tech["A初期損切り"],np.nan)

    def heat_label(ext):
        if not np.isfinite(ext): return "－"
        if ext <= 3: return "🟢 ブレイク直後"
        if ext <= 7: return "🟡 やや上昇済み"
        return "🔴 上がりすぎ注意"

    def a_reason(r):
        if not bool(r["A"]): return "A条件外"
        return (
            f"75日線は上向き（{r['75日線_比較期間前比%']:+.2f}%）／"
            f"株価は75日線より下（乖離{r['75日線_乖離率%']:+.2f}%）／"
            f"75日線に近いほど高評価"
        )

    def b_reason(r):
        if not bool(r["B"]): return "B条件外"
        ext = r["Bブレイク上昇率%"]
        return (
            f"{int(breakout_days)}日高値 {r['Bブレイク水準']:,.0f}円を突破／"
            f"現在はブレイク水準から{ext:+.2f}%／"
            f"出来高は20日平均の{r['B出来高倍率']:.2f}倍／"
            f"20日騰落率{r['20日騰落率%']:+.2f}%／{heat_label(ext)}"
        )

    def c_reason(r):
        if not bool(r["C"]): return "C条件外または財務データ不足"
        og=r["営業利益_前年同期比%"]; rg=r["売上高_前年同期比%"]
        ogs="-" if not np.isfinite(og) else f"{og:+.1f}%"
        rgs="-" if not np.isfinite(rg) else f"{rg:+.1f}%"
        return f"営業利益前年同期比 {ogs}／売上高前年同期比 {rgs}"

    def d_reason(r):
        if not bool(r["D"]): return "D条件外"
        return (
            f"60日騰落率{r['60日騰落率%']:+.2f}%／"
            f"強い上昇後の押し目から短期反発"
        )

    tech["A評価理由"]=tech.apply(a_reason,axis=1)
    tech["B評価理由"]=tech.apply(b_reason,axis=1)
    tech["C評価理由"]=tech.apply(c_reason,axis=1)
    tech["D評価理由"]=tech.apply(d_reason,axis=1)
    tech["B過熱度"]=tech["Bブレイク上昇率%"].apply(heat_label)

    tech["該当戦略"]=tech.apply(lambda r:"・".join([s for s in ["A","B","C","D"] if bool(r[s])]) or "－",axis=1)
    tech["判定"]=tech["該当戦略数"].map(lambda n:"🔥 最優先候補" if n>=3 else "🟢 強候補" if n==2 else "🟡 候補" if n==1 else "⚪ 見送り")
    tech=tech.sort_values(["総合スコア","該当戦略数","Aスコア","Dスコア"],ascending=False).reset_index(drop=True)
    tech.insert(0,"順位",np.arange(1,len(tech)+1))
    for s in ["A","B","C","D"]:
        tech[s]=tech[s].map(mark)

    ranked=tech[tech["該当戦略数"]>=1].copy()

    st.subheader(f"🏆 {market}｜今日の短期上昇候補ランキング")
    if ranked.empty:
        st.info("現在の条件ではA〜Dに該当する銘柄がありません。")
    else:
        cols=["順位","銘柄","Yahoo!チャート","株価","75日線","75日線_比較期間前比%","75日線_乖離率%","A","B","C","D","該当戦略数","買い価格目安","A初期損切り目安","総合スコア","判定"]
        st.dataframe(
            ranked[cols],
            use_container_width=True,hide_index=True,
            column_config={
                "銘柄":st.column_config.TextColumn("銘柄コード・銘柄名",width="large"),
                "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo!チャート ↗"),
                "株価":st.column_config.NumberColumn("株価",format="%.0f円"),
                "75日線":st.column_config.NumberColumn("75日線",format="%.0f円"),
                "75日線_比較期間前比%":st.column_config.NumberColumn(f"75日線（{slope_days}営業日前比）",format="%+.2f%%"),
                "75日線_乖離率%":st.column_config.NumberColumn("75日線乖離",format="%+.2f%%"),
                "買い価格目安":st.column_config.NumberColumn("買い価格目安",format="%.0f円"),
                "A初期損切り目安":st.column_config.NumberColumn("A初期損切り",format="%.0f円"),
                "総合スコア":st.column_config.NumberColumn("総合",format="%.1f"),
            }
        )

    st.subheader("🧭 なぜこの評価？")
    st.caption("各銘柄のスコア根拠を数値で確認できます。特にBは『高値を突破したか』だけでなく、ブレイク後に上がりすぎていないかも表示します。")
    reason_cols=["順位","銘柄","A評価理由","B評価理由","B過熱度","C評価理由","D評価理由"]
    st.dataframe(
        tech[tech["該当戦略数"]>=1][reason_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "銘柄":st.column_config.TextColumn("銘柄",width="medium"),
            "A評価理由":st.column_config.TextColumn("A｜評価理由",width="large"),
            "B評価理由":st.column_config.TextColumn("B｜評価理由",width="large"),
            "B過熱度":st.column_config.TextColumn("B｜過熱度",width="medium"),
            "C評価理由":st.column_config.TextColumn("C｜評価理由",width="large"),
            "D評価理由":st.column_config.TextColumn("D｜評価理由",width="large"),
        }
    )

    st.subheader("🔎 戦略別ランキング")
    tabs=st.tabs(["A｜75日線押し目","B｜ブレイクアウト","C｜決算モメンタム","D｜モメンタム＋押し目"])
    for tab,s in zip(tabs,["A","B","C","D"]):
        with tab:
            x=tech[tech[s]=="🟢"]
            if x.empty:
                st.info("該当銘柄はありません。")
            else:
                st.dataframe(
                    x[["順位","銘柄","Yahoo!チャート","株価","75日線","買い価格目安","A初期損切り目安","B過熱度","総合スコア","該当戦略"]],
                    use_container_width=True,hide_index=True,
                    column_config={
                        "Yahoo!チャート":st.column_config.LinkColumn("チャート",display_text="Yahoo!チャート ↗"),
                        "株価":st.column_config.NumberColumn("株価",format="%.0f円"),
                        "75日線":st.column_config.NumberColumn("75日線",format="%.0f円"),
                        "買い価格目安":st.column_config.NumberColumn("買い価格目安",format="%.0f円"),
                        "A初期損切り目安":st.column_config.NumberColumn("A初期損切り",format="%.0f円"),
                        "総合スコア":st.column_config.NumberColumn("総合",format="%.1f"),
                    }
                )

    with st.expander("📊 詳細な計算値を見る"):
        st.dataframe(
            tech[["順位","銘柄","75日線_比較期間前比%","75日線_乖離率%","出来高_20日平均比","20日騰落率%","60日騰落率%","ブレイク水準からの上昇率%","営業利益_前年同期比%","売上高_前年同期比%","Aスコア","Bスコア","Cスコア","Dスコア","総合スコア"]],
            use_container_width=True,hide_index=True
        )

    csvcols=["順位","コード","銘柄名","市場","業種","株価","75日線","75日線_比較期間前比%","75日線_乖離率%","A","B","C","D","該当戦略","買い価格目安","A初期損切り目安","営業利益_前年同期比%","売上高_前年同期比%","A評価理由","B評価理由","B過熱度","C評価理由","D評価理由","総合スコア","判定","Yahoo!チャート"]
    csv=tech[csvcols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        f"📥 {market}ランキングをCSV保存",
        csv,
        f"短期上昇株ハンター_{market}.csv",
        "text/csv",
        use_container_width=True
    )

st.divider()
st.caption("本アプリはユーザー指定ルールによるスクリーニング補助です。データには遅延・欠損・取得制限があり得ます。将来の値上がりを保証するものではありません。")
