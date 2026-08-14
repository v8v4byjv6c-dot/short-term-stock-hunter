
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="短期上昇株ハンター", page_icon="🎯", layout="wide")

# ---------- utilities ----------
def tick_size(p):
    if p < 3000: return 1
    if p < 5000: return 5
    if p < 10000: return 10
    if p < 30000: return 30
    if p < 50000: return 50
    if p < 100000: return 100
    return 100

def yen(x):
    return "-" if pd.isna(x) else f"{x:,.0f}円"

@st.cache_data(ttl=900, show_spinner=False)
def prices(ticker, period="1y"):
    d=yf.download(ticker, period=period, interval="1d", auto_adjust=False,
                  progress=False, threads=False)
    if d is None or d.empty: return pd.DataFrame()
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    cols=[c for c in ["Open","High","Low","Close","Volume"] if c in d.columns]
    return d[cols].dropna()

@st.cache_data(ttl=1800, show_spinner=False)
def fundamentals(ticker):
    # Yahoo Financeの四半期データ。取得できない場合はCを「判定不能」にする。
    try:
        y=yf.Ticker(ticker)
        q=y.quarterly_financials
        if q is None or q.empty: return {}
        # 行名はYahoo側の表示変更に備えて候補を探す
        def row(names):
            for n in names:
                if n in q.index: return q.loc[n]
            return None
        rev=row(["Total Revenue","Operating Revenue"])
        op=row(["Operating Income"])
        if rev is None or op is None or len(rev)<2 or len(op)<2: return {}
        rev=pd.to_numeric(rev,errors="coerce").dropna()
        op=pd.to_numeric(op,errors="coerce").dropna()
        if len(rev)<2 or len(op)<2: return {}
        # 最新四半期 vs 前四半期ではなく、取得できる範囲で直前期比を参考値として使う
        r_growth=(float(rev.iloc[0])/float(rev.iloc[1])-1)*100 if rev.iloc[1] else np.nan
        o_growth=(float(op.iloc[0])/float(op.iloc[1])-1)*100 if op.iloc[1] else np.nan
        return {"revenue_growth":r_growth,"op_growth":o_growth}
    except Exception:
        return {}

def analyze(ticker, max_dev=5, slope_days=20, breakout_days=60):
    d=prices(ticker)
    if len(d)<100: return None
    d["MA25"]=d.Close.rolling(25).mean()
    d["MA75"]=d.Close.rolling(75).mean()
    d["MA75prev"]=d.MA75.shift(slope_days)
    d["Dev75"]=(d.Close/d.MA75-1)*100
    d["Ret20"]=d.Close.pct_change(20)*100
    d["Ret60"]=d.Close.pct_change(60)*100
    d["High60"]=d.High.shift(1).rolling(breakout_days).max()
    d["Vol20"]=d.Volume.rolling(20).mean()
    d["VolRatio"]=d.Volume/d.Vol20

    x=d.iloc[-1]; p=d.iloc[-2]
    close=float(x.Close); high=float(x.High); low=float(x.Low)
    ma25=float(x.MA25); ma75=float(x.MA75)
    slope=(ma75/float(x.MA75prev)-1)*100
    dev=float(x.Dev75); ret20=float(x.Ret20); ret60=float(x.Ret60)
    volratio=float(x.VolRatio) if np.isfinite(x.VolRatio) else 0
    high60=float(x.High60) if np.isfinite(x.High60) else np.nan

    # A: 75日線押し目
    A=(slope>0 and -max_dev<=dev<=2.0)
    # 75日線付近で下から回復した/反転した
    crossed=(float(p.Close)<=float(p.MA75) and close>ma75)
    near_rebound=(dev<=0 and close>float(p.Close) and close>float(p.Open))
    A_score=min(100,max(0,50+slope*8-abs(dev)*8+(25 if crossed else 0)+(10 if near_rebound else 0)))

    # B: 60日高値ブレイク＋出来高増加＋トレンド
    B=(np.isfinite(high60) and close>high60 and volratio>=1.3 and slope>0)
    B_score=min(100,max(0,(ret20*3)+(volratio-1)*30+(30 if B else 0)))

    # C: 決算モメンタム（Yahooの四半期財務が取れた場合のみ）
    f=fundamentals(ticker)
    rg=f.get("revenue_growth",np.nan); og=f.get("op_growth",np.nan)
    C=bool(np.isfinite(og) and og>=20 and (not np.isfinite(rg) or rg>=0))
    C_score=(min(100,max(0,og*1.5)) if np.isfinite(og) else 0)

    # D: 強いモメンタム→調整→再上昇
    # 60日で上昇、現在は25日線近辺/75日線上、直近5日で反発
    ret5=float(d.Close.pct_change(5).iloc[-1])*100
    pullback=(ret60>=10 and close<=float(d.Close.tail(20).max())*0.97)
    rebound=(ret5>0 and close>ma25)
    D=(ret60>=10 and pullback and rebound and slope>0)
    D_score=min(100,max(0,ret60*2+ret5*4+(25 if D else 0)))

    # 買い価格：Aは75日線上抜け、B/Dは直近高値/当日高値の1ティック上
    if A and close<=ma75:
        buy=ma75+tick_size(ma75); setup="A 75日線上抜け待ち"
    elif B:
        buy=high+tick_size(high); setup="B ブレイクアウト"
    elif D:
        buy=high+tick_size(high); setup="D モメンタム再上昇"
    else:
        buy=ma75+tick_size(ma75); setup="監視"

    stop=buy*0.98
    target=buy+(buy-stop)*2
    hits=sum([A,B,C,D])
    total=0.30*A_score+0.25*B_score+0.20*C_score+0.25*D_score + hits*5

    if hits>=3: verdict="🔥 最優先"
    elif hits>=2: verdict="🟢 強候補"
    elif A or B or C or D: verdict="🟡 候補"
    else: verdict="⚪ 見送り"

    return {
        "コード":ticker.replace(".T",""),"株価":close,"75日線":ma75,
        "75日線傾き%":slope,"75日線乖離%":dev,
        "A 75日線押し目":"🟢" if A else "－",
        "B ブレイク":"🟢" if B else "－",
        "C 決算モメンタム":"🟢" if C else "－",
        "D モメンタム押し目":"🟢" if D else "－",
        "買い価格":buy,"損切り目安":stop,"利確目安(2R)":target,
        "総合スコア":total,"判定":verdict,"買いセットアップ":setup,
        "営業利益成長率%":og
    }

# ---------- UI ----------
st.title("🎯 短期上昇株ハンター")
st.caption("A〜Dの4戦略で日本株をスクリーニング。買い価格まで自動計算。")

with st.sidebar:
    st.header("A：75日線押し目")
    max_dev=st.slider("75日線からの許容乖離（%）",1.0,10.0,5.0,0.5)
    slope_days=st.slider("75日線の傾きを見る期間",5,40,20)
    st.header("B：ブレイクアウト")
    breakout_days=st.slider("高値更新判定期間（日）",20,120,60,10)
    st.header("リスク管理")
    stop_pct=st.slider("損切り目安（%）",0.5,10.0,2.0,0.5)
    period=st.selectbox("株価取得期間",["6mo","1y","2y"],index=1)
    st.caption("CはYahoo Financeで四半期財務が取得できる場合のみ判定します。決算サプライズ（市場予想比）は別途データが必要です。")

default="""8343
8306
8316
8411
8591
8604
8331
8354
8366
8377
8385
1301
1332
1605
1925
1928
2914
3382
3402
4063
4502
4503
4568
5401
6501
6758
6861
6902
7011
7203
7267
7741
7974
8001
8031
8058
8766
8801
8802
9020
9021
9022
9432
9433
9434
9501
9503
9531
9983
9984"""

txt=st.text_area("監視銘柄（東証コードを1行1銘柄）",default,height=230)
tickers=[]
for s in txt.splitlines():
    s=s.strip().upper()
    if not s: continue
    if s.isdigit(): s=s.zfill(4)+".T"
    tickers.append(s)

if st.button("🚀 4戦略でスクリーニング",type="primary",use_container_width=True):
    rows=[]; prog=st.progress(0)
    for i,t in enumerate(tickers):
        try:
            r=analyze(t,max_dev,slope_days,breakout_days)
            if r: rows.append(r)
        except Exception:
            pass
        prog.progress((i+1)/max(1,len(tickers)))
    prog.empty()

    if rows:
        df=pd.DataFrame(rows).sort_values("総合スコア",ascending=False)
        st.subheader("🏆 総合ランキング")
        cols=["コード","株価","75日線","75日線傾き%","75日線乖離%",
              "A 75日線押し目","B ブレイク","C 決算モメンタム","D モメンタム押し目",
              "買い価格","損切り目安","利確目安(2R)","総合スコア","判定"]
        st.dataframe(df[cols].style.format({
            "株価":yen,"75日線":yen,"買い価格":yen,"損切り目安":yen,"利確目安(2R)":yen,
            "75日線傾き%":"{:+.2f}","75日線乖離%":"{:+.2f}","総合スコア":"{:.1f}"
        }),use_container_width=True,hide_index=True)

        st.subheader("🔎 戦略別")
        tabs=st.tabs(["A 押し目","B ブレイク","C 決算","D モメンタム押し目"])
        checks=[
            df["A 75日線押し目"]=="🟢",
            df["B ブレイク"]=="🟢",
            df["C 決算モメンタム"]=="🟢",
            df["D モメンタム押し目"]=="🟢"
        ]
        for tab,mask in zip(tabs,checks):
            with tab:
                st.dataframe(df[mask][cols],use_container_width=True,hide_index=True)
        st.subheader("💰 買い価格について")
        st.info("Aは75日線上抜け＋1ティック、B/Dは当日高値＋1ティックを基本値として表示。損切りは現在2%の暫定値。")
        csv=df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSV保存",csv,"short_term_stock_hunter.csv","text/csv")
    else:
        st.error("データを取得できる銘柄がありませんでした。")

st.divider()
st.caption("投資判断を自動化するための研究用スクリーナーです。買い・売りを保証するものではありません。")
