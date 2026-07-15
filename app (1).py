#!/usr/bin/env python3
"""
Application web du bot d'analyse et de signaux de trading.
------------------------------------------------------------------
Interface graphique locale (dans ton navigateur) basée sur Streamlit.
Même moteur d'analyse que signal_bot.py (SMA, MACD, RSI, Bollinger).

Installation :
    pip install -r requirements.txt --break-system-packages

Lancement :
    streamlit run app.py

Ça ouvre automatiquement une page dans ton navigateur (généralement
http://localhost:8501). L'appli tourne sur TON ordinateur, en local :
personne d'autre n'y a accès sauf toi.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    st.error("Il manque la librairie yfinance. Lance : pip install -r requirements.txt --break-system-packages")
    st.stop()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Il manque la librairie plotly. Lance : pip install -r requirements.txt --break-system-packages")
    st.stop()


# ----------------------------------------------------------------------
# Moteur d'analyse (identique à signal_bot.py)
# ----------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Aucune donnée trouvée pour '{symbol}'. Vérifie le symbole.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    df["BB_mid"] = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * std20
    df["BB_lower"] = df["BB_mid"] - 2 * std20

    return df


def generate_signal(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reasons = []

    if last["SMA20"] > last["SMA50"]:
        score += 1
        reasons.append("Tendance haussière (SMA20 > SMA50)")
    elif last["SMA20"] < last["SMA50"]:
        score -= 1
        reasons.append("Tendance baissière (SMA20 < SMA50)")

    if last["MACD"] > last["MACD_signal"] and prev["MACD"] <= prev["MACD_signal"]:
        score += 2
        reasons.append("Croisement MACD haussier")
    elif last["MACD"] < last["MACD_signal"] and prev["MACD"] >= prev["MACD_signal"]:
        score -= 2
        reasons.append("Croisement MACD baissier")
    elif last["MACD"] > last["MACD_signal"]:
        score += 1
        reasons.append("MACD au-dessus de sa ligne de signal")
    else:
        score -= 1
        reasons.append("MACD en dessous de sa ligne de signal")

    rsi = last["RSI"]
    if rsi < 30:
        score += 2
        reasons.append(f"RSI en zone de survente ({rsi:.1f})")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI en zone de surachat ({rsi:.1f})")
    else:
        reasons.append(f"RSI neutre ({rsi:.1f})")

    price = last["Close"]
    if price <= last["BB_lower"]:
        score += 1
        reasons.append("Prix proche/sous la bande de Bollinger basse")
    elif price >= last["BB_upper"]:
        score -= 1
        reasons.append("Prix proche/au-dessus de la bande de Bollinger haute")

    if score >= 3:
        signal = "ACHAT FORT"
    elif score >= 1:
        signal = "ACHAT"
    elif score <= -3:
        signal = "VENTE FORTE"
    elif score <= -1:
        signal = "VENTE"
    else:
        signal = "NEUTRE"

    return {"signal": signal, "score": score, "price": price, "reasons": reasons}


SIGNAL_COLORS = {
    "ACHAT FORT": "#0a7d2c",
    "ACHAT": "#4caf50",
    "NEUTRE": "#9e9e9e",
    "VENTE": "#e57373",
    "VENTE FORTE": "#b71c1c",
}


SPINNER_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:30px 0;">
  <div style="
      width:64px;height:64px;border-radius:50%;
      border:5px solid rgba(76,175,80,0.15);
      border-top:5px solid #4caf50;
      border-right:5px solid #4caf50;
      animation:tb-spin 0.9s linear infinite;
  "></div>
  <div style="margin-top:14px;color:#9e9e9e;font-size:0.95em;letter-spacing:0.5px;">
    Analyse de {symbol} en cours…
  </div>
</div>
<style>
@keyframes tb-spin {{
  0%   {{ transform: rotate(0deg); }}
  100% {{ transform: rotate(360deg); }}
}}
</style>
"""


def make_glow_ring(score: int, signal: str) -> str:
    """Anneau circulaire lumineux (façon 'winrate') qui visualise le score du signal."""
    color = SIGNAL_COLORS.get(signal, "#9e9e9e")
    percent = max(0, min(100, ((score + 6) / 12) * 100))
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:8px 0 22px;">
      <div style="font-size:0.78em;color:#8a8f98;letter-spacing:2px;margin-bottom:10px;">SCORE DU SIGNAL</div>
      <div style="
          width:150px;height:150px;border-radius:50%;
          background: conic-gradient({color} {percent}%, rgba(255,255,255,0.07) {percent}% 100%);
          display:flex;align-items:center;justify-content:center;
          box-shadow: 0 0 22px {color}55;
          animation: tb-ring-in 0.6s ease-out;
      ">
        <div style="
            width:112px;height:112px;border-radius:50%;background:#0e1117;
            display:flex;flex-direction:column;align-items:center;justify-content:center;
        ">
          <div style="font-size:1.8em;font-weight:800;color:{color};text-shadow:0 0 14px {color}aa;">{score:+d}</div>
          <div style="font-size:0.65em;color:#8a8f98;letter-spacing:1px;">sur 12</div>
        </div>
      </div>
    </div>
    <style>
    @keyframes tb-ring-in {{
      0%   {{ opacity:0; transform:scale(0.85); }}
      100% {{ opacity:1; transform:scale(1); }}
    }}
    </style>
    """


def make_simple_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Graphique épuré : juste le prix + moyenne mobile + bandes de Bollinger. Pensé pour mobile."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_lower"], line=dict(width=0), fill="tonexty",
        fillcolor="rgba(76,175,80,0.08)", name="Zone Bollinger", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(width=2, color="#4caf50"), name="Prix"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], line=dict(width=1, color="#90a4ae", dash="dot"), name="Moyenne 20j"))

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=25, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    return fig


def make_detailed_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Graphique complet (bougies + RSI + MACD), pour ceux qui veulent le détail technique."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.2, 0.3],
        vertical_spacing=0.06,
        subplot_titles=("Prix & Bollinger", "RSI", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Prix"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], line=dict(width=1), name="SMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], line=dict(width=1), name="SMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], line=dict(width=1, dash="dot"), name="BB haute"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], line=dict(width=1, dash="dot"), name="BB basse"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(width=1.5, color="purple"), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(width=1), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(width=1), name="Signal"), row=3, col=1)

    fig.update_layout(
        height=650, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0, font=dict(size=10)),
    )
    return fig


# ----------------------------------------------------------------------
# Interface Streamlit
# ----------------------------------------------------------------------
st.set_page_config(page_title="Bot de signaux de trading", page_icon="📈", layout="wide")

# --- PWA : rend l'appli "installable" sur mobile (icône sur l'écran d'accueil) ---
st.markdown(
    """
    <link rel="manifest" href="./app/static/manifest.json">
    <link rel="icon" href="./app/static/icon.svg" type="image/svg+xml">
    <link rel="apple-touch-icon" href="./app/static/icon.svg">
    <meta name="theme-color" content="#0e1117">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <script>
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('./app/static/service-worker.js').catch(function(){});
      }
    </script>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #4caf50, #4fc3f7);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 0;
    ">📈 Bot d'analyse et de signaux</h1>
    """,
    unsafe_allow_html=True,
)
st.caption("Outil d'aide à la décision — aucun ordre n'est passé automatiquement. Ce n'est pas un conseil financier.")

WATCHLISTS = {
    "Personnalisé": "AAPL, BTC-USD",
    "Aérospatiale / Espace": "RKLB, ASTS, ARKX, LMT, NOC, BA, RTX",
    "Tech US": "AAPL, MSFT, NVDA, GOOGL, AMZN, META",
    "Crypto majeures": "BTC-USD, ETH-USD, SOL-USD, XRP-USD, BNB-USD",
    "Forex majeures": "EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X",
    "Indices": "^GSPC, ^IXIC, ^FCHI, ^GDAXI",
}

with st.sidebar:
    st.header("Paramètres")
    preset = st.selectbox("Watchlist thématique", list(WATCHLISTS.keys()))
    default_value = WATCHLISTS[preset]
    symbols_input = st.text_input("Symboles (séparés par des virgules)", value=default_value)
    st.caption(
        "ℹ️ SpaceX n'étant pas cotée en bourse, la watchlist \"Aérospatiale / Espace\" "
        "propose des équivalents publics : Rocket Lab (RKLB), AST SpaceMobile (ASTS), "
        "l'ETF ARK Space Exploration (ARKX), Lockheed Martin, Northrop Grumman, Boeing, RTX."
    )
    period = st.selectbox("Période historique", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Intervalle des bougies", ["1d", "1h", "30m", "15m", "1wk"], index=0)
    st.markdown("---")
    st.markdown(
        "**Exemples de symboles**\n"
        "- Actions : `AAPL`, `MSFT`, `TTE.PA`\n"
        "- Crypto : `BTC-USD`, `ETH-USD`\n"
        "- Forex : `EURUSD=X`, `USDJPY=X`\n"
        "- Indices : `^GSPC`, `^FCHI`"
    )
    run = st.button("🔍 Analyser", type="primary", use_container_width=True)

if run:
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    if not symbols:
        st.warning("Entre au moins un symbole.")
    for symbol in symbols:
        st.markdown("---")
        placeholder = st.empty()
        placeholder.markdown(SPINNER_HTML.format(symbol=symbol), unsafe_allow_html=True)
        try:
            df = fetch_data(symbol, period, interval)
            df = compute_indicators(df)
            result = generate_signal(df)
        except Exception as e:
            placeholder.empty()
            st.error(f"{symbol} : {e}")
            continue
        placeholder.empty()

        color = SIGNAL_COLORS.get(result["signal"], "#9e9e9e")

        # --- Bandeau signal, compact et lisible sur mobile ---
        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,{color}dd,{color}99);color:white;padding:14px 18px;
            border-radius:14px;display:flex;justify-content:space-between;align-items:center;
            box-shadow:0 0 18px {color}44;'>
                <span style='font-size:1.15em;font-weight:bold;letter-spacing:0.5px;'>{symbol}</span>
                <span style='font-size:1.15em;font-weight:bold;'>{result['signal']}</span>
                <span style='font-size:1em;'>{result['price']:.4f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Anneau lumineux du score, façon "winrate" ---
        st.markdown(make_glow_ring(result["score"], result["signal"]), unsafe_allow_html=True)

        # --- Graphique simple, toujours visible ---
        st.plotly_chart(make_simple_chart(df, symbol), use_container_width=True, config={"displayModeBar": False})

        # --- Détails repliés par défaut : moins de bruit visuel ---
        with st.expander("Pourquoi ce signal ?"):
            for r in result["reasons"]:
                st.markdown(f"- {r}")

        with st.expander("Graphiques détaillés (bougies, RSI, MACD)"):
            st.plotly_chart(make_detailed_chart(df, symbol), use_container_width=True)

        # --- Verdict final, bien visible après toute l'analyse ---
        if result["signal"] in ("ACHAT", "ACHAT FORT"):
            verdict_icon, verdict_text = "✅", "ACHETER"
        elif result["signal"] in ("VENTE", "VENTE FORTE"):
            verdict_icon, verdict_text = "🔻", "VENDRE"
        else:
            verdict_icon, verdict_text = "➖", "ATTENDRE"

        st.markdown(
            f"""
            <div style='margin-top:8px;padding:16px;border-radius:14px;
            border:2px solid {color};text-align:center;box-shadow:0 0 16px {color}33;'>
                <div style='font-size:0.85em;color:#9e9e9e;letter-spacing:2px;'>VERDICT</div>
                <div style='font-size:1.6em;font-weight:800;color:{color};text-shadow:0 0 10px {color}66;'>{verdict_icon} {verdict_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Renseigne un ou plusieurs symboles dans la barre de gauche, puis clique sur **Analyser**.")
