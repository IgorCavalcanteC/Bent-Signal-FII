"""
Scanner de pre-sinal e sinal confirmado (HiLo + Squeeze + Inflexao de momentum)
para Fundos Imobiliarios (FIIs) da B3. Roda sobre o universo FII_UNIVERSE,
aplica um filtro de liquidez minima (volume financeiro medio diario), detecta
sinais no ultimo candle FECHADO e envia alerta via Telegram.
 
Uso local:
    pip install -r requirements.txt
    export TELEGRAM_BOT_TOKEN="123456:abc..."
    export TELEGRAM_CHAT_ID="123456789"
    python scanner.py
"""
 
import os
import re
import json
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
 
# ============ CONFIG ============
TIMEFRAME = "1d"                  # candle diario
PERIOD = "2y"                     # historico buscado (garante indicadores estabilizados)
MIN_LIQUIDEZ_DIARIA = 300_000     # volume financeiro medio (R$/dia) minimo p/ entrar no scan
 
FII_UNIVERSE = [                  # universo amplo e curado de FIIs B3 (tijolo, papel, FoF, hibrido, fiagro)
    # Tijolo - Lajes corporativas / Escritorios
    "HGRE11.SA", "KNRI11.SA", "RCRB11.SA", "JSRE11.SA", "PVBI11.SA",
    "BRCO11.SA", "TRXF11.SA", "RBRP11.SA", "PATL11.SA",
    # Tijolo - Logistica / Galpoes
    "HGLG11.SA", "XPLG11.SA", "VILG11.SA", "BTLG11.SA", "LVBI11.SA",
    "GGRC11.SA", "VINO11.SA", "RBRL11.SA",
    # Tijolo - Shoppings
    "VISC11.SA", "XPML11.SA", "HSML11.SA", "SHPH11.SA",
    "ABCP11.SA",
    # Tijolo - Renda urbana / hibrido / agencias
    "HGRU11.SA", "HGBS11.SA", "TGAR11.SA", "RBVA11.SA", "MFII11.SA",
    # Papel / Recebiveis (CRI)
    "MXRF11.SA", "KNCR11.SA", "KNIP11.SA", "KNSC11.SA",
    "CPTS11.SA", "VGIP11.SA", "RBRR11.SA", "HGCR11.SA", "VGIR11.SA",
    "RECR11.SA", "DEVA11.SA", "OUJP11.SA", "URPR11.SA", "VRTA11.SA",
    "SNCI11.SA", "CACR11.SA", "BCRI11.SA", "RECT11.SA",
    # Fundo de Fundos (FoF)
    "HFOF11.SA", "RBRF11.SA", "KFOF11.SA",
    "XPSF11.SA", "ALZR11.SA",
    # Fiagro
    "RZAG11.SA", "VGIA11.SA", "RURA11.SA", "KNCA11.SA", "AGRX11.SA",
]
 
# ---- Criterio fundamentalista (Status Invest) ----
PVP_MAX = 0.90      # P/VP abaixo disso = fundo descontado
DY_MIN = 9.5        # Dividend Yield 12m (%) acima disso = rendimento elevado
RSI_VALOR_MAX = 45  # confirmacao tecnica: nao esta em queda livre forte
 
HILO_LEN = 34
RSI_LEN = 14
BB_LEN = 20
BB_MULT = 2.0
SQUEEZE_LOOKBACK = 50
SQUEEZE_TOL = 1.15
EMA_FAST, EMA_MID, EMA_SLOW = 8, 21, 34
ATR_PERIOD = 10
ST_MULT = 2.0
 
STATE_FILE = "state.json"
RESULTS_FILE = "docs/results.json"   # alimenta o painel (GitHub Pages)
# =================================
 
 
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()
 
 
def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)
 
 
def supertrend(df: pd.DataFrame, period: int, mult: float):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
 
    upper = high.rolling(period).max() - mult * atr
    lower = low.rolling(period).min() + mult * atr
 
    st = [close.iloc[0]]
    direction = [1]
    for i in range(1, len(df)):
        c = close.iloc[i]
        prev_st = st[-1]
        d = 1 if c > prev_st else (-1 if c < prev_st else direction[-1])
        new_st = max(upper.iloc[i], prev_st) if d == 1 else min(lower.iloc[i], prev_st)
        st.append(new_st)
        direction.append(d)
    df["supertrend"] = st
    df["st_dir"] = direction
    return df
 
 
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hima"] = ema(df["high"], HILO_LEN)
    df["loma"] = ema(df["low"], HILO_LEN)
 
    hilo = [df["close"].iloc[0]]
    for i in range(1, len(df)):
        c = df["close"].iloc[i]
        if c < df["loma"].iloc[i]:
            hilo.append(df["hima"].iloc[i])
        elif c > df["hima"].iloc[i]:
            hilo.append(df["loma"].iloc[i])
        else:
            hilo.append(hilo[-1])
    df["hilo"] = hilo
 
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["ema8"] = ema(df["close"], EMA_FAST)
    df["ema21"] = ema(df["close"], EMA_MID)
    df["ema34"] = ema(df["close"], EMA_SLOW)
 
    basis = df["close"].rolling(BB_LEN).mean()
    dev = BB_MULT * df["close"].rolling(BB_LEN).std()
    df["bb_width"] = (2 * dev) / basis
    df["bb_width_low"] = df["bb_width"].rolling(SQUEEZE_LOOKBACK).min()
    df["is_squeeze"] = df["bb_width"] <= df["bb_width_low"] * SQUEEZE_TOL
 
    df["rsi_curl_up"] = (
        (df["rsi"] > df["rsi"].shift(1))
        & (df["rsi"].shift(1) <= df["rsi"].shift(2))
        & (df["rsi"] < 55)
    )
    df["rsi_curl_down"] = (
        (df["rsi"] < df["rsi"].shift(1))
        & (df["rsi"].shift(1) >= df["rsi"].shift(2))
        & (df["rsi"] > 45)
    )
 
    ema8_slope = df["ema8"].diff()
    df["slope_curl_up"] = (ema8_slope > 0) & (ema8_slope.shift(1) <= 0)
    df["slope_curl_down"] = (ema8_slope < 0) & (ema8_slope.shift(1) >= 0)
 
    df["pre_buy"] = (
        df["is_squeeze"] & df["rsi_curl_up"] & df["slope_curl_up"] & (df["close"] < df["hima"])
    )
    df["pre_sell"] = (
        df["is_squeeze"] & df["rsi_curl_down"] & df["slope_curl_down"] & (df["close"] > df["loma"])
    )
 
    df = supertrend(df, ATR_PERIOD, ST_MULT)
 
    buy_cross = (df["ema8"] > df["ema21"]) & (df["ema8"].shift(1) <= df["ema21"].shift(1))
    sell_cross = (df["ema8"] < df["ema21"]) & (df["ema8"].shift(1) >= df["ema21"].shift(1))
 
    df["buy_signal"] = (
        (df["close"] > df["hilo"])
        & (df["st_dir"] == 1)
        & (df["rsi"] > 50)
        & buy_cross
        & (df["ema8"] > df["ema34"])
    )
    df["sell_signal"] = (
        (df["close"] < df["hilo"])
        & (df["st_dir"] == -1)
        & (df["rsi"] < 50)
        & sell_cross
        & (df["ema8"] < df["ema34"])
    )
    return df
 
 
def fetch_fundamentos(symbol: str) -> dict | None:
    """Raspa P/VP e Dividend Yield (12m) da pagina publica do Status Invest.
    Nao ha API oficial; a pagina e renderizada em HTML estatico (sem JS/login).
    Retorna None se nao conseguir extrair os dois indicadores."""
    ticker = symbol.replace(".SA", "").lower()
    url = f"https://statusinvest.com.br/fundos-imobiliarios/{ticker}"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; BentSignalFII/1.0)"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        html = r.text
 
        def extrai_valor(label: str) -> float | None:
            # padrao comum do Status Invest: <h3 class="title">LABEL</h3> ... <strong ...>0,79</strong>
            m = re.search(
                re.escape(label) + r"</(?:h3|span|div)>.*?<strong[^>]*>([\d.,]+)</strong>",
                html,
                re.IGNORECASE | re.DOTALL,
            )
            if not m:
                return None
            valor = m.group(1).replace(".", "").replace(",", ".")
            try:
                return float(valor)
            except ValueError:
                return None
 
        pvp = extrai_valor("P/VP")
        dy = extrai_valor("Dividend Yield")
        if pvp is None or dy is None:
            return None
        return {"pvp": pvp, "dy": dy}
    except Exception as e:
        print(f"[{symbol}] erro ao buscar fundamentos (Status Invest): {e}")
        return None
 
 
def build_row(symbol: str, df: pd.DataFrame) -> dict:
    """Monta a linha do painel a partir do ultimo candle fechado."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    var_pct = (last["close"] - prev["close"]) / prev["close"] * 100
 
    score = 0
    score += 20 if last["close"] > last["hilo"] else -20
    score += 15 if last["st_dir"] == 1 else -15
    score += (last["rsi"] - 50) * 0.6
    score += 15 if last["buy_signal"] else (-15 if last["sell_signal"] else 0)
    score += 8 if last["pre_buy"] else (-8 if last["pre_sell"] else 0)
    score = max(-100, min(100, round(score)))
 
    if last["buy_signal"]:
        sinal = "COMPRAR"
    elif last["sell_signal"]:
        sinal = "VENDER"
    elif last["pre_buy"]:
        sinal = "PRE-COMPRA"
    elif last["pre_sell"]:
        sinal = "PRE-VENDA"
    else:
        sinal = "AGUARDAR"
 
    return {
        "ativo": symbol,
        "preco": round(float(last["close"]), 6),
        "var_pct": round(float(var_pct), 2),
        "sinal": sinal,
        "score": score,
        "indicadores": {
            "hilo": "Acima" if last["close"] > last["hilo"] else "Abaixo",
            "supertrend": "Alta" if last["st_dir"] == 1 else "Baixa",
            "rsi": round(float(last["rsi"]), 1),
            "squeeze": bool(last["is_squeeze"]),
        },
        "timeframe": TIMEFRAME,
    }
 
 
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}
 
 
def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
 
 
def format_price(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
 
 
def send_telegram(msg: str):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
 
    # Telegram limita a 4096 caracteres por mensagem - quebra em partes se precisar
    limite = 3800
    partes = [msg[i:i + limite] for i in range(0, len(msg), limite)] or [msg]
 
    for parte in partes:
        params = {"chat_id": chat_id, "text": parte}
        r = requests.get(url, params=params, timeout=15)
        print("Telegram:", r.status_code, r.text[:200])
 
 
def build_summary_message(compras, pre_compras, vendas, pre_vendas, oportunidades_valor, data_ref) -> str:
    linhas = [f"📊 Sinais do dia (FIIs) — {data_ref}", ""]
 
    tem_algo = compras or pre_compras or vendas or pre_vendas or oportunidades_valor
    if not tem_algo:
        linhas.append("Não houveram movimentos consistentes no dia de hoje, mantenha suas posições.")
        return "\n".join(linhas).strip()
 
    if compras or pre_compras:
        linhas.append("🟢 ENTRADAS")
        for sym, preco in compras:
            linhas.append(f"Compra de {sym} ao valor de {format_price(preco)}")
        for sym, preco in pre_compras:
            linhas.append(f"Pré-compra de {sym} ao valor de {format_price(preco)}")
        linhas.append("")
 
    if vendas or pre_vendas:
        linhas.append("🔴 SAÍDAS")
        for sym, preco in vendas:
            linhas.append(f"Venda de {sym} ao valor de {format_price(preco)}")
        for sym, preco in pre_vendas:
            linhas.append(f"Pré-venda de {sym} ao valor de {format_price(preco)}")
        linhas.append("")
 
    if oportunidades_valor:
        linhas.append("🔵 OPORTUNIDADE DE VALOR")
        for sym, preco, pvp, dy in oportunidades_valor:
            linhas.append(
                f"{sym} a {format_price(preco)} — P/VP {pvp:.2f}, DY 12m {dy:.1f}%"
            )
 
    return "\n".join(linhas).strip()
 
 
def main():
    state = load_state()
    results = []
    compras, pre_compras, vendas, pre_vendas = [], [], [], []
    oportunidades_valor = []
    descartados_liquidez = []
 
    for symbol in FII_UNIVERSE:
        try:
            df = yf.Ticker(symbol).history(period=PERIOD, interval=TIMEFRAME)
            df = df.reset_index()
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            df["ts"] = pd.to_datetime(df[ts_col]).astype("int64") // 10**9
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })[["ts", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            print(f"[{symbol}] erro ao buscar candles: {e}")
            continue
        df = df.iloc[:-1] if TIMEFRAME != "1d" else df  # so descarta candle "em formacao" no intrabar
        if len(df) < SQUEEZE_LOOKBACK + 5:
            continue
 
        # filtro de liquidez: volume financeiro medio dos ultimos 21 pregoes
        vol_financeiro_medio = (df["close"] * df["volume"]).tail(21).mean()
        if pd.isna(vol_financeiro_medio) or vol_financeiro_medio < MIN_LIQUIDEZ_DIARIA:
            descartados_liquidez.append(symbol.replace(".SA", ""))
            continue
 
        df = compute_indicators(df)
        results.append(build_row(symbol, df))
 
        last = df.iloc[-1]
        last_ts = str(int(last["ts"]))
 
        key_pre = f"{symbol}:pre:{last_ts}"
        key_buy = f"{symbol}:buy:{last_ts}"
        key_sell = f"{symbol}:sell:{last_ts}"
        nome = symbol.replace(".SA", "")
 
        if last["pre_buy"] and not state.get(key_pre):
            pre_compras.append((nome, float(last["close"])))
            state[key_pre] = True
 
        if last["pre_sell"] and not state.get(key_pre + "s"):
            pre_vendas.append((nome, float(last["close"])))
            state[key_pre + "s"] = True
 
        if last["buy_signal"] and not state.get(key_buy):
            compras.append((nome, float(last["close"])))
            state[key_buy] = True
 
        if last["sell_signal"] and not state.get(key_sell):
            vendas.append((nome, float(last["close"])))
            state[key_sell] = True
 
        # criterio fundamentalista (Status Invest): desconto + yield elevado
        fund = fetch_fundamentos(symbol)
        if fund and fund["pvp"] < PVP_MAX and fund["dy"] > DY_MIN and last["rsi"] < RSI_VALOR_MAX:
            key_valor = f"{symbol}:valor:{last_ts}"
            if not state.get(key_valor):
                oportunidades_valor.append((nome, float(last["close"]), fund["pvp"], fund["dy"]))
                state[key_valor] = True
 
        time.sleep(0.6)
 
    if descartados_liquidez:
        print(f"Descartados por liquidez (< R$ {MIN_LIQUIDEZ_DIARIA:,.0f}/dia): {', '.join(descartados_liquidez)}")
 
    data_ref = pd.Timestamp.now("UTC").tz_convert("America/Sao_Paulo").strftime("%d/%m/%Y")
    msg = build_summary_message(compras, pre_compras, vendas, pre_vendas, oportunidades_valor, data_ref)
    send_telegram(msg)
 
    # mantem so as ultimas ~2000 chaves pra nao crescer infinito
    if len(state) > 2000:
        keys = list(state.keys())[-2000:]
        state = {k: state[k] for k in keys}
    save_state(state)
 
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    payload = {
        "updated_at": pd.Timestamp.now("UTC").isoformat(),
        "timeframe": TIMEFRAME,
        "ativos": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
 
 
if __name__ == "__main__":
    main()
 
