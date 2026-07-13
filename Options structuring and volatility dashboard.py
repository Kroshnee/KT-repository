
"""
Options Structuring & Volatility Dashboard
-------------------------------------------
Pulls delayed options chain data (via Yahoo Finance / yfinance), backs out
Black-Scholes implied volatility for every strike/expiry, and renders:
  - IV smile (IV vs strike) for a chosen expiry
  - IV surface (IV vs strike vs time-to-expiry) across multiple expiries
  - Greeks table (delta, gamma, vega, theta, rho) for the selected chain
  - Basic structuring view: payoff diagram for simple 2-leg spreads
 
Data source note: yfinance pulls Yahoo Finance's options chain, which is
free but NOT real-time — quotes are typically delayed ~15-20 min and bid/ask
can be stale for illiquid strikes. This is fine for research/dashboarding,
not for live execution.
 
Run:
    pip install yfinance dash plotly pandas numpy scipy --break-system-packages
    python options_vol_dashboard.py
Then open http://127.0.0.1:8060
"""
 
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import yfinance as yf
from datetime import datetime
 
# --------------------------------------------------------------------------
# Black-Scholes pricing, Greeks, and implied volatility solver
# --------------------------------------------------------------------------
 
def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Black-Scholes-Merton price with continuous dividend yield q."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
 
 
def bs_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Return dict of delta, gamma, vega, theta, rho. Vega/theta per 1.0 vol / 1 year."""
    if T <= 0 or sigma <= 0:
        return dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
 
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100.0  # per 1 vol point (1%)
 
    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365.0
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365.0
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
 
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
 
 
def implied_vol(price, S, K, T, r, q=0.0, option_type="call"):
    """Solve for sigma via Brent's method on the pricing residual. Returns NaN if unsolvable."""
    if price <= 0 or T <= 0:
        return np.nan
    intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
    if price < intrinsic * np.exp(-r * T) - 1e-6:
        return np.nan  # price below no-arbitrage floor, bad quote
 
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, q, option_type) - price
 
    try:
        return brentq(objective, 1e-4, 6.0, xtol=1e-6, maxiter=200)
    except ValueError:
        return np.nan
 
 
# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------
 
def get_risk_free_rate():
    """Approximate risk-free rate from 13-week T-bill yield (^IRX quotes yield *100)."""
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        if not irx.empty:
            return float(irx["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.045  # fallback approx
 
 
def get_dividend_yield(ticker_obj):
    try:
        info = ticker_obj.info
        y = info.get("trailingAnnualDividendYield") or info.get("dividendYield") or 0.0
        return float(y)
    except Exception:
        return 0.0
 
 
def fetch_chain(symbol, expiry, r, q, spot):
    """Fetch calls+puts for one expiry, compute mid price, IV, and greeks."""
    tk = yf.Ticker(symbol)
    try:
        opt = tk.option_chain(expiry)
    except Exception as e:
        print(f"fetch_chain error for {expiry}: {e}")
        return pd.DataFrame()
 
    T = (pd.Timestamp(expiry) - pd.Timestamp.today()).days / 365.0
    T = max(T, 1e-4)
 
    frames = []
    for df, otype in [(opt.calls, "call"), (opt.puts, "put")]:
        if df.empty:
            continue
        df = df.copy()
        df["mid"] = np.where(
            (df["bid"] > 0) & (df["ask"] > 0), (df["bid"] + df["ask"]) / 2, df["lastPrice"]
        )
        df = df[(df["mid"] > 0.01) & (df["strike"] > 0)]
        df["type"] = otype
        df["T"] = T
        df["iv_calc"] = df.apply(
            lambda row: implied_vol(row["mid"], spot, row["strike"], T, r, q, otype), axis=1
        )
        frames.append(df)
 
    if not frames:
        return pd.DataFrame()
 
    chain = pd.concat(frames, ignore_index=True)
    chain = chain.dropna(subset=["iv_calc"])
    chain = chain[(chain["iv_calc"] > 0.01) & (chain["iv_calc"] < 4.0)]  # drop garbage IVs
 
    greeks = chain.apply(
        lambda row: bs_greeks(spot, row["strike"], row["T"], r, row["iv_calc"], q, row["type"]),
        axis=1,
        result_type="expand",
    )
    chain = pd.concat([chain.reset_index(drop=True), greeks.reset_index(drop=True)], axis=1)
    chain["moneyness"] = chain["strike"] / spot
    return chain
 
 
# --------------------------------------------------------------------------
# Dash app
# --------------------------------------------------------------------------
 
app = dash.Dash(__name__, external_stylesheets=["https://codepen.io"])
app.title = "Vol Surface Dashboard"
 
DARK = "#111111"
PANEL = "#1a1a1a"
FONT = "monospace"
 
app.layout = html.Div(
    style={"backgroundColor": DARK, "color": "#FFFFFF", "padding": "20px", "fontFamily": FONT, "minHeight": "100vh"},
    children=[
        html.H1("Options Structuring & Volatility Dashboard", style={"textAlign": "center"}),
 
        html.Div(
            style={"display": "flex", "justifyContent": "center", "gap": "20px", "marginBottom": "20px", "flexWrap": "wrap"},
            children=[
                dcc.Input(id="ticker-input", type="text", value="AAPL", debounce=True,
                           style={"backgroundColor": PANEL, "color": "#0f0", "border": "1px solid #444", "padding": "8px", "width": "120px"}),
                html.Button("Load", id="load-btn", n_clicks=0,
                            style={"backgroundColor": "#00ffcc", "color": "#000", "border": "none", "padding": "8px 16px", "cursor": "pointer"}),
                dcc.Dropdown(id="expiry-dropdown", placeholder="Select expiry for smile",
                             style={"width": "220px", "color": "#000"}),
                dcc.RadioItems(
                    id="option-type-radio",
                    options=[{"label": " Calls", "value": "call"}, {"label": " Puts", "value": "put"}, {"label": " Both", "value": "both"}],
                    value="both", inline=True, style={"color": "#fff"},
                ),
            ],
        ),
 
        html.Div(id="status-msg", style={"textAlign": "center", "color": "#ff9900", "marginBottom": "10px"}),
 
        # Top metrics
        html.Div(
            id="metrics-bar",
            style={"display": "flex", "justifyContent": "space-around", "marginBottom": "20px"},
        ),
 
        # Smile + Surface side by side
        html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "20px"},
            children=[
                html.Div([dcc.Graph(id="smile-graph")], style={"flex": "1", "minWidth": "450px"}),
                html.Div([dcc.Graph(id="surface-graph")], style={"flex": "1", "minWidth": "450px"}),
            ],
        ),
 
        html.H3("Chain & Greeks (selected expiry)", style={"marginTop": "30px"}),
        html.Div(id="greeks-table-container"),
 
        # Hidden store for fetched chain data across callbacks
        dcc.Store(id="chain-store"),
        dcc.Store(id="meta-store"),
    ],
)
 
 
# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
 
@app.callback(
    [Output("expiry-dropdown", "options"),
     Output("expiry-dropdown", "value"),
     Output("meta-store", "data"),
     Output("status-msg", "children")],
    [Input("load-btn", "n_clicks")],
    [State("ticker-input", "value")],
)
def load_ticker(n_clicks, symbol):
    if not symbol:
        return [], None, {}, ""
    symbol = symbol.strip().upper()
    try:
        tk = yf.Ticker(symbol)
        expiries = tk.options
        if not expiries:
            return [], None, {}, f"No listed options found for {symbol}."
        hist = tk.history(period="1d")
        if hist.empty:
            return [], None, {}, f"Could not fetch spot price for {symbol}."
        spot = float(hist["Close"].iloc[-1])
        r = get_risk_free_rate()
        q = get_dividend_yield(tk)
        meta = {"symbol": symbol, "spot": spot, "r": r, "q": q, "expiries": list(expiries)}
        options = [{"label": e, "value": e} for e in expiries]
        msg = f"{symbol} spot=${spot:,.2f}  r={r*100:.2f}%  q={q*100:.2f}%  ({len(expiries)} expiries)"
        return options, expiries[0], meta, msg
    except Exception as e:
        return [], None, {}, f"Error loading {symbol}: {e}"
 
 
@app.callback(
    [Output("chain-store", "data"),
     Output("metrics-bar", "children")],
    [Input("expiry-dropdown", "value")],
    [State("meta-store", "data")],
)
def load_chain(expiry, meta):
    if not expiry or not meta:
        return None, []
 
    symbol, spot, r, q = meta["symbol"], meta["spot"], meta["r"], meta["q"]
 
    # Build IV surface across up to 8 nearest expiries, and detailed chain for selected expiry
    surface_frames = []
    for exp in meta["expiries"][:8]:
        df = fetch_chain(symbol, exp, r, q, spot)
        if not df.empty:
            surface_frames.append(df)
    all_chain = pd.concat(surface_frames, ignore_index=True) if surface_frames else pd.DataFrame()
 
    if all_chain.empty:
        return None, []
 
    # ATM IV for metrics (closest strike to spot, calls, nearest expiry)
    nearest_T = all_chain["T"].min()
    atm_slice = all_chain[(all_chain["T"] == nearest_T) & (all_chain["type"] == "call")]
    atm_iv = np.nan
    if not atm_slice.empty:
        atm_row = atm_slice.iloc[(atm_slice["strike"] - spot).abs().argsort()[:1]]
        atm_iv = float(atm_row["iv_calc"].iloc[0])
 
    skew_25d = np.nan
    put_slice = all_chain[(all_chain["T"] == nearest_T) & (all_chain["type"] == "put")]
    if not put_slice.empty and not atm_slice.empty:
        otm_put = put_slice[put_slice["strike"] < spot * 0.95]
        otm_call = atm_slice[atm_slice["strike"] > spot * 1.05]
        if not otm_put.empty and not otm_call.empty:
            skew_25d = float(otm_put["iv_calc"].mean() - otm_call["iv_calc"].mean())
 
    metrics = [
        html.Div([html.H4("Spot", style={"color": "#888"}), html.H2(f"${spot:,.2f}", style={"color": "#00ffcc"})]),
        html.Div([html.H4("ATM IV (front month)", style={"color": "#888"}), html.H2(f"{atm_iv*100:.1f}%" if not np.isnan(atm_iv) else "N/A", style={"color": "#ff9900"})]),
        html.Div([html.H4("25d Put-Call Skew", style={"color": "#888"}), html.H2(f"{skew_25d*100:+.1f} vol pts" if not np.isnan(skew_25d) else "N/A", style={"color": "#ff3366"})]),
        html.Div([html.H4("Risk-free rate", style={"color": "#888"}), html.H2(f"{r*100:.2f}%", style={"color": "#00ffcc"})]),
    ]
 
    return all_chain.to_json(date_format="iso", orient="split"), metrics
 
 
@app.callback(
    [Output("smile-graph", "figure"),
     Output("surface-graph", "figure"),
     Output("greeks-table-container", "children")],
    [Input("chain-store", "data"),
     Input("expiry-dropdown", "value"),
     Input("option-type-radio", "value")],
    [State("meta-store", "data")],
)
def update_graphs(chain_json, expiry, opt_type, meta):
    empty_fig = go.Figure().update_layout(template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=PANEL)
    if not chain_json or not meta:
        return empty_fig, empty_fig, html.Div("Load a ticker to see data.")
 
    #df = pd.read_json(chain_json, orient="split")
    from io import StringIO
    df = pd.read_json(StringIO(chain_json), orient="split")
    spot = meta["spot"]
 
    # ---- Smile: filter to selected expiry ----
    target_T = None
    if expiry:
        target_T = (pd.Timestamp(expiry) - pd.Timestamp.today()).days / 365.0
    if target_T is not None:
        smile_df = df.iloc[(df["T"] - target_T).abs().argsort()]
        smile_df = smile_df[np.isclose(smile_df["T"], smile_df["T"].iloc[0], atol=1e-3)]
    else:
        smile_df = df
 
    smile_fig = go.Figure()
    for otype, color in [("call", "#00ff22"), ("put", "#ff3333")]:
        if opt_type != "both" and opt_type != otype:
            continue
        sub = smile_df[smile_df["type"] == otype].sort_values("strike")
        if sub.empty:
            continue
        smile_fig.add_trace(go.Scatter(
            x=sub["strike"], y=sub["iv_calc"] * 100, mode="markers+lines",
            name=f"{otype.capitalize()} IV", line=dict(color=color), marker=dict(size=6),
        ))
    smile_fig.add_vline(x=spot, line_dash="dash", line_color="#888", annotation_text="Spot")
    smile_fig.update_layout(
        title=f"Volatility Smile — {expiry or ''}",
        xaxis_title="Strike ($)", yaxis_title="Implied Volatility (%)",
        template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=PANEL, height=430,
    )
 
    # ---- Surface: across all fetched expiries ----
    surf_df = df[df["type"] == (opt_type if opt_type in ("call", "put") else "call")]
    surf_fig = go.Figure()
    if len(surf_df["T"].unique()) >= 2:
        x = surf_df["moneyness"].values
        y = surf_df["T"].values * 365  # days to expiry
        z = surf_df["iv_calc"].values * 100
 
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 40),
            np.linspace(y.min(), y.max(), 40),
        )
        grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")
 
        surf_fig.add_trace(go.Surface(
            x=grid_x, y=grid_y, z=grid_z, colorscale="Viridis", showscale=True,
            contours={"z": {"show": True, "usecolormap": True, "project_z": False}},
        ))
        surf_fig.update_layout(
            title=f"Implied Volatility Surface ({opt_type})",
            scene=dict(
                xaxis_title="Moneyness (K/S)",
                yaxis_title="Days to Expiry",
                zaxis_title="IV (%)",
                bgcolor=DARK,
            ),
            template="plotly_dark", paper_bgcolor=DARK, height=430, margin=dict(l=0, r=0, t=40, b=0),
        )
    else:
        surf_fig.update_layout(
            title="Need 2+ expiries to build a surface",
            template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=PANEL, height=430,
        )
 
    # ---- Greeks table ----
    display_df = smile_df.copy()
    if opt_type in ("call", "put"):
        display_df = display_df[display_df["type"] == opt_type]
    cols = ["type", "strike", "mid", "iv_calc", "delta", "gamma", "vega", "theta", "rho"]
    display_df = display_df[cols].sort_values("strike").round(4)
    display_df = display_df.rename(columns={"iv_calc": "IV"})
    display_df["IV"] = (display_df["IV"] * 100).round(2)
 
    table = dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display_df.columns],
        style_header={"backgroundColor": "#222", "color": "#00ffcc", "fontWeight": "bold", "fontFamily": FONT},
        style_cell={"backgroundColor": PANEL, "color": "#FFF", "textAlign": "center", "fontFamily": FONT, "fontSize": "12px"},
        page_size=15,
        sort_action="native",
    )
 
    return smile_fig, surf_fig, table
 
 
if __name__ == "__main__":
    app.run(debug=True, port=8060)