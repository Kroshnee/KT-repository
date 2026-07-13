import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import requests
import plotly.graph_objects as go

# Initialize Dash application
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io'])

# Global token - Kraken uses XBTUSD for Bitcoin to USD
SYMBOL = "XBTUSD"

# Define dark terminal-style layout
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'padding': '20px', 'fontFamily': 'monospace'}, children=[
    html.H1(f"📊 Real-Time Market Order Book Visualiser ({SYMBOL})", style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # 500ms reactive update ticker
    dcc.Interval(id='interval-component', interval=500, n_intervals=0),
    
    # Top Metrics Bar
    html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}, children=[
        html.Div([html.H4("💡 Mid Price", style={'color': '#888'}), html.H2(id='mid-price', style={'color': '#00ffcc'}, children="$0.00")]),
        html.Div([html.H4("↔️ Bid-Ask Spread", style={'color': '#888'}), html.H2(id='spread', style={'color': '#ff9900'}, children="$0.00")]),
        html.Div([html.H4("🔥 Top Ask Liquidity", style={'color': '#888'}), html.H2(id='liquidity', style={'color': '#ff3366'}, children="0.0000 XBT")])
    ]),
    
    # Plotly Graph wrapper
    html.Div([
        dcc.Graph(id='depth-graph', config={'displayModeBar': False})
    ], style={'marginBottom': '30px'}),
    
    # Data columns
    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            html.H3("🟢 Top Bids (Buys)", style={'color': '#00ff22'}),
            html.Div(id='bids-table-container')
        ]),
        html.Div(className='six columns', children=[
            html.H3("🔴 Top Asks (Sells)", style={'color': '#ff3333'}),
            html.Div(id='asks-table-container')
        ])
    ])
])

# Robust data extraction from unrestricted Kraken REST architecture
def fetch_order_book(symbol=SYMBOL, limit=20):
    url = f"https://api.kraken.com/0/public/Depth?pair={symbol}&count={limit}"
    try:
        response = requests.get(url, timeout=4)
        response.raise_for_status()
        data = response.json()

        if data.get('error'):
            print("⚠️ Kraken API error:", data['error'])
            return pd.DataFrame(), pd.DataFrame()

        if 'result' in data:
            # Kraken returns the result keyed by its internal pair name
            pair_key = next(iter(data['result']))
            result = data['result'][pair_key]
            bids = pd.DataFrame(result['bids'], columns=['Price', 'Quantity', 'Timestamp'], dtype=float)
            asks = pd.DataFrame(result['asks'], columns=['Price', 'Quantity', 'Timestamp'], dtype=float)
            return bids[['Price', 'Quantity']], asks[['Price', 'Quantity']]
        else:
            print("⚠️ API payload missing fields:", data)
            return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"❌ FETCH EXCEPTION: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Main state loop callback logic
@app.callback(
    [Output('mid-price', 'children'),
     Output('spread', 'children'),
     Output('liquidity', 'children'),
     Output('depth-graph', 'figure'),
     Output('bids-table-container', 'children'),
     Output('asks-table-container', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_market_data(n):
    bids, asks = fetch_order_book()
    
    # CRUCIAL GUARD: If network fails, do NOT push empty traces; stop update lifecycle
    if bids.empty or asks.empty or 'Price' not in bids.columns:
        return dash.no_update
    
    # Sort and process depths securely
    bids = bids.sort_values(by='Price', ascending=False)
    asks = asks.sort_values(by='Price', ascending=True)
    
    bids['Cumulative Volume'] = bids['Quantity'].cumsum()
    asks['Cumulative Volume'] = asks['Quantity'].cumsum()
    
    # Calculate spreads using precise positional scalar indexing .iloc[]
    best_bid = float(bids['Price'].iloc[0])
    best_ask = float(asks['Price'].iloc[0])
    spread = best_ask - best_bid
    mid_price = (best_bid + best_ask) / 2
    top_ask_liq = float(asks['Quantity'].iloc[0])
    
    # Build complete Plotly layout structure dynamically
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bids['Price'], y=bids['Cumulative Volume'],
        fill='tozeroy', mode='lines', name='Bids (Demand)', line=dict(color='#00ff22', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=asks['Price'], y=asks['Cumulative Volume'],
        fill='tozeroy', mode='lines', name='Asks (Supply)', line=dict(color='#ff3333', width=2)
    ))
    
    fig.update_layout(
        title="Microstructure Order Book & Liquidity Walls",
        xaxis_title="Price ($)", yaxis_title="Cumulative Depth Volume",
        template="plotly_dark", paper_bgcolor='#111111', plot_bgcolor='#1a1a1a',
        margin=dict(l=40, r=40, t=40, b=40), height=400
    )
    
    # Convert DataFrames into Dash presentation tables
    bids_table = dash_table.DataTable(
        data=bids[['Price', 'Quantity']].head(10).to_dict('records'),
        columns=[{"name": i, "id": i} for i in ['Price', 'Quantity']],
        style_header={'backgroundColor': '#222', 'color': '#00ff22', 'fontWeight': 'bold', 'fontFamily': 'monospace'},
        style_cell={'backgroundColor': '#111', 'color': '#FFF', 'textAlign': 'center', 'fontFamily': 'monospace'}
    )
    
    asks_table = dash_table.DataTable(
        data=asks[['Price', 'Quantity']].head(10).to_dict('records'),
        columns=[{"name": i, "id": i} for i in ['Price', 'Quantity']],
        style_header={'backgroundColor': '#222', 'color': '#ff3333', 'fontWeight': 'bold', 'fontFamily': 'monospace'},
        style_cell={'backgroundColor': '#111', 'color': '#FFF', 'textAlign': 'center', 'fontFamily': 'monospace'}
    )
    
    return (
        f"${mid_price:,.2f}", 
        f"${spread:,.2f}", 
        f"{top_ask_liq:.4f} XBT", 
        fig, 
        bids_table, 
        asks_table
    )

if __name__ == '__main__':
    # Modern Dash startup sequence running cleanly on port 8050
    app.run(debug=True, port=8050)




