import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from fmp_python.fmp import FMP          # for financial API:  pip install fmp-python
import smtplib                          # for sending emails
from email.message import EmailMessage  # for sending emails
from datetime import datetime
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
from plotly.graph_objs import Line
import dash_daq as daq
from prophet import Prophet
from plotly.io import to_json
from plotly.tools import mpl_to_plotly
from dash import html
import os


load_figure_template("darkly")

# Get the current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the file path to the CSV file
csv_file_path = os.path.join(current_directory, 'data', 'technology_stocks.csv')

# Read the CSV file
ticker_list = pd.read_csv(csv_file_path)

# ticker_list = pd.read_csv('./data/technology_stocks.csv')

# def create_card(column,col_val, description):
#     card = dbc.Card(
#         dbc.CardBody(
#             [
#                 html.H4(column, id="card-title"),
#                 html.H2(col_val),
#                 html.P(description, id="card-description")
#             ]
#         )
#     )
#     return card
# Ticker = yf.Ticker('TSLA')
# marketcap_card = create_card('Marketcap',Ticker.info['marketCap'], '')



def create_candlestickchart(hist, selected_dropdown_value):
    ma1 = hist['Close'].rolling(window=10).mean()  # Calculate a 10-period moving average
    ma2 = hist['Close'].rolling(window=20).mean()  # Calculate a 20-period moving average
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / abs(loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate MACD line
    shortEMA = hist['Close'].ewm(span=12, adjust=False).mean()
    longEMA = hist['Close'].ewm(span=26, adjust=False).mean()
    MACD = shortEMA - longEMA

    # Calculate signal line
    signal = MACD.ewm(span=9, adjust=False).mean()

    # Calculate MACD histogram
    histogram = MACD - signal
    #Creating the layout of the stock chart
    fig = make_subplots(
        rows = 4,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing=None,
        row_heights = [10,2,2,2]
    )
    fig.add_trace(
    go.Candlestick(x=hist.index,
                   open=hist['Open'],
                   high=hist['High'],
                   low=hist['Low'],
                   close=hist['Close'],
                   name='candles',
                   showlegend=True),
    row=1,
    col=1
    )
    fig.update_layout(
        title={
        'text': f'{selected_dropdown_value}',
        'x': 0.5,  # Center the title horizontally
        'y': 0.9,  # Adjust this value for vertical centering
        'xanchor': 'center',  # Anchor the title to the center
        'yanchor': 'top'  # Anchor the title to the top
    },
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
     
    )
#     Adding 10 day simple moving average
    fig.add_trace(
        go.Line(x = hist.index, y = hist[f'10_ma'], name = f'10 SMA'), 
    row=1,
    col=1
    )
   # Adding 20 day simple moving average 
    fig.add_trace(
        go.Line(x = hist.index, y = hist[f'20_ma'], name = f'20 SMA'),
    row=1,
    col=1
    )
    #     Adding 10 day simple moving average
    fig.add_trace(
        go.Line(x = hist.index, y = hist[f'100_ma'], name = f'100 SMA'),
    row=1,
    col=1
    )
   # Adding 20 day simple moving average 
    fig.add_trace(
        go.Line(x = hist.index, y = hist[f'200_ma'], name = f'200 SMA'),
    row=1,
    col=1
    )
    #Adding Volume Chart
    fig.add_trace(
        go.Bar(x = hist.index, y = hist['Volume'], name = 'Volume', marker_color=['red' if close < open else 'green' for close, open in zip(hist['Close'], hist['Open'])], legendgroup = '2'), 
        row = 2,
        col = 1,
    )
    
    # Adding RSI
    fig.add_trace(
    go.Scatter(x=hist.index, y=rsi, mode='lines', name='RSI', line=dict(color='blue'), legendgroup = '3'),
    row = 3,
    col = 1,
    )
    fig.add_trace(
    go.Scatter(x = hist.index, y=[70] * len(hist),mode='lines', name = f'Overbought',line=dict(color='pink',dash='dash')),
    row = 3,
    col = 1,
    )
    fig.add_trace(
    go.Scatter(x = hist.index, y=[30] * len(hist),mode='lines', name = f'Oversold',line=dict(color='pink', dash='dash')),
    row = 3,
    col = 1,
    )
    # Plot MACD line
    fig.add_trace(
        go.Scatter(x=hist.index, y=MACD, mode='lines', name='MACD', line=dict(color='red')),
        row=4, col=1
    )

    # Plot signal line
    fig.add_trace(
        go.Scatter(x=hist.index, y=signal, mode='lines', name='Signal', line=dict(color='green')),
        row=4, col=1
    )

    # Plot MACD histogram
    fig.add_trace(
        go.Bar(x=hist.index, y=histogram, name='Histogram', marker_color=['red' if val < 0 else 'green' for val in histogram]),
        row=4, col=1
    )

    fig.update_yaxes(title_text="Volume", row=2, col=1)  # Add label to the y-axis of the volume subplot
    fig.update_yaxes(title_text="RSI", row=3, col=1)  # Add label to the y-axis of the RSI subplot
    fig.update_yaxes(title_text="MACD", row=4, col=1)  # Add label to the y-axis of the MACD subplot
    
    return fig

    
def format_marketcap(market_cap):
    if market_cap is not None:
        if market_cap >= 10**12:
            return "{:.2f} trillion".format(market_cap / 10**12) 
        elif market_cap >= 10**9:
            return "{:.2f} billion".format(market_cap / 10**9) 
        elif market_cap >= 10**9:
            return "{:.2f} million".format(market_cap / 10**6) 
    return "N/A"


def send_alert(subject, body, to):
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to

    user = 'alevinton151@gmail.com'                                                   # <-- Update here-------------------
    msg['from'] = user
    password = 'goyk lzle mymr uqxa'                                                  # <-- Update here-------------------

    # set server parameters
    server = smtplib.SMTP('smtp.gmail.com', 587) # create server variable
    server.starttls()
    server.login(user,password)
    server.send_message(msg)

    server.quit()

# Enable dark mode
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.DARKLY],suppress_callback_exceptions=True)

server = app.server
# this code represents the layout of the dashboard
app.layout = html.Div([
#     dbc.Row([
#         dbc.Col([marketcap_card]), dbc.Col([marketcap_card]), dbc.Col([marketcap_card]), dbc.Col([marketcap_card])
#     ]),
#     dbc.Row([
#         dbc.Col([marketcap_card]), dbc.Col([marketcap_card]), dbc.Col([marketcap_card]), dbc.Col([marketcap_card])
#     ]),
    
    #title
    html.H1("Stock Market Alert System", style={'textAlign': 'center'}),
    #trigger
    dcc.Interval(id='trigger', interval=1000*30), # 30 seconds
    #stock ticker dropdown menu
    html.P('Select Stock:', className = 'fix_label', style = {'color': 'white'}),
    dcc.Dropdown(id="my-dropdown", options=ticker_list['Ticker'], value='TSLA', clearable=False, style={'width': '50%','display': True}, className = 'dcc_compon'),
    #placeholder for PE, marketcap, time
    html.Div(id='price-placeholder', children=[]),
    #number of days slider
    html.Div([
    daq.Slider(id='my-daq-slider-ex-1', min=0, max=1000, value=365, marks={i: str(i) for i in range(0, 1001, 100)}, handleLabel='#days'),
    html.Div(id='slider-output-1', style={'margin-top': '20px'})  # Add margin-top to separate from the slider
]),
    
    #first plot - candlestick chart
    html.Div([
        dcc.Graph(id='Candle_Graph', style={'height': '1000px','width': '65%','display': 'inline-block'}),
        dcc.Graph(id='Earnings', style={'height': '1000px','width': '35%','display': 'inline-block'})
    ]),
    
#      html.Div([
#          dcc.Graph(id='My_Graph',  figure={},config={'editable': True}, style={'height': '500px','width': '60%','display': 'inline-block'})
#      ]),
    
    #text and email alerts
    html.Div('Would you like to set up email or phone alerts for price changes?'),
    dcc.RadioItems(id='alert-permission', options=['No','Yes, email alerts', 'Yes, phone alerts'], value='No'),
    html.Div('Alert me when share price is equal or above:'),
    dcc.Input(id='alert-value', type='number', min=0, max=10000, value=0),
    html.Hr(),
    dcc.Input(id='ticker-name', type='text', value='', style={'display': 'none'}),
    
    #second graph - intended for time series analysis
    html.H1("Prediction Chart", style={'textAlign': 'center'}),
    
    #prediction chart - forecasting with prophet model
    dcc.Graph(id='Prediction-Chart', config={'editable': True})
    # Hidden input for ticker name
], style={'width': '200'})

@app.callback(Output('price-placeholder', 'children'),[Input('my-dropdown', 'value')],[Input('trigger', 'n_intervals')])

def trigger(selected_dropdown_value,_):
    Ticker = yf.Ticker(selected_dropdown_value)
#     hist = Ticker.history(period="Max")
    current_price = round(Ticker.history(period="1d")["Close"].iloc[-1], 2)
    current_time = datetime.now().strftime("%H:%M:%S")
    pe_ratio = round(Ticker.info['forwardPE'], 2)
    market_cap = Ticker.info['marketCap']
    
    return html.Pre(f"Time: {current_time}\nStock Price: ${current_price}\nMarket Cap: {format_marketcap(market_cap)}\nP/E Ratio: {pe_ratio}")
    


@app.callback(Output('Candle_Graph', 'figure'),Output('slider-output-1', 'children'),Output('Prediction-Chart', 'figure'),Output('Earnings', 'figure'),
              [Input('my-dropdown', 'value')],[Input('my-daq-slider-ex-1', 'value')],
              [State('alert-permission', 'value'),
               State('alert-value', 'value'),])

def graph(selected_dropdown_value,value,alert_permission, alert_value):
    
    Ticker = yf.Ticker(selected_dropdown_value)
    hist = Ticker.history(period="10y")
    

    current_price = round(Ticker.history(period="1d")["Close"].iloc[-1], 2)
    pe_ratio = round(Ticker.info['forwardPE'], 2)
    market_cap = Ticker.info['marketCap']
    
    # Update x-axis range based on the slider value
    start_date = hist.index[-value]
    end_date = hist.index[-1]
    hist_filtered = hist.loc[start_date:end_date]
    hist_filtered['10_ma'] = hist_filtered['Close'].rolling(10).mean()
    hist_filtered['20_ma'] = hist_filtered['Close'].rolling(20).mean()
    hist_filtered['100_ma'] = hist_filtered['Close'].rolling(100).mean()
    hist_filtered['200_ma'] = hist_filtered['Close'].rolling(200).mean()
    
    candle_graph = create_candlestickchart(hist_filtered, selected_dropdown_value)
    if alert_permission == 'Yes, phone alerts':
        if current_price <= alert_value:
            send_alert('Alert: Buy Stock',
                        f'{selected_dropdown_value} passed your alert threshold of ${alert_value}'
                        f'and is now at ${current_price} per share.',
                        '6312521859@txt.att.net') 
    

    elif alert_permission == 'Yes, email alerts':
        if current_price <= alert_value:
            send_alert('Alert: Buy Stock',
                        f'{selected_dropdown_value} passed your alert threshold of ${alert_value}'
                        f'and is now at ${current_price} per share.',
                        'datajanitor1581@gmail.com')
            
           
    

    ma1 = hist['Close'].rolling(window=10).mean()  # Calculate a 10-period moving average
    ma2 = hist['Close'].rolling(window=20).mean()  # Calculate a 20-period moving average
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / abs(loss)
    rsi = 100 - (100 / (1 + rs))


    
    # Calculate MACD line
    shortEMA = hist['Close'].ewm(span=12, adjust=False).mean()
    longEMA = hist['Close'].ewm(span=26, adjust=False).mean()
    MACD = shortEMA - longEMA

    # Calculate signal line
    signal = MACD.ewm(span=9, adjust=False).mean()

    # Calculate MACD histogram
    histogram = MACD - signal
    
    
    #model prediction
    split_date = '2024-03-16'
    stock_train = hist.loc[hist.index <= split_date].copy()
    stock_test = hist.loc[hist.index > split_date].copy()
    train_fig = px.line(stock_train, x=stock_train.index, y='High')  
    test_fig = px.line(stock_test, x=stock_test.index, y='High') 
    
    model = Prophet()
    
    
    stock_train_prophet = stock_train.reset_index().rename(columns = {'Date':'ds', 'High':'y'})
    stock_train_prophet['ds'] = stock_train_prophet['ds'].dt.tz_localize(None)
    
    stock_test_prophet = stock_test.reset_index().rename(columns = {'Date':'ds', 'High':'y'})
    stock_test_prophet['ds'] = stock_test_prophet['ds'].dt.tz_localize(None)
    model.fit(stock_train_prophet)
#     forecast = model.predict(stock_test_prophet)
    future = model.make_future_dataframe(periods=365*24, freq='h',include_history=False)
    forecast = model.predict(future)
    history_fig = px.line(forecast, x=forecast['ds'], y='yhat')

    
#     history_fig.add_scatter(x=stock_train.index, y=stock_train['High'], mode='lines', name = 'train')
    
#     history_fig.add_scatter(x=stock_test.index, y=stock_test['High'], mode='lines', name = 'test')
    financial_plot = make_subplots(
        rows = 3,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing = 0.05,
        row_heights = [5,5,5]
    )
    
    financial_plot.update_layout(
    title_text="Financial Data",
    title_x=0.5,  # Set the title's horizontal position to the center
    title_y=0.95,  # Set the title's vertical position (optional)
    )
    #adding in earnings
    earnings = Ticker.earnings_dates
    earnings.columns = ['Estimate','Reported','Surprise(%)']
    earnings_plot = px.scatter(earnings, x=earnings.index, y='Estimate',hover_name='Estimate',title='Earnings')
    # Update title alignment
    earnings_plot.update_layout(title_x=0.5)
    earnings_plot.add_scatter(x=earnings.index, y=earnings.Reported, mode="markers",name="Reported")
    earnings_plot.add_scatter(x=earnings.index, y=earnings.Estimate, mode="markers",name="Estimate")
    
    financial_plot.add_trace(
        go.Scatter(x = earnings.index, y = earnings['Estimate'],mode='markers', name = 'Estimate'),
        row = 1,
        col = 1,
    )
    financial_plot.add_trace(
    go.Scatter(x = earnings.index, y = earnings['Reported'],mode='markers', name = 'Reported'),
        row = 1,
        col = 1,
    )
    
    #adding in the balance sheet
    balance_sheet = Ticker.balance_sheet
    balance_sheet = balance_sheet.transpose().sort_index()
    balance_sheet.columns = [x.replace(' ','_') for x in balance_sheet.columns]
    balance_sheet = balance_sheet.reset_index()
    balance_sheet['date'] = balance_sheet['index'].dt.strftime('%m-%y')


    #adding in revenue and net income
    financial_data = Ticker.financials
    financial_data = financial_data.transpose().sort_index()
    financial_data = financial_data.reset_index()
    financial_data[['Total Revenue','Net Income']]
    financial_data['date'] = financial_data['index'].dt.strftime('%m-%y')

    
    financial_plot.add_trace(
            go.Bar(x = balance_sheet.date, y = balance_sheet['Total_Assets'], name = 'Total_Assets'),
            row = 2,
            col = 1,
        )
    financial_plot.add_trace(
            go.Bar(x = balance_sheet.date, y = balance_sheet['Current_Liabilities'], name = 'Liabilities'),
            row = 2,
            col = 1,
        )
    financial_plot.add_trace(
            go.Bar(x = financial_data.date, y = financial_data['Total Revenue'], name = 'Total Revenue'),
            row = 3,
            col = 1,
        )
    financial_plot.add_trace(
            go.Bar(x = financial_data.date, y = financial_data['Net Income'], name = 'Net Income'),
            row = 3,
            col = 1,
        )
    financial_plot.update_yaxes(title_text="EPS", row=1, col=1)  
    financial_plot.update_yaxes(title_text="Balance Sheet", row=2, col=1)  
    financial_plot.update_yaxes(title_text="Income", row=3, col=1)  
    
    return candle_graph,f'\nDays to plot: {value}.',history_fig, financial_plot

    
    
                         
                         
                         





if __name__ == "__main__":
    app.run_server(debug=True)
