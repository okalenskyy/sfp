import streamlit as st
import pandas as pd
from pathlib import Path
import yfinance as yf

from lib.models import rnn_model, lstm_horizon_model
from Classes.YahooTickerClass import YahooTicker
from streamlit_pills import pills
from streamlit_extras.stylable_container import stylable_container

st.markdown('<style>' + open('sfp\css\styles.css').read() + '</style>', unsafe_allow_html=True)


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Stocks Analysis',
    page_icon=':chart:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def load_css(file_name):
    with open(file_name) as f:
        css = f.read()
    return css

@st.cache_data
def fetch_data(ticker, start_date, end_date):
    # Create an instance of the class
    AaplTicker = YahooTicker(ticker, start_date, end_date)
    # Call the fetch_data() method
    AaplTicker.fetch_data()

    X_train, y_train, X_test, y_test, sc_extra = AaplTicker.prepare_data_feat_step(n_lags = 1, training_data_percent = 0.3, target_column='Open',extra_features=['Close'], reshape_for_lstm=True) #.   'Close'
      
    return(X_train, y_train, X_test, y_test, sc_extra)

def select_model(model, X_train, y_train):
    match model:
        case 'LSTM':
            return(setup_lstm(X_train, y_train))         
        # If an exact match is not confirmed, this last case will be used if provided
        case _:
            return f'Sorrry, but {model} model not yet implemented.' 
        
@st.cache_data
def setup_lstm(X_train, y_train):
    lstm_model = lstm_horizon_model(X_train, y_train)
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.3)
    return lstm_model


@st.cache_data
def get_tickers():
    DATA_FILENAME = Path(__file__).parent/'data/nasdaq_tickers.csv'
  
    try:
        # Fetch a list of stock tickers
        tickers_df = pd.read_csv(DATA_FILENAME)
        tickers_df.key = 'Symbol'
        return tickers_df
    except Exception as e:
        st.error(f"An error occurred while fetching tickers: {e}")
        return []

def predict():
    tickers_dataset = fetch_data(selected_ticker, begin_date, end_date)
    model = select_model(selected_ticker, tickers_dataset[0], tickers_dataset[1])
    y_pred=model.predict(tickers_dataset[0])
    #get the right scaller
    sc=tickers_dataset[4]['Open']
    y_test=tickers_dataset[3].iloc[:-1]


    y_pred_df=pd.DataFrame({'LSTM test forecast':y_pred.reshape(-1),
                         'Original data':y_test['Open'].values},
                         index=y_test.index.values)
    y_pred_df.plot()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
#  :chart: Stocks Forecasting

Forecasting of the stock market data with different models.
'''

# Add some spacing
''
''

css_file = "css/styles.css"
css_content = load_css(css_file)

st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

models = ['LSTM','2','3']
# Load the tickers
tickers_df = get_tickers()
tickers = tickers_df.iloc[:,0]

col1, col2 = st.columns([1,4])

with col1:
    # Select box to choose one item from the list
    selected_ticker = st.selectbox('Ticker:', tickers)
    begin_date = st.date_input('Begin Date')
    end_date = st.date_input('End Date')

with col2:
    ''
    st.subheader(f'{tickers_df[tickers_df.Symbol == selected_ticker].Name.values[0]}')
    ''
    st.write(f'{tickers_df[tickers_df.Symbol == selected_ticker].Country.values[0]}')
    st.write(f'IPO Year: {int(tickers_df[tickers_df.Symbol == selected_ticker].iloc[0,3])}')
    st.write(f'Sector: {tickers_df[tickers_df.Symbol == selected_ticker].Sector.values[0]}')
    st.write(f'Industry: {tickers_df[tickers_df.Symbol == selected_ticker].Industry.values[0]}')
    
selected_model = pills('Select Model', models, ["🟠","🟡","🟢"])
# ["🟠","🟡","🟢","🟣","🟤","🔵","🔴","⚫","⚪"]
''
with stylable_container(
    key="green_button",
    css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;

            }
            """,
):
    if st.button('Run prediction'):
        st.write('done')

''
st.header(f'{selected_ticker}: hystorical data vs forecast', divider='red')

''

# st.line_chart(
#         filtered_gdp_df,
#         x='Year',
#         y='GDP',
#         color='Country Code',
#     )

