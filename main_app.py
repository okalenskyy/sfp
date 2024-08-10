import streamlit as st
import pandas as pd
from pathlib import Path
import yfinance as yf

from lib.models import rnn_model, lstm_horizon_model
from Classes.YahooTickerClass import YahooTicker
from Classes.UIClass import UI

from streamlit_pills import pills
from streamlit_extras.stylable_container import stylable_container

from st_on_hover_tabs import on_hover_tabs

# st.set_page_config(layout="wide")

# Importing stylesheet
# st.markdown('<style>' + open("css/styles.css").read() + '</style>', unsafe_allow_html=True)

# Set the title and favicon that appear in the Browser's tab bar.

st.set_page_config(
    page_title='Stocks Forecast Playground',
    page_icon=':chart:', # This is an emoji shortcode. Could be a URL too.
    layout='wide'
)


# -----------------------------------------------------------------------------
# Declare some useful functions.

# @st.cache_data
# def load_css(file_name):
#     with open(file_name) as f:
#         css = f.read()
#     return css


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


# def construct_sidebar():
#     with st.sidebar:
#         # tabs = on_hover_tabs(tabName=['Dashboard', 'Money', 'Economy'], 
#         #                   iconName=['dashboard', 'money', 'economy'], default_choice=0)
    
#     # Use widgets' returned values in variables

#     # -test

#         # # Load the tickers
#         # tickers_df = get_tickers()
#         # tickers = tickers_df.iloc[:,0]
#         # selected_ticker = st.selectbox('Ticker:', tickers)

#         # models = ['LSTM','Model 2','Model 3','Model 4','Model 5','Model 6','Model 7','Model 8','Model 9']
#         # selected_model = pills('Select Model', models, ["🟠","🟡","🟢","🟣","🟤","🔵","🔴","⚫","⚪"])
        
#         # # # ["🟠","🟡","🟢","🟣","🟤","🔵","🔴","⚫","⚪"]

# # -/test

#         # ok=0

#         # for i in range(int(st.number_input('Num:'))): ok=ok+1
#         # if st.sidebar.selectbox('I:',['f','j']) == 'f':
#         my_slider_val = st.slider('Prediction days', 1, 7)




# st.markdown('#  :chart: Stocks Forecasting \n Forecasting of the stock market data with different models.')
# Add some spacing
''
''

# construct_sidebar()


# if tabs =='Dashboard':
#     st.title("Navigation Bar")
#     st.write('Name of option is {}'.format(tabs))

# elif tabs == 'Money':
#     st.title("Paper")
#     st.write('Name of option is {}'.format(tabs))

# elif tabs == 'Economy':
#     st.title("Tom")
#     st.write('Name of option is {}'.format(tabs))



#----------------

# col1, col2 = st.columns([1,4])

# with col1:
#     # Select box to choose one item from the list
#     # selected_ticker = st.selectbox('Ticker:', tickers)
#     begin_date = st.date_input('Begin Date')
#     end_date = st.date_input('End Date')

# with col2:
#     ''
#     st.subheader(f'{tickers_df[tickers_df.Symbol == selected_ticker].Name.values[0]}')
#     ''
#     st.write(f'{tickers_df[tickers_df.Symbol == selected_ticker].Country.values[0]}')
#     st.write(f'IPO Year: {int(tickers_df[tickers_df.Symbol == selected_ticker].iloc[0,3])}')
#     st.write(f'Sector: {tickers_df[tickers_df.Symbol == selected_ticker].Sector.values[0]}')
#     st.write(f'Industry: {tickers_df[tickers_df.Symbol == selected_ticker].Industry.values[0]}')
    
# selected_model = pills('Select Model', models, ["🟠","🟡","🟢"])
# # ["🟠","🟡","🟢","🟣","🟤","🔵","🔴","⚫","⚪"]
# ''

# with stylable_container(
#     key="button",
#     css_styles=button_css_content,
# ):
#     if st.button('Update prediction'):
#         st.write('done')


''
''



# st.line_chart(
#         filtered_gdp_df,
#         x='Year',
#         y='GDP',
#         color='Country Code',
#     )




# ----[ NEW ]-------
def render_page():
    sfpUI = UI('main')
    st.markdown('#  :chart: Stocks Forecasting \n Forecasting of the stock market data with different models.')
    with st.sidebar:
        selected_model = pills('Select Model', sfpUI.models,sfpUI.icons)
        my_slider_val = st.slider('Prediction days', 1, sfpUI.predict_days)

