import streamlit as st
import pandas as pd
from pathlib import Path
import yfinance as yf

# from lib.models import rnn_model, lstm_horizon_model
# from Classes.YahooTickerClass import YahooTicker
from Classes.UIClass import UI

from streamlit_pills import pills
from streamlit_extras.stylable_container import stylable_container

from st_on_hover_tabs import on_hover_tabs

# -----------------------------------------------------------------------------
# Declare some useful functions.

def predict(ticker, model, begin_date, end_date):
    # tickers_dataset = fetch_data(ticker, begin_date, end_date)
    X_train, y_train, X_test, y_test, sc_extra = fetch_data(ticker, begin_date, end_date)
    # model = select_model(model, tickers_dataset[0], tickers_dataset[1])
    
    model = select_model(model, X_train, y_train)
    

    # y_pred=model.predict(tickers_dataset[0])
    y_pred=model.predict(X_train)
    
    #get the right scaller
    # sc=tickers_dataset[4]['Open']
    sc=sc_extra['Open']
    

    # y_test=tickers_dataset[3].iloc[:-1]
    y_test=y_test.iloc[:-1]

    y_pred_df=pd.DataFrame({'LSTM test forecast':y_pred.reshape(-1),
                         'Original data':y_test['Open'].values},
                         index=y_test.index.values)
    return y_pred_df, y_test, sc

# ----[ NEW ]-------
def render_page():
    #init UI object
    sfpUI = UI('main')
    # page---------
    st.set_page_config(
        page_title=f'{sfpUI.title}',
        page_icon=f'{sfpUI.icon}', 
        layout='wide'
       )
    
    # sidebar------
    with st.sidebar:

        sfpUI.begin_date = st.date_input('Begin Date')
        sfpUI.end_date = st.date_input('End Date')
        
        sfpUI.selected_ticker = st.selectbox('Ticker:', sfpUI.tickers)

        st.write(f'{sfpUI.selected_ticker_name}')
   
        sfpUI.selected_model = pills('Select Model', sfpUI.models,sfpUI.icons)
        sfpUI.pred_days = st.slider('Prediction days', 1, sfpUI.predict_days)

    # header-------
    st.markdown(f'## {sfpUI.icon} {sfpUI.title}')
    # st.markdown(f'{sfpUI.subtitle}')
    

    st.subheader(f'{sfpUI.selected_ticker_name}')

    col1, col2 = st.columns([3,1])
    with col1:
        st.write(f'{sfpUI.selected_ticker_sector}')
        st.write(f'{sfpUI.selected_ticker_industry}')
    with col2:
        st.write(f'{sfpUI.selected_ticker_country}')
        st.write(f'{sfpUI.selected_ticker_ipo_year}')
    
    
    # run calculations ----
    sfpUI.y_pred_df, sfpUI.y_test_df, sfpUI.sc = predict(sfpUI.selected_ticker, sfpUI.selected_model,'2024-01-01', '2024-08-01')
    # body--------
    tab_chart, tab_data, tab_model = st.tabs(["Chart", "Data", "Model"])

    with tab_chart:
        st.header("Chart")
    
    with tab_data:
        st.header("Data")
        # st.dataframe(sfpUI.y_pred_df)
        
    with tab_model:
        st.header("Model")

    return(sfpUI)   

sfpUI:UI = render_page()

# predict(sfpUI.selected_ticker, sfpUI.begin_date, sfpUI.end_date)


