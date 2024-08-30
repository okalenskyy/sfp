import streamlit as st
import pandas as pd
from pathlib import Path
import yfinance as yf
import datetime

# from lib.models import rnn_model, lstm_horizon_model
# from Classes.YahooTickerClass import YahooTicker
from Classes.UIClass import UI

from streamlit_pills import pills
from streamlit_extras.stylable_container import stylable_container

from st_on_hover_tabs import on_hover_tabs

# -----------------------------------------------------------------------------
# Declare some useful functions.

def ini_UI_adapter()->UI:
    #init UI object
    sfpUI:UI = UI('main')
    return(sfpUI)

def init_page(adapter: UI):
  
    # page---------
    st.set_page_config(
        page_title=f'{adapter.title}',
        page_icon=f'{adapter.icon}', 
        layout='wide'
       )
def init_side_bar(adapter: UI):
    # sidebar------
    with st.sidebar:      
        adapter.begin_date = st.date_input('Begin Date', datetime.date(2023, 1, 1))
        adapter.end_date = st.date_input('End Date', datetime.date(2023, 12, 31))
        
        adapter.selected_ticker = st.selectbox('Ticker:', adapter.tickers)

        st.write(f'{adapter.selected_ticker_name}')
   
        adapter.selected_model = pills('Select Model', adapter.models,adapter.icons)
        adapter.pred_days = st.slider('Prediction days', 1, adapter.predict_days)


def update_data(adapter: UI):
 # run calculations ---- TODO
    # adapter.y_pred_df, adapter.y_test_df, sfpUI.sc = 
    adapter.predict()

def render_page(adapter: UI):
    # header-------
    st.markdown(f'## {adapter.icon} {adapter.title}')
    # st.markdown(f'{sfpUI.subtitle}')

    st.subheader(f'{adapter.selected_ticker_name}')

    col1, col2 = st.columns([3,1])
    with col1:
        st.write(f'{adapter.selected_ticker_sector}')
        st.write(f'{adapter.selected_ticker_industry}')
    with col2:
        st.write(f'{adapter.selected_ticker_country}')
        st.write(f'{adapter.selected_ticker_ipo_year}')
   
    # body--------
    tab_chart, tab_data, tab_model = st.tabs(["Chart", "Data", "Model"])

    with tab_chart:
        st.header("Chart")
    
    with tab_data:

        st.header("Data")
        # st.dataframe(sfpUI.y_pred_df)
        
    with tab_model:
        st.header("Model")

    # return(sfpUI)   

# sfpUI:UI = render_page()
# sfpUI.
# predict(sfpUI.selected_ticker, sfpUI.begin_date, sfpUI.end_date)

sfpUI:UI = ini_UI_adapter()
init_page(sfpUI)
init_side_bar(sfpUI)
update_data(sfpUI)
render_page(sfpUI)


