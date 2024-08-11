import json
from pathlib import Path
import pandas as pd

class UI():
    """
    UI handler Class
    """
    def __init__(self, id:str):
       """
       Initialise PageUI with minimum required parameters
       """
       self.id = id
       self.get_configuration()
       self.construct_sidebar()

       self.title:str = "# " + self.config['title'] 
       self.icon:str = self.config['icon']  
       self.subtitle:str = self.config['subtitle']  
    
    def get_configuration(self):
        config_file_path = Path(__file__).parent.parent/'config/config.json'
        with open(config_file_path) as json_file:
            self.config = json.load(json_file)
            
    def get_tickers(self):
        tickers_file_path = Path(__file__).parent.parent/'data/nasdaq_tickers.csv'
        # Fetch a list of stock tickers
        tickers_df = pd.read_csv(tickers_file_path)
        tickers_df.key = 'Symbol'
        return tickers_df

    def get_models_list(self):
        tickers_file_path = Path(__file__).parent/'data/nasdaq_tickers.csv'
        # Fetch a list of stock tickers
        tickers_df = pd.read_csv(tickers_file_path)
        tickers_df.key = 'Symbol'
        return tickers_df

    def construct_sidebar(self):
        # Load the tickers
        tickers_df = self.get_tickers()
        self.tickers = tickers_df.iloc[:,0]

        # selected_ticker = st.selectbox('Ticker:', tickers)
        # self.models = ['LSTM','Model 2','Model 3','Model 4','Model 5','Model 6','Model 7','Model 8','Model 9']
        self.models = self.config['models']
        self.icons = self.config['model_icons']
        self.predict_days = self.config['predict_days']

        # selected_model = pills('Select Model', models, ["🟠","🟡","🟢","🟣","🟤","🔵","🔴","⚫","⚪"])
        # # ["🟠","🟡","🟢","🟣","🟤","🔵","🔴","⚫","⚪"]