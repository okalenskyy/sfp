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

       self.title:str = self.config['title'] 
       self.icon:str = self.config['icon']  
       self.subtitle:str = self.config['subtitle']  

       self.selected_ticker:str = ''
       self.selected_model:str = ''
       self.pred_days:int=1 

       self.begin_date:str = ''
       self.end_date:str = ''

       self.y_pred_df:pd.DataFrame = pd.DataFrame()
       self.y_test_df:pd.DataFrame = pd.DataFrame()
       self.sc  = {}
        
    
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

        self.models = self.config['models'].split(',')
        self.icons = self.config['model_icons'].split(',')[:len(self.models)]
        self.predict_days = self.config['predict_days']
