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

        self._selected_ticker:str = ''
        self.selected_model:str = ''
        self.pred_days:int=1 

        self.begin_date:str = ''
        self.end_date:str = ''

        # self.tickers_df:pd.DataFrame = pd.DataFrame()

        self._selected_ticker_name:str = ''
        self._selected_ticker_country:str = ''
        self._selected_ticker_ipo_year:int = ''
        self._selected_ticker_sector:str = ''
        self._selected_ticker_industry:str = ''

        self.y_pred_df:pd.DataFrame = pd.DataFrame()
        self.y_test_df:pd.DataFrame = pd.DataFrame()
        self.sc  = {}
        
    def _get_ticker_property(self, ):
        return()
        #selected_ticker
    
    @property
    def selected_ticker(self):
        return self._selected_ticker

    @selected_ticker.setter
    def selected_ticker(self, value):
        if not isinstance(value, str):
            raise TypeError("selected_ticker must be a str")
        self._selected_ticker = value
        self._selected_ticker_name:str = self.tickers_df[self.tickers_df.Symbol == value].Name.values[0]
        self._selected_ticker_country:str = self.tickers_df[self.tickers_df.Symbol == value].Country.values[0]
        self._selected_ticker_ipo_year:int = int(self.tickers_df[self.tickers_df.Symbol == value].iloc[0,3])
        self._selected_ticker_sector:str = self.tickers_df[self.tickers_df.Symbol == value].Sector.values[0]
        self._selected_ticker_industry:str = self.tickers_df[self.tickers_df.Symbol == value].Industry.values[0]

    @property
    def selected_ticker_name(self):
        return self._selected_ticker_name
    
    @property
    def selected_ticker_country(self):
        return self._selected_ticker_country
    
    @property
    def selected_ticker_ipo_year(self):
        return self._selected_ticker_ipo_year

    @property
    def selected_ticker_sector(self):
        return self._selected_ticker_sector

    @property
    def selected_ticker_industry(self):
        return self._selected_ticker_industry
      
    def get_configuration(self):
        config_file_path = Path(__file__).parent.parent/'config/config.json'
        with open(config_file_path) as json_file:
            self.config = json.load(json_file)
            
    def get_tickers(self):
        tickers_file_path = Path(__file__).parent.parent/'data/nasdaq_tickers.csv'
        # Fetch a list of stock tickers
        self.tickers_df = pd.read_csv(tickers_file_path)
        self.tickers_df.key = 'Symbol'
        return self.tickers_df

    # def get_models_list(self):
    #     tickers_file_path = Path(__file__).parent/'data/nasdaq_tickers.csv'
    #     # Fetch a list of stock tickers
    #     tickers_df = pd.read_csv(tickers_file_path)
    #     tickers_df.key = 'Symbol'
    #     return tickers_df

    def construct_sidebar(self):
        # Load the tickers
        # self.tickers_df = 
        self.get_tickers()
        self.tickers = self.tickers_df.iloc[:,0]

        self.models = self.config['models'].split(',')
        self.icons = self.config['model_icons'].split(',')[:len(self.models)]
        self.predict_days = self.config['predict_days']
