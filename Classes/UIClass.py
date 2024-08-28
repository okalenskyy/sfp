import json
from pathlib import Path
import pandas as pd

from Classes.YahooTickerClass import YahooTicker
from lib.models import rnn_model, lstm_horizon_model

class UI():
    """
    UI Adapter Class
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
        self._selected_model:str = ''
        self.pred_days:int=1 

        self._begin_date:str = ''
        self._end_date:str = ''

        # self.tickers_df:pd.DataFrame = pd.DataFrame()

        self._selected_ticker_name:str = ''
        self._selected_ticker_country:str = ''
        self._selected_ticker_ipo_year:int = ''
        self._selected_ticker_sector:str = ''
        self._selected_ticker_industry:str = ''

        self._y_pred_df:pd.DataFrame = pd.DataFrame()
        self._y_test_df:pd.DataFrame = pd.DataFrame()
        self._sc  = {}

        self._X_train:pd.DataFrame = pd.DataFrame()
        self._y_train:pd.DataFrame = pd.DataFrame() 
        self._X_test:pd.DataFrame = pd.DataFrame()
        self._y_test:pd.DataFrame = pd.DataFrame()

        self.model:object
                 
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
        self.selected_ticker_name = self.tickers_df[self.tickers_df.Symbol == value].Name.values[0]
        self.selected_ticker_country = self.tickers_df[self.tickers_df.Symbol == value].Country.values[0]
        self.selected_ticker_ipo_year = int(self.tickers_df[self.tickers_df.Symbol == value].iloc[0,3])
        self.selected_ticker_sector = self.tickers_df[self.tickers_df.Symbol == value].Sector.values[0]
        self.selected_ticker_industry = self.tickers_df[self.tickers_df.Symbol == value].Industry.values[0]

    @property
    def selected_model(self):
        return self._selected_model

    @selected_model.setter
    def selected_model(self, value):
        if not isinstance(value, str):
            raise TypeError("selected_model must be a str")
        self._selected_model = value
        match value:
            case 'LSTM':
                self.model = lstm_horizon_model(self.X_train, self.y_train)      
            case 'RNN':
                self.model = rnn_model(self.X_train)      
            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise Exception(f'Sorry, but {Value} model not yet implemented.' )
# ---------

# def setup_lstm(X_train, y_train):
#     lstm_model = lstm_horizon_model(X_train, y_train)
#     lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.3)
#     return lstm_model

# ---------
    @property
    def begin_date(self):
        return self._begin_date
    
    @begin_date.setter
    def begin_date(self, value):
        self._begin_date=value

    @property
    def end_date(self):
        return self._end_date
    
    @end_date.setter
    def end_date(self, value):
        self._end_date=value

    @property
    def selected_ticker_name(self):
        return self._selected_ticker_name
    
    @selected_ticker_name.setter
    def selected_ticker_name(self, value):
        self._selected_ticker_name=value

    @property
    def selected_ticker_country(self):
        return self._selected_ticker_country

    @selected_ticker_country.setter
    def selected_ticker_country(self, value):
        self._selected_ticker_country=value

    @property
    def selected_ticker_ipo_year(self):
        return self._selected_ticker_ipo_year

    @selected_ticker_ipo_year.setter
    def selected_ticker_ipo_year(self, value):
        self._selected_ticker_ipo_year=value

    @property
    def selected_ticker_sector(self):
        return self._selected_ticker_sector

    @selected_ticker_sector.setter
    def selected_ticker_sector(self, value):
        self._selected_ticker_sector=value

    @property
    def selected_ticker_industry(self):
        return self._selected_ticker_industry
    
    @selected_ticker_industry.setter
    def selected_ticker_industry(self, value):
        self._selected_ticker_industry=value

    @property
    def y_pred_df(self):
        return self._y_pred_df
        
    @y_pred_df.setter
    def y_pred_df(self, value):
            self._y_pred_df=value

    @property
    def y_test_df(self):
        return self._y_test_df
        
    @y_test_df.setter
    def y_test_df(self, value):
            self._y_test_df=value
        
    @property
    def sc(self):
        return self._sc
        
    @sc.setter
    def sc(self, value):
            self._sc=value
        
    @property
    def X_train(self):
        return self._X_train
        
    @X_train.setter
    def X_train(self, value):
            self._X_train=value
      
    @property
    def y_train(self):
        return self._y_train
        
    @y_train.setter
    def y_train(self, value):
            self._y_train=value

    @property
    def X_test(self):
        return self._X_test
        
    @X_test.setter
    def X_test(self, value):
            self._X_test=value

    @property
    def y_test(self):
        return self._y_test
        
    @y_test.setter
    def y_test(self, value):
            self._y_test=value

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

    def construct_sidebar(self):
        # Load the tickers
        # self.tickers_df = 
        self.get_tickers()
        self.tickers = self.tickers_df.iloc[:,0]

        self.models = self.config['models'].split(',')
        self.icons = self.config['model_icons'].split(',')[:len(self.models)]
        self.predict_days = self.config['predict_days']

    def fetch_data(self, ticker:str, start_date:str, end_date:str):
        # Create an instance of the class
        self.SelectedTicker = YahooTicker(ticker, start_date, end_date)
        # Call the fetch_data() method
        self.SelectedTicker.fetch_data()
    
    def prepare_datasets(self, training_data_percent:float = 0.2, n_lags:int=60, predict_days:int=0, target_column:str=None, extra_features=[], reshape_for_lstm:bool=False):
        self.X_train, self.y_train, self.X_test, self.y_test, self.sc = self.SelectedTicker.prepare_data_feat_step(n_lags = 1, training_data_percent = 0.3, target_column='Open',extra_features=['Close'], reshape_for_lstm=True) #.   'Close'
        
    # ////
     
    def predict(self):
        self.X_train, self.y_train, self.X_test, self.y_test, self.sc_extra = self.fetch_data(self.selected_ticker, self.begin_date, self.end_date)
        self.y_pred=self.selected_model.predict(self.X_train)
        self.y_pred=self.sc.inverse_transform(self.y_pred)
    
        #get the right scaller
        self.sc=self.sc_extra['Open']
    
        self.y_test=self.y_test.iloc[:-1]

        self.y_pred_df=pd.DataFrame({'LSTM test forecast':self.y_pred.reshape(-1),
                               'Original data':self.y_test['Open'].values},
                               index=self.y_test.index.values)
    