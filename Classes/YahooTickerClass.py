import yfinance as yf
import pandas as pd
import numpy as np

# import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler

# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_absolute_percentage_error
# from sklearn.metrics import root_mean_squared_error


# import matplotlib.pyplot as plt
# import seaborn as sns

class YahooTicker():
    """
    Class for fetching and processing historical stock data from Yahoo Finance API.
    """
    def __init__(self, ticker, start_date, end_date):
        """
        Initialise YahooTicker with minimum required parameters
        """
        self.ticker:str = ticker
        self.start_date:str = start_date
        self.end_date:str = end_date

        self._stock_data:pd.DataFrame = pd.DataFrame()
        self._dataset_columns:list = []
        self._dataset_size:int = 0
        self._training_data_percent:float = 0
        self._n_training_days:int = 0
        self._n_prediction_days:int = 0
        self._n_lags:int = 0
        self._target_column:str = None
        self._extra_features:list = []
        # self._sc_extra:set = []  # TODO To check

        self._scaler = {}

        # self._features_scaled:pd.DataFrame = pd.DataFrame()
        self._features_scaled = {}

        self._columns_list:list = []

        # self._dataset_train:pd.DataFrame = pd.DataFrame()
        # self._dataset_test:pd.DataFrame = pd.DataFrame()

        self._dataset_train:np.ndarray= np.empty(0)
        self._dataset_test:np.ndarray= np.empty(0)


        self._X_train:np.ndarray = np.empty(0)
        self._x_train:np.ndarray = np.empty(0)

        self._X_test:np.ndarray = np.empty(0)
        self._x_test:np.ndarray = np.empty(0)

        self._y_train:np.ndarray = np.empty(0)

        self._real_stock_price:pd.DataFrame = pd.DataFrame()

    #stock_data
    @property
    def stock_data(self):
        return self._stock_data

    @stock_data.setter
    def stock_data(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError("stock_data must be a DataFrame")
        self._stock_data = value

    @stock_data.deleter
    def stock_data(self):
        raise AttributeError("Do not delete, stock_data can be set to None")

    #dataset_size
    @property
    def dataset_size(self):
        return self._dataset_size

    @dataset_size.setter
    def dataset_size(self, value):
        if not isinstance(value, int):
            raise TypeError("dataset_size must be a int")
        self._dataset_size = value

    @dataset_size.deleter
    def dataset_size(self):
        raise AttributeError("Do not delete, dataset_size can be set to 0")

    #dataset_columns
    @property
    def dataset_columns(self):
        return self._dataset_columns

    @dataset_columns.setter
    def dataset_columns(self, value):
        if not isinstance(value, list):
            raise TypeError("dataset_size must be a list")
        self._dataset_columns = value

    @dataset_columns.deleter
    def dataset_columns(self):
        raise AttributeError("Do not delete, dataset_columns can be set to []")

    #training_data_percent
    @property
    def training_data_percent(self):
        return self._training_data_percent

    @training_data_percent.setter
    def training_data_percent(self, value):
        if not isinstance(value, float):
            raise TypeError("training_data_percent must be a float")
        if value > 1 or value < 0:
            raise ValueError("training_data_percent must be between 0 and 1.")
        self._training_data_percent = value

    @training_data_percent.deleter
    def training_data_percent(self):
        raise AttributeError("Do not delete, training_data_percent can be set to 0")

    #n_training_days
    @property
    def n_training_days(self):
        return self._n_training_days

    @n_training_days.setter
    def n_training_days(self, value):
        if not isinstance(value, int):
            raise TypeError("n_training_days must be a int")
        self._n_training_days = value
    @n_training_days.deleter
    def n_training_days(self):
        raise AttributeError("Do not delete, n_training_days can be set to 0")

   #n_prediction_days
    @property
    def n_prediction_days(self):
        return self._n_prediction_days

    @n_prediction_days.setter
    def n_prediction_days(self, value):
        if not isinstance(value, int):
            raise TypeError("n_prediction_days must be a int")
        if value ==0:
          value = 1
        self._n_prediction_days = value
    @n_prediction_days.deleter
    def n_prediction_days(self):
        raise AttributeError("Do not delete, n_prediction_days can be set to 0")

    #n_lags
    @property
    def n_lags(self):
        return self._n_lags

    @n_lags.setter
    def n_lags(self, value):
        if not isinstance(value, int):
            raise TypeError("n_lags must be a int")
        if value < 1:
            raise ValueError("Number of lags must be at least 1.")
        self._n_lags = value
    @n_lags.deleter
    def n_lags(self):
        raise AttributeError("Do not delete, n_lags can be set to 0")

    #target_column
    @property
    def target_column(self):
        return self._target_column

    @target_column.setter
    def target_column(self, value):
        if not isinstance(value, str):
            raise TypeError("target_column must be a str")
        # if value not in self.stock_data.columns:
        #     raise ValueError("Target column not found in data.")
        self._target_column = value
    @target_column.deleter
    def target_column(self):
        raise AttributeError("Do not delete, target_column can be set to None")

    #extra_features
    @property
    def extra_features(self):
        return self._extra_features

    @extra_features.setter
    def extra_features(self, value):
        if not isinstance(value, list):
            raise TypeError("extra_features must be a list")
        self._extra_features = value
    @extra_features.deleter
    def extra_features(self):
        raise AttributeError("Do not delete, extra_features can be set to None")

    #scaler
    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        # if not isinstance(value, MinMaxScaler):
        #     raise TypeError("sc_extra must be MinMaxScaler")
        self._scaler = value

    @scaler.deleter
    def scaler(self):
        raise AttributeError("Do not delete, scaler can be set to {}")

    #features_scaled
    @property
    def features_scaled(self):
        return self._features_scaled

    @features_scaled.setter
    def features_scaled(self, value):
         if not isinstance(value, pd.DataFrame):
             raise TypeError("features_scaled must be pd.DataFrame")
         self._features_scaled = value

    @features_scaled.deleter
    def features_scaled(self):
        raise AttributeError("Do not delete, features_scaled can be set to {}")

    #columns_list
    @property
    def columns_list(self):
        return self._columns_list

    @columns_list.setter
    def columns_list(self, value):
        if not isinstance(value, list ):
            raise TypeError("columns_list must be list")
        self._columns_list = value

    @columns_list.deleter
    def columns_list(self):
        raise AttributeError("Do not delete, columns_list can be set to []")

    #dataset_train
    @property
    def dataset_train(self):
        return self._dataset_train

    @dataset_train.setter
    def dataset_train(self, value):
        # if not isinstance(value, array):
        #     raise TypeError("dataset_train must be array")
        self._dataset_train = value

    @dataset_train.deleter
    def dataset_train(self):
        raise AttributeError("Do not delete, dataset_train can be set to None")

    #dataset_test
    @property
    def dataset_test(self):
        return self._dataset_test

    @dataset_test.setter
    def dataset_test(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f'dataset_test must be pd.DataFrame, but the type is {type(value)}')
        self._dataset_test = value

    @dataset_test.deleter
    def dataset_test(self):
        raise AttributeError("Do not delete, dataset_test can be set to None")

    #X_train
    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        # if not isinstance(value, array):
        #     raise TypeError("X_train must be array")
        self._X_train = value

    @X_train.deleter
    def X_train(self):
        raise AttributeError("Do not delete, X_train can be set to []")

    #x_train
    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, value):
        # if not isinstance(value, array):
        #     raise TypeError("x_train must be array")
        self._x_train = value

    @x_train.deleter
    def x_train(self):
        raise AttributeError("Do not delete, x_train can be set to []")

    #x_test
    @property
    def x_test(self):
        return self._x_test

    @x_test.setter
    def x_test(self, value):
        # if not isinstance(value, array):
        #     raise TypeError("x_test must be array")
        self._x_test = value

    @x_test.deleter
    def x_test(self):
        raise AttributeError("Do not delete, x_test can be set to []")

    #X_test
    @property
    def X_test(self):
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        # if not isinstance(value, array):
        #     raise TypeError("X_test must be array")
        self._X_test = value

    @X_test.deleter
    def X_test(self):
        raise AttributeError("Do not delete, X_test can be set to []")

    #y_train
    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        # if not isinstance(value, array):
        #     raise TypeError("y_train must be darray")
        self._y_train = value

    @y_train.deleter
    def y_train(self):
        raise AttributeError("Do not delete, y_train can be set to []")

    #real_stock_price
    @property
    def real_stock_price(self):
        return self._real_stock_price

    @real_stock_price.setter
    def real_stock_price(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError("real_stock_price must be DataFrame")
        self._real_stock_price = value

    @real_stock_price.deleter
    def real_stock_price(self):
        raise AttributeError("Do not delete, real_stock_price can be set to []")

    # +------------------------------------+
    # |        Methods                     |
    # +------------------------------------+
    def fetch_data(self):
        """
        Fetch historical data from Yahoo Finance API.
        """
        self.stock_data:pd.DataFrame = pd.DataFrame(yf.download(self.ticker, start=self.start_date, end=self.end_date))
        self.dataset_size = len(self.stock_data)
        self.dataset_columns = self.stock_data.columns.tolist()

    def _split_dataset(self)->None:
        """
        Split the dataset into training and testing sets.
        Protected method
        """
        self.n_training_days = int(self.dataset_size * self.training_data_percent)
        # self.dataset_train, self.dataset_test = pd.DataFrame(self.stock_data.iloc[:self.n_training_days]), pd.DataFrame(self.stock_data.iloc[self.n_training_days:])
        self.dataset_train, self.dataset_test = self.stock_data.iloc[:self.n_training_days], self.stock_data.iloc[self.n_training_days:]

    def _scale_features(self)->None:
        """
        Scale the features using MinMaxScaler.
        Protected method
        """
        for f in self.columns_list:
            self.scaler[f] = MinMaxScaler(feature_range=(0, 1))
            self.features_scaled[f] = self.scaler[f].fit_transform(self.dataset_train[[f]].values)

    def _create_data_structure(self)->None:
        """
        Create a data structure with n_lags timesteps and step output.
        Protected method
        """
        X_train = []
        y_train = []
        for i in range(self.n_lags, len(self.features_scaled[self.target_column])-self.n_prediction_days):
            x_train=self.features_scaled[self.target_column][i-self.n_lags:i].squeeze()

            for f in self.extra_features:
                x_train = np.hstack((x_train, self.features_scaled[f][i-self.n_lags:i].squeeze()))

            X_train.append(x_train)
            y_train.append(self.features_scaled[self.target_column][i + self.n_prediction_days])
            # y_train.append(self.dataset_train[self.target_column][i + self.n_prediction_days])

        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

    def _prepare_test_set(self)->None:
        """
        Prepare the test set for evaluation.
        Protected method
        """

        self.real_stock_price = self.dataset_test[self.columns_list]
        dataset_total = pd.concat((self.dataset_train[self.columns_list], self.dataset_test[self.columns_list]), axis=0)
        dataset_total = dataset_total[len(dataset_total)-len(self.dataset_test)-self.n_lags:]

        for f in self.columns_list:
            dataset_total[f] = self.scaler[f].transform(dataset_total[[f]])

        X_test = []
        x_test = []

        for i in range(self.n_lags, len(dataset_total)-self.n_prediction_days):
          x_test = dataset_total[self.target_column][i-self.n_lags:i].values
          for f in self.extra_features:
              x_test = np.hstack((x_test, dataset_total[f][i - self.n_lags:i].values))
          X_test.append(x_test)
        X_test = np.array(X_test)

        self.X_test = X_test


    def _reshape_for_lstm(self)->None:
        """
        Reshape the data for LSTM.
        Protected method
        """
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.n_lags, len(self.columns_list)))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.n_lags, len(self.columns_list)))
        print(self.X_train[0])
        print(self.X_train[1])
        


    def prepare_data_feat_step(self, training_data_percent:float = 0.2, n_lags:int=60, predict_days:int=0, target_column:str=None, extra_features=[], reshape_for_lstm:bool=False)->tuple:
        """
        Prepare the data for training and testing.

        Prepares data for LSTM model considering a window of n_lags past observations to create X_train and X_test.
        y_train is prepared for long-term forecasting with a specified forecast horizon.
        IMPUT Arg:
            training_data_percent (float)  - define the %% of data to be used for training. The value is strictly between 0 and 1, default 0.2
            n_lags (int) - Number of past observations to use as features. Default value = 60
            predict_days (int) - Required number of days to be forecasted. Default value = 0 - the forecast for the next day
            target_columnn (str) - The name of the target column in the dataset. This argument is mandatory.
            extra_features (array) - List of extra features to be included in the dataset. Default value = []
        OUTPUT:
            tuple
                X_train (pd.DataFrame) Training features
                y_train (pd.DataFrame) Training targets
                X_test (pd.DataFrame)  Testing features
                real_stock_price (pd.DataFrame) Real stock prices for comparison
                sc_target (MinMaxScaler) Object used for scaling the target column
        """

        self.training_data_percent = training_data_percent
        self.n_lags = n_lags
        self.n_prediction_days = predict_days
        self.target_column = target_column
        self.extra_features = extra_features
        self.columns_list = [self.target_column] + self.extra_features

        self._split_dataset()
        self._scale_features()
        self._create_data_structure()
        self._prepare_test_set()

        if reshape_for_lstm:
            self._reshape_for_lstm()
       
        return(self.X_train, self.y_train, self.X_test, self.real_stock_price, self.scaler)
