import uuid
import pandas as pd
import datetime
from src.common.database import Database
import src.models.stocks.constants as StockConstants
import src.models.stocks.errors as StockErrors
import quandl
quandl.ApiConfig.api_key = StockConstants.API
#http://finance.google.com/finance/historical?q=NYSEARCA:SPY&startdate=Jan+01%2C+2009&enddate=Aug+2%2C+2017&output=csv
"http://finance.google.com/finance/historical?q=NYSEARCA:SPY&startdate=Jan+01%2C+2009&enddate="+"Nov+2%2C+2017"+"&output=csv"

class Stock(object):
    def __init__(self, ticker, returns, mu, std, _id = None):
        # Stock class creates stock instances of assets stored/allowed
        # Only needs to enter ticker name and run get_Params to fill in the rest.
        self.ticker = ticker
        self.returns = returns
        self.mu = mu
        self.std = std
        self._id = uuid.uuid4().hex if _id is None else _id

    def __repr__(self):
        return "<Asset: {}>".format(self.ticker)

    @classmethod
    def get_Params(cls, ticker,):
        '''
        Gets ticker data from Quandl API and saves stock to database

        :param ticker: {type:string} Asset Ticker (ex: 'AAPL')
        :param start_date: {type:string} time-series start date (ex: YYYY-MM-DD '2006-01-01')
        :param end_date: {type:string} time-series end date (ex: YYYY-MM-DD '2006-01-01')
        :return: Stock instance
        '''

        error = False
        try:
            # sets path to eleventh column (adjusted closing) of WIKI EOD table/ ticker
            #code = StockConstants.TABLE + ticker + '.11'
            # retrieve data from Quandl API [start_date, end_date] aggregated monthly
            #data = quandl.get(code, start_date=start_date, end_date=end_date, collapse=StockConstants.COLLAPSE)
            today = datetime.datetime.now()
            url = "http://finance.google.com/finance/historical?q=NYSEARCA:SPY&startdate=Jan+01%2C+2009&enddate={0}+{1}%2C+{2}&output=csv".format(today.strftime("%b"), today.day, today.year)
            data = pd.read_csv(url)
        except quandl.errors.quandl_error.NotFoundError:
            error = True

        if error is True:
            raise StockErrors.IncorrectTickerError("The ticker {} is invalid!".format(ticker))

        rets = data.pct_change().dropna()   # create return timeseries
        rets.columns = [ticker]

        mu = rets.mean().values[0]
        std = rets.std().values[0]

        stock = cls(ticker = ticker, returns = rets.to_json(orient='index'), mu = mu, std = std)    # create instance of stock
        stock.save_to_mongo()   # save instance to db

        return stock

    def save_to_mongo(self):
        Database.update(StockConstants.COLLECTION,{'_id':self._id},self.json())

    def json(self):     # Creates JSON representation of stock instance
        return{
            "_id" : self._id,
            "ticker" : self.ticker,
            "returns" : self.returns,
            "mu" : self.mu,
            "std": self.std
        }

    @classmethod
    def get_by_id(cls, stock_id):        # Retrieves stock from MongoDB by its unique id
        return cls(**Database.find_one(StockConstants.COLLECTION,{'_id' : stock_id}))

    @classmethod
    def get_by_ticker(cls, stock_ticker): # Retrieves stock from MongoDB by its unique ticker
        return cls(**Database.find_one(StockConstants.COLLECTION, {'ticker': stock_ticker}))

    @classmethod
    def all(cls):   # Retrieves all stock records in MongoDB
        return [cls(**elem) for elem in Database.find(StockConstants.COLLECTION, {})]

    def remove(self):   # Removes stock from MongoDB by its unique id
        return Database.remove(StockConstants.COLLECTION, {"_id": self._id})
