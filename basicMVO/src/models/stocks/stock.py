import uuid
import pandas as pd
import datetime
from src.common.database import Database
import src.models.stocks.constants as StockConstants
import matplotlib.pyplot as plt
import time
plt.style.use('ggplot')
import src.models.portfolios.constants as PortfolioConstants
import numpy as np
import src.models.stocks.errors as StockErrors
import quandl
quandl.ApiConfig.api_key = StockConstants.API
#http://finance.google.com/finance/historical?q=NYSEARCA:SPY&startdate=Jan+01%2C+2009&enddate=Aug+2%2C+2017&output=csv
#"http://finance.google.com/finance/historical?q=NYSEARCA:SPY&startdate=Jan+01%2C+2009&enddate="+"Nov+2%2C+2017"+"&output=csv"

class Stock(object):
    def __init__(self, ticker, returns, prices, mu, std, last_updated, info_url, _id = None):
        # Stock class creates stock instances of assets stored/allowed
        # Only needs to enter ticker name and run get_Params to fill in the rest.
        self.ticker = ticker
        self.returns = returns
        self.prices = prices
        self.mu = mu
        self.std = std
        self.last_updated = last_updated
        self.info_url = info_url
        self._id = uuid.uuid4().hex if _id is None else _id

    def __repr__(self):
        return "<Asset: {}>".format(self.ticker)

    @classmethod
    def get_Params(cls, ticker, url, info_url):
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
            #Please be noted that the link to retrieve S&P index data is different from the links to retrieve the price for other assets

            today = datetime.datetime.now()
            url = "{0}&startdate=Jan+01%2C+2007&enddate={1}+{2}%2C+{3}&output=csv".format(url, today.strftime("%b"), today.day, today.year)
            data = pd.read_csv(url)
            data.Date = pd.to_datetime(data.Date)
            last_updated = time.time()
            data = pd.DataFrame(data.Close.values, index = data.Date, columns = ['price'])

        except quandl.errors.quandl_error.NotFoundError:
            error = True

        if error is True:
            raise StockErrors.IncorrectTickerError("The ticker {} is invalid!".format(ticker))

        rets = np.negative(data.pct_change().dropna())  # create return timeseries
        rets.columns = [ticker]

        mu = rets.mean().values[0]*252
        std = rets.std().values[0]*np.sqrt(252)

        stock = cls(ticker = ticker, returns = rets.to_json(orient='index'), prices=data.to_json(), mu = mu, std = std, last_updated = last_updated, info_url=info_url)    # create instance of stock
        stock.save_to_mongo()   # save instance to db

        return stock




    def plot_stock(self):
        '''
        Plots pie chart of portfolio constituents
        :return: matplotlib matplotlib.figure.Figure object
        '''

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        #prices = []
        price_data = pd.read_json(self.prices)
        #prices.append(pd.read_json(self.prices))
        prices = np.array(price_data.price)
        index = np.array(price_data.index)

        print(type(index[0]))
        #plt.ylim(0, 80)
        #plt.plot(np.array(prices))
        #print(len(price_data.index))
        #print(price_data.columns)
        #print(np.array(price_data.index[0]))
        #print(np.array(price_data.index[1]).shape)

        plt.plot(index,prices)
        plt.tick_params(index)
        plt.title("Historical Price Chart")
        plt.xlabel("Date")
        plt.ylabel("Price in $")

        #pd.to_datetime(prices.index),
        return fig
    def save_to_mongo(self):
        Database.update(StockConstants.COLLECTION, {'ticker': self.ticker}, self.json())


    def json(self):     # Creates JSON representation of stock instance
        return{

            "ticker" : self.ticker,
            "returns" : self.returns,
            "prices": self.prices,
            "mu" : self.mu,
            "std": self.std,
            "last_updated": self.last_updated,
            "info_url": self.info_url,
            "_id": self._id
        }



    @classmethod
    def get_by_ticker(cls, stock_ticker): # Retrieves stock from MongoDB by its unique ticker
        return cls(**Database.find_one(StockConstants.COLLECTION, {'ticker': stock_ticker}))

    @classmethod
    def all(cls):   # Retrieves all stock records in MongoDB
        return [cls(**elem) for elem in Database.find(StockConstants.COLLECTION, {})]

    def remove(self):   # Removes stock from MongoDB by its unique id
        return Database.remove(StockConstants.COLLECTION, {"ticker": self.ticker})

