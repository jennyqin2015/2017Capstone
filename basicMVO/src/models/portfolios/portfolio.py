import datetime
import time
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')
import cvxpy as cvx
from src.common.database import Database
import src.models.portfolios.constants as PortfolioConstants
from src.models.stocks.stock import Stock
from src.models.logic.business_logic import stochastic
import time
import random
class Portfolio(object):
    # Portfolio class creates portfolio instances for user portfolios using stocks in Stock class

    def __init__(self, user_email, description, amount,initial_deposit, years, importance, risk_appetite, start_time = None, tickers = None, weights = None, quarters = None, all_stock_prices = None, rets = None, account_balance = None, _id=None):
        self.user_email = user_email
        self.description = description
        self.amount = amount # target amount of money to achieve by the terminal date of this goal
        self.initial_deposit = initial_deposit
        self.years = years
        self.importance = importance
        self.risk_appetite = risk_appetite
        self.start_time = start_time # this stores the starting time of the user's portfolio in the format of seconds
        self.tickers = PortfolioConstants.TICKERS if tickers is None else tickers
        self.weights = weights
        self.quarters = quarters
        self.all_stock_prices = all_stock_prices
        self.account_balance = account_balance
        self.rets = rets
        #self.optimized_result = 0 if optimized_result is None else optimized_result
        self._id = uuid.uuid4().hex if _id is None else _id

    def __repr__(self):
        return "<Portfolio for user {}>".format(self.user_email)


    def get_Params(self):
        '''
        Checks MongoDB stocks collection to see if assets already exist. If not, runs Stock.get_Params to
        get asset return time series from Quandl. Once all time series are collected, computes vector of
        expected returns and the covariance matrix

        :param start_date: time-series start date as string format (ex: YYYY-MM-DD '2006-01-01')
        :param end_date: time-series start date as string format (ex: YYYY-MM-DD '2016-12-31')
        :param tickers: list of tickers to retrieve

        :return: Expected Return, Covariance Matrix, Variance, Standard Deviations
        '''

        tickers = self.tickers.copy()
        urls = PortfolioConstants.URLS
        n = len(tickers)
        rets = []
        mu = []
        for i, ticker in enumerate(tickers):
            try:
                stock = Stock.get_by_ticker(stock_ticker = ticker) # Check if stock exists in db
                if stock.last_updated != time.time():
                    #update the database when the user logs in another day
                    stock = Stock.get_Params(ticker=ticker, url=urls[i])
                #if yes, update information in stock
                #stock.update_data()
            except:                                                   # If not, get params from Quandl
                stock = Stock.get_Params(ticker = ticker, url = urls[i])

            rets.append(pd.read_json(stock.returns, orient='index'))
            mu.append(stock.mu)
            returns = pd.concat(rets, axis=1)

        mu = np.array(mu).reshape([n, 1])
        cov = returns.cov()
        cov = cov.values

        return mu, cov
    def get_month_stock(self, ticker, start_time, timestamps):
            '''
            :param ticker:
            :param start_time:
            :return: a list with 4 stock prices for all 4 quarters
            '''
            stock = Stock.get_by_ticker(stock_ticker=ticker)
            prices = pd.read_json(stock.prices)
            list = prices.index
            print(list)

            # time_to_find = []
            # for i in timestamps:
            #     time_to_find.append(datetime.datetime.fromtimestamp(i))

            market_prices = []
            #print(prices.index)
            #print(os.getcwd())
            prices.to_csv("common/price_vti.csv")
            #print(ticker)
            print(prices.loc["2014-01-02"])
            print(timestamps)
            new_stamps = []

            for i in timestamps:
                #i is a datetime object
                time_key = datetime.datetime.fromtimestamp(i)
                #print(time_key)
                #print(type(time_key))
                #time_key.timestamp()
                #print(time_key)
                try:
                    market_price = prices.loc[time_key]


                    market_price = np.array(market_price)

                    market_prices.append(market_price[0])
                except KeyError:
                    '''
                    random_num = 3
                    day = datetime.datetime.fromtimestamp(i).day + random_num
                    year = datetime.datetime.fromtimestamp(i).year
                    month = datetime.datetime.fromtimestamp(i).month
                    time_key = '{0}-{1}-{2}'.format(year,month,day)
                    market_price = prices.loc[time_key]
                    market_prices.append(market_price)
                    '''
                    search_key = []
                    date_list = [x for x in range(50)]

                    for j in date_list:
                        # day = datetime.datetime.fromtimestamp(i).day + j
                        # year = datetime.datetime.fromtimestamp(i).year
                        # month = datetime.datetime.fromtimestamp(i).month
                        diff_in_seconds = datetime.timedelta(days=j).total_seconds()
                        key = i + diff_in_seconds*j
                        data_time_key = datetime.datetime.fromtimestamp(key)
                        day = data_time_key.day
                        year = data_time_key.year
                        month = data_time_key.month
                        time_key = '{0}-{1}-{2}'.format(year, month, day)
                        data_time_key = datetime.datetime.strptime(time_key, "%Y-%m-%d")


                        search_key.append(data_time_key)

                    print(search_key)
                    #print(prices.loc["2014-01-09"])
                    #print(prices.index)
                    market_price = prices.loc[search_key]

                    market_price = np.array(market_price)
                    market_price = market_price[~np.isnan(market_price)]
                    # print(market_price)
                    market_prices.append(market_price[0])

            # print(market_prices)
            return market_prices


    def get_month_index(self, start_time, diff_month):
        #this function creates the index

        '''
        imagine one user created his goal on the first day of 2014, and 2 years have past. Our model should be able to return the weight allocation
        by the end of each season. Therefore, our model should provided 7 weight allocations by now (2017 Nov 18th).
        '''
        market_prices = []
        #market_index = Stock.get_by_ticker('SNP')
        #prices = pd.read_json(market_index.prices)
        data = pd.read_csv('common/SNP_Price.csv')
        prices = pd.DataFrame(data.Close.values, index=data.Date, columns=['price'])
        #print(prices)
        ''' this is for actual use of the model
        start_date = self.start_time
        '''
        '''
        set start_date to be the time
        '''
        #datetime_time = datetime.datetime.strptime(str_time, "%Y-%m-%d")
        start_date = start_time.timestamp()
        diff_in_seconds = datetime.timedelta(days = 31).total_seconds()
        time_stamps = []
        for i in range(diff_month):
            s = start_date + i*diff_in_seconds
            time_stamps.append(s)

        # s1, s2, s3, s4 are the four timestamps for first, second, third quarters
        # s1 = start_date
        # s2 = s1 + diff_in_seconds
        # s3 = s2 + diff_in_seconds
        # s4 = s3 + diff_in_seconds
        # time_stamps = [s1, s2, s3, s4]
        #datetime_ls = []
        # for i in time_stamps:
        #     datetime_ls.append(datetime.datetime.fromtimestamp(i).strftime('%Y-%m-%d'))
        #print(datetime_ls)
        for i in time_stamps:
            time_key = datetime.datetime.fromtimestamp(i).strftime('%Y-%m-%d')
            print(time_key)
            #print(time_key)
            try:
                market_price = prices.loc[time_key]


                market_price = np.array(market_price)

                market_prices.append(market_price[0])
            except KeyError:
                '''
                random_num = 3
                day = datetime.datetime.fromtimestamp(i).day + random_num
                year = datetime.datetime.fromtimestamp(i).year
                month = datetime.datetime.fromtimestamp(i).month
                time_key = '{0}-{1}-{2}'.format(year,month,day)
                market_price = prices.loc[time_key]
                market_prices.append(market_price)
                '''
                search_key = []
                date_list = [x for x in range(100)]
                for j in date_list:
                        # day = datetime.datetime.fromtimestamp(i).day + j
                        # year = datetime.datetime.fromtimestamp(i).year
                        # month = datetime.datetime.fromtimestamp(i).month
                        diff_in_seconds = datetime.timedelta(days=j).total_seconds()
                        time_key = i + diff_in_seconds
                        time_key = datetime.datetime.fromtimestamp(time_key).strftime('%Y-%m-%d')
                        #print(time_key)
                        search_key.append(time_key)

                print(search_key)
                market_price = prices.loc[search_key]


                market_price = np.array(market_price)
                market_price = market_price[~np.isnan(market_price)]
                #print(market_price)
                market_prices.append(market_price[0])

        #print(market_prices)
        return market_prices, time_stamps # market_prices returns a list which contains 4 prices

    '''
    def compute_r_q(self):
        if self.importance ==
    '''


    def get_market_index(self,start_date, diff_year):
        market_index = []
        start_year = start_date.year
        for i in range(diff_year):
            market_prices = self.get_one_year_index(start_date)
            market_index.append(market_prices)
            start_year+=1
            start_date = start_date.replace(year = start_year)
        market_index = np.array(market_index)
        market_index = market_index.reshape(market_index.size)
        #print(market_index.size)
        return market_index

    def get_stock_price(self, ticker, start_date, diff_year):
        market_index = []
        start_year = start_date.year
        for i in range(diff_year):
            market_prices = self.get_one_year_stock(ticker, start_date)
            market_index.append(market_prices)
            start_year+=1
            start_date = start_date.replace(year = start_year)
        market_index = np.array(market_index)
        market_index = market_index.reshape(market_index.size)
        #print(market_index.size)
        return market_index

    def run_logic(self):
        str_time = self.start_time

        datetime_time = datetime.datetime.strptime(str_time, "%Y-%m-%d")
        now_date = datetime.datetime.now()
        diff = now_date - datetime_time

        #diff_year = int(diff.days/365)
        diff_month = int(diff.days/31)
        int(self.years)/5





        #if self.years <= 5:
        #change diff_year to diff_month, and multiplied year with 12

        if diff_month > int(self.years)*12:
            diff_month = int(self.years)*12
            print(diff_month)
        else:
            diff_month = diff_month
        market_list, timestamps = self.get_month_index(datetime_time, diff_month) #this returns a list with market indexes for all months
        diff_quarters = int(diff_month/3)
        market_index = []

        start_i = 0
        while start_i < len(market_list):
            #print("this is added element: {}".format(market_list[i*3]))
            market_index.append(market_list[start_i])
            start_i += 3

        print(market_index)
        #print(len(market_index))
        tickers = self.tickers.copy()
        urls = PortfolioConstants.URLS
        rets = []

        for i, ticker in enumerate(tickers):

            try:
                stock = Stock.get_by_ticker(stock_ticker=ticker)  # Check if stock exists in db
                # if yes, update information in stock
            except:  # If not, get params from Quandl
                stock = Stock.get_Params(ticker=ticker, url=urls[i])
            #print(stock.prices)
            rets.append(pd.read_json(stock.returns, orient='index'))
            #print(type(rets))
                #prices = pd.concat(prices, axis=1)
        rets = pd.concat(rets, axis=1)
        rets = rets.dropna()
        rets = rets.ix[:datetime_time]
        #rets should stop
        print(rets)
        # print(market_index)
        market_index = np.array(market_index)
        print(market_list, diff_month)
        print("this is market index :{}".format(market_index))
        #print(float(self.initial_deposit),int(self.years), float(self.amount), market_index)

        port_stochastic = stochastic(float(self.initial_deposit), int(self.years), rets, float(self.amount), 2, 1, market_index,2000)
        #print(port_stochastic.strategy())
        #print(port_stochastic.strategy())
        #print(strat)
        #print("the weight is {0}, and the optimized result is {1}".format(weights, optimized_result))
        weights = port_stochastic.weights()
        print(weights)
        #print(weights)
        alloc_weights = {}
        for i, j in enumerate(weights[0]):
            alloc_weights[i] = j
        weights_df = pd.DataFrame([alloc_weights], columns=alloc_weights.keys())
        #weights_df.to_json("common/test")
        #print(weights_df)
        #print(weights_df.shape)
        #weights_df.reshape
        weights = np.array(weights)
        #print(weights)
        print(weights.size, weights.shape)
        #np.savetxt("common/test.csv", weights, delimiter=",")
        weights = weights.reshape(weights.size)
        weights = weights.tolist()
        #weights = [x.tolist() for x in weights]
        #print(weights)

        all_stock_prices = {}
        for i in self.tickers:
            all_prices = self.get_month_stock(i, datetime_time, timestamps)
            stock_prices = []
            start_i = 0
            print("length of all prices is {}".format(len(all_prices)))
            while start_i < len(all_prices):
                # print("this is added element: {}".format(market_list[i*3]))
                stock_prices.append(all_prices[start_i])
                start_i += 3

            all_stock_prices[i] = stock_prices
        #print(all_stock_prices)
        prices_df = pd.DataFrame([all_stock_prices], columns=all_stock_prices.keys())
        #print(prices_df)

        #print(prices_df)
        #self.get_actual_return(prices_df, weights_df)
        self.all_stock_prices = prices_df.to_json(orient = "index")
        #print(np.array(pd.read_json(self.all_stock_prices)))
        self.weights = weights
        self.quarters = round(len(weights)/len(PortfolioConstants.TICKERS))
        #self.optimized_result = optimized_result
        self.save_to_mongo()
        self.cal_actual_rets()
        return

    def cal_actual_rets(self):
        prices_df = pd.read_json(self.all_stock_prices)
        #print(weights_df)
        #print(weights_df.size)
        weights = self.weights
        print(len(weights))
        num_tickers = len(PortfolioConstants.TICKERS)
        num_periods = int(len(weights)/num_tickers)

        #weights_df = pd.read_json(self.weights)
        prices_final = []
        for i in PortfolioConstants.TICKERS:
            prices_final.append(prices_df.loc[i])
        prices_final = np.array(prices_final)
        print(prices_final)
        prices_final = prices_final.T
        prices_final = pd.DataFrame(prices_final.reshape((len(prices_df.ix[0, 0]),len(PortfolioConstants.TICKERS))))
        etf_return = prices_final.pct_change().dropna()
        print("this is etf{}".format(etf_return))
        achieved_rets = []
        start = 0
        for i in range(num_periods-1):
            achieved_rets.append(np.dot(etf_return.loc[i + 1, :], weights[start:start+num_tickers]))
        wealth = int(self.initial_deposit)
        account_balance = []
        account_balance.append(wealth)
        for i in achieved_rets:
            wealth *= 1+i
            account_balance.append(wealth)
        self.account_balance = account_balance
        self.rets = achieved_rets
        self.save_to_mongo()
        # print(etf_return)
        # print(achieved_rets)
        # print(account_balance)
        return achieved_rets, account_balance

    def plot_portfolio(self, weights):
        '''
        Plots pie chart of portfolio constituents
        :return: matplotlib matplotlib.figure.Figure object
        '''

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        print(self.tickers)
        print(weights)
        plt.pie(weights, labels=self.tickers, explode=[0.01]*len(weights), autopct='%1.1f%%')

        return fig

    def plot_performance(self, account_balance):

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        plt.plot(account_balance)

        return fig

    def save_to_mongo(self):
        Database.update(PortfolioConstants.COLLECTION, {'_id': self._id},self.json())

    def json(self):     # Creates JSON representation of portfolio instance
        return{
            "_id" : self._id,
            "description":self.description,
            "amount": self.amount,
            "initial_deposit": self.initial_deposit,
            "years": self.years,
            "importance": self.importance,
            "user_email" : self.user_email,
            "start_time" : self.start_time,
            "risk_appetite" : self.risk_appetite,
            "tickers" : self.tickers,
            "weights": self.weights,
            "quarters": self.quarters,
            "all_stock_prices": self.all_stock_prices,
            "account_balance": self.account_balance,
            "rets": self.rets
            #"optimized_result": self.optimized_result
        }

    @classmethod
    def get_by_id(cls, port_id):        # Retrieves portfolio from MongoDB by its unique id
        return cls(**Database.find_one(PortfolioConstants.COLLECTION,{'_id' : port_id}))

    @classmethod
    def get_by_email(cls, email):  # Retrieves portfolio(s) from MongoDB by user's email
        return [cls(**elem) for elem in Database.find(PortfolioConstants.COLLECTION, {'user_email': email})]
