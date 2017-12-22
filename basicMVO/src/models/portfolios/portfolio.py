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
import pickle
from matplotlib import cm
from bson.binary import Binary

class Portfolio(object):
    # Portfolio class creates portfolio instances for user portfolios using stocks in Stock class

    def __init__(self, user_email, description, goal_type, amount,initial_deposit, years, importance, injection, start_time = None, tickers = None, weights = None, quarters = None, all_stock_prices = None, rets = None, wealth_allocated=None, account_balance = None,terminal_date=None, tree=None, goal_achieved=None, _id=None):
        self.user_email = user_email
        self.description = description
        self.goal_type = goal_type
        self.amount = amount # target amount of money to achieve by the terminal date of this goal
        self.initial_deposit = initial_deposit
        self.years = years
        self.importance = importance
        self.injection = injection
        self.start_time = start_time # this stores the starting time of the user's portfolio in the format of seconds
        self.terminal_date = terminal_date
        self.tickers = PortfolioConstants.TICKERS if tickers is None else tickers
        self.weights = weights
        self.quarters = quarters
        self.all_stock_prices = all_stock_prices
        self.account_balance = account_balance
        self.rets = rets
        self.wealth_allocated = wealth_allocated
        self.tree = tree
        self.goal_achieved = goal_achieved
        #self.index = 0 if index is None else index

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
        info_urls = PortfolioConstants.Info_URLs
        n = len(tickers)
        rets = []
        mu = []
        for i, ticker in enumerate(tickers):
            try:
                stock = Stock.get_by_ticker(stock_ticker = ticker) # Check if stock exists in db
                if stock.last_updated != time.time():
                    #update the database when the user logs in another day
                    stock = Stock.get_Params(ticker=ticker, url=urls[i], info_url= info_urls[i])
                #if yes, update information in stock
                #stock.update_data()
            except:                                                   # If not, get params from Quandl
                stock = Stock.get_Params(ticker = ticker, url = urls[i], info_url= info_urls[i])

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



            market_prices = []

            prices.to_csv("common/price_vti.csv")

            new_stamps = []

            for i in timestamps:
                #i is a datetime object
                time_key = datetime.datetime.fromtimestamp(i)

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

                        diff_in_seconds = datetime.timedelta(days=j).total_seconds()
                        key = i + diff_in_seconds*j
                        data_time_key = datetime.datetime.fromtimestamp(key)
                        day = data_time_key.day
                        year = data_time_key.year
                        month = data_time_key.month
                        time_key = '{0}-{1}-{2}'.format(year, month, day)
                        data_time_key = datetime.datetime.strptime(time_key, "%Y-%m-%d")


                        search_key.append(data_time_key)
                    market_price = prices.loc[search_key]

                    market_price = np.array(market_price)
                    market_price = market_price[~np.isnan(market_price)]
                    market_prices.append(market_price[0])

            return market_prices


    def get_month_index(self, start_time, diff_month):
        '''
        imagine one user created his goal on the first day of 2014, and 2 years have past. Our model should be able to return the weight allocation
        by the end of each season. Therefore, our model should provided 7 weight allocations by now (2017 Nov 18th).
        '''
        market_prices = []

        data = pd.read_csv('common/SNP_Price.csv')
        prices = pd.DataFrame(data.Close.values, index=data.Date, columns=['price'])
        start_date = start_time.timestamp()
        diff_in_seconds = datetime.timedelta(days = 31).total_seconds()
        time_stamps = []
        for i in range(diff_month+1):
            s = start_date + i*diff_in_seconds
            time_stamps.append(s)


        for i in time_stamps[:-1]:
            time_key = datetime.datetime.fromtimestamp(i).strftime('%Y-%m-%d')

            try:
                market_price = prices.loc[time_key]


                market_price = np.array(market_price)

                market_prices.append(market_price[0])
            except KeyError:
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

                market_price = prices.loc[search_key]


                market_price = np.array(market_price)
                market_price = market_price[~np.isnan(market_price)]
                market_prices.append(market_price[0])

        return market_prices, time_stamps # market_prices returns a list which contains 4 prices

    #compute r and q coefficients
    def get_r_q(self):
        q = 1
        if self.importance =="3":
            r = random.randint(6, 9)

        elif self.importance == "2":
            r = random.randint(3, 6)
        else:
            r = random.randint(1, 3)
        return r


    def run_logic(self):
        str_time = self.start_time
        #convert self.start_time to a datetime object
        datetime_time = datetime.datetime.strptime(str_time, "%Y-%m-%d")
        datetime_terminal_date = datetime.datetime(datetime_time.year+int(self.years), datetime_time.month, datetime_time.day)
        self.terminal_date = datetime.datetime.strftime(datetime_terminal_date,"%Y-%m-%d")
        now_date = datetime.datetime.now()
        diff = now_date - datetime_time

        #compute the number of months between the starting date to today's date
        diff_month = int(diff.days/31)
        #compare the number of months with the number of months between the starting date with the terminal date for the goal
        if diff_month > int(self.years)*12:
            diff_month = int(self.years)*12
        else:
            diff_month = diff_month
        #get market index parameter for our business model
        #get the market index price for the starting date of each month
        market_list, timestamps = self.get_month_index(datetime_time, diff_month) #this returns a list with market indexes for all months
        diff_quarters = int(diff_month/3)
        market_index = []

        start_i = 0
        while start_i < len(market_list):
            market_index.append(market_list[start_i])
            start_i += 3
        market_index = np.array(market_index)
        # get asset returns matrix for the business model
        tickers = self.tickers.copy()
        urls = PortfolioConstants.URLS
        info_urls = PortfolioConstants.Info_URLs
        rets = []

        for i, ticker in enumerate(tickers):

            try:
                stock = Stock.get_by_ticker(stock_ticker=ticker)  # Check if stock exists in db
                # if yes, update information in stock
            except:  # If not, get params from Quandl
                stock = Stock.get_Params(ticker=ticker, url=urls[i], info_url = info_urls[i])
            #print(stock.prices)
            rets.append(pd.read_json(stock.returns, orient='index'))
        rets = pd.concat(rets, axis=1)
        rets = rets.dropna()
        rets = rets.ix[:datetime_time]

        #get other parameters
        r = self.get_r_q()
        perc_init = float(self.initial_deposit)/float(self.amount)
        perc_inj = float(self.injection)/float(self.amount)
        first_deposit = 100*perc_init
        inj = 100*perc_inj
        target = 100
        #run business model to get all strategies for all possible nodes in the multiperiod binomial tree
        port_stochastic = stochastic(first_deposit, int(self.years), rets, target, r, 1, market_index, inj)
        #get the weights allocated with the known market movement during the past periods

        weights = port_stochastic.weights()
        weights = np.array(weights)
        weights = weights.reshape(weights.size)
        weights = weights.tolist()
        #this will store the computed strategies into our database and the strategies will be loaded for the purpose of portfolio rebalancing every 3 months
        tree = Binary(pickle.dumps(np.array(port_stochastic.fcs()),protocol=2),subtype=128)
        self.tree = tree
        all_stock_prices = {}
        for i in self.tickers:
            all_prices = self.get_month_stock(i, datetime_time, timestamps)
            #print("length of all prices:{}".format(len(all_prices)))
            stock_prices = []
            start_i = 0
            while start_i < len(all_prices):
                # print("this is added element: {}".format(market_list[i*3]))
                stock_prices.append(all_prices[start_i])
                start_i += 3

            all_stock_prices[i] = stock_prices
        prices_df = pd.DataFrame([all_stock_prices], columns=all_stock_prices.keys())

        self.all_stock_prices = prices_df.to_json(orient = "index")
        self.weights = weights
        self.quarters = round(len(weights)/len(PortfolioConstants.TICKERS))
        self.save_to_mongo()
        self.cal_actual_rets()
        return
    #function that achieves portfolio rebalancing or suggest a new portfolio allocation every 3 months
    def update_weights(self):
        tree = pickle.loads(self.tree)
        tree = tree.tolist()
        str_time = self.start_time

        datetime_time = datetime.datetime.strptime(str_time, "%Y-%m-%d")
        datetime_terminal_date = datetime.datetime(datetime_time.year + int(self.years), datetime_time.month,
                                                   datetime_time.day)
        self.terminal_date = datetime.datetime.strftime(datetime_terminal_date, "%Y-%m-%d")
        now_date = datetime.datetime.now()
        diff = now_date - datetime_time
        diff_month = int(diff.days / 31)
        int(self.years) / 5
        if diff_month > int(self.years) * 12:
            diff_month = int(self.years) * 12
        else:
            diff_month = diff_month
        market_list, timestamps = self.get_month_index(datetime_time,
                                                       diff_month)  # this returns a list with market indexes for all months
        diff_quarters = int(diff_month / 3)
        market_index = []

        start_i = 0
        while start_i < len(market_list):
            market_index.append(market_list[start_i])
            start_i += 3
        big_lis = [market_index[i:i + 4] for i in range(0, len(market_index), 4)]
        final_lis = []
        for j in range(len(big_lis)):
            family_list = [0]
            price_lis = big_lis[j]
            for i in range(1, len(price_lis)):
                cur_node = price_lis[i]
                pre_node = price_lis[i - 1]

                if cur_node >= pre_node * 1.04:
                    cur_node = family_list[i - 1] * 2 + 1
                else:
                    cur_node = family_list[i - 1] * 2 + 2

                family_list.append(cur_node)

            final_lis.append(family_list)
        res = tree
        expected = res[-1]
        res.pop(-1)
        stra = res
        optimal_index = final_lis
        liss = []
        a = min(len(stra), len(optimal_index))
        for i in range(a):
            lis = []
            op_id = optimal_index[i]
            op_stra = stra[i]
            for j in range(len(op_id)):
                k = op_id[j]
                lis.append(op_stra[k])
            liss.append(lis)
        liss.append(expected)
        lis = liss
        lis.pop(-1)
        nlis = lis

        for i in range(len(nlis)):
            for j in range(len(nlis[i])):
                nlis[i][j] = nlis[i][j] / np.sum(nlis[i][j])
        weights = nlis
        weights = np.array(weights)
        # print(weights)

        # np.savetxt("common/test.csv", weights, delimiter=",")
        weights = weights.reshape(weights.size)
        weights = weights.tolist()
        self.weights = weights
        return
    #function that calculates achieved returns for each of the periods
    def cal_actual_rets(self):
        self.update_weights()
        prices_df = pd.read_json(self.all_stock_prices)
        #print(weights_df)
        #print(weights_df.size)
        weights = self.weights

        num_tickers = len(PortfolioConstants.TICKERS)
        num_periods = int(len(weights)/num_tickers)

        #weights_df = pd.read_json(self.weights)
        prices_final = []
        for i in PortfolioConstants.TICKERS:
            prices_final.append(prices_df.loc[i])
        prices_final = np.array(prices_final)

        prices_final = prices_final.T
        prices_final = pd.DataFrame(prices_final.reshape((len(prices_df.ix[0, 0]),len(PortfolioConstants.TICKERS))))
        etf_return = prices_final.pct_change().dropna()

        achieved_rets = []
        start = 0
        start_i = 1
        if len(etf_return) > 0:
            while start_i < len(etf_return)+1: #used to be num_periods-1

                achieved_rets.append(np.dot(etf_return.loc[start_i, :], weights[start:start+num_tickers]))
                start+=num_tickers
                start_i += 1
        else:
            achieved_rets = []
        wealth = int(self.initial_deposit)
        account_balance = []
        account_balance.append(wealth)
        if len(achieved_rets) > 0:
            for i in achieved_rets:
                wealth *= 1+i
                wealth +=float(self.injection)
                account_balance.append(wealth)
        else:
            account_balance = account_balance
        self.account_balance = account_balance
        self.rets = [0]*len(account_balance)
        self.rets[1:] = achieved_rets
        self.calculate_wealth_allocated()
        self.goal_achieved = "Yes" if account_balance[-1]>float(self.amount) else "No"
        self.save_to_mongo()
        return achieved_rets, account_balance

    def calculate_wealth_allocated(self):

        weights = self.weights
        start=0
        account_balance = self.account_balance
        num_tickers = len(PortfolioConstants.TICKERS)
        num_periods = int(len(weights) / num_tickers)
        wealth_allocated = [0]*((num_periods)*num_tickers) #used to be num_periods-1
        for i in range(num_periods): #used to be num_periods-1

            wealth_allocated[start:start+num_tickers] = [j*account_balance[i] for j in weights[start:start+num_tickers]]

            start+=num_tickers
        self.wealth_allocated = wealth_allocated
        self.save_to_mongo()
        return


    def plot_portfolio(self, weights, quarter_index):
        '''
        Plots pie chart of portfolio constituents
        :return: matplotlib matplotlib.figure.Figure object
        '''

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        total = sum(weights)
        num = len(PortfolioConstants.TICKERS)
        cs = cm.Set1(np.arange(num) / num)
        plt.pie(weights, labels=self.tickers, explode=[0.01]*len(weights),autopct="%.2f%%", colors=cs)
        plt.title("Q{} Portfolio Allocation".format(quarter_index))
        # plt.title("Portfolio allocation")
        return fig
    #plot allocated portfolio performance
    def plot_performance(self, account_balance):

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        account_balance_ls = [round(x,2) for x in account_balance]
        plt.plot(account_balance_ls)
        plt.title("Account Balance Plot")
        plt.ylabel("Account Balance in $")
        plt.xlabel("Quarter Index")
        for i,j in enumerate(account_balance_ls):
            ax.annotate("%s"%j, xy=(i,j),xytext=(15,0), textcoords="offset points")


        return fig
    #plot achieved return during all periods
    def plot_return(self, rets):

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        rets_ls = [round(x,2) for x in rets]

        plt.plot(rets_ls)
        plt.title("Achieved Returns Plot")
        plt.ylabel("Achieved Return")
        plt.xlabel("Quarter Index")
        for i,j in enumerate(rets_ls):
            ax.annotate("%s"%j, xy=(i,j),xytext=(15,0), textcoords="offset points")

        return fig

    def save_to_mongo(self):
        Database.update(PortfolioConstants.COLLECTION, {'_id': self._id},self.json())

    def json(self):     # Creates JSON representation of portfolio instance
        return{
            "_id" : self._id,
            "description":self.description,
            "goal_type": self.goal_type,
            "amount": self.amount,
            "initial_deposit": self.initial_deposit,
            "years": self.years,
            "importance": self.importance,
            "user_email" : self.user_email,
            "start_time" : self.start_time,
            "injection": self.injection,
            "tickers" : self.tickers,
            "weights": self.weights,
            "quarters": self.quarters,
            "all_stock_prices": self.all_stock_prices,
            "account_balance": self.account_balance,
            "rets": self.rets,
            "terminal_date": self.terminal_date,
            "wealth_allocated":self.wealth_allocated,
            "tree": self.tree,
            "goal_achieved": self.goal_achieved
        }

    @classmethod
    def get_by_id(cls, port_id):        # Retrieves portfolio from MongoDB by its unique id
        return cls(**Database.find_one(PortfolioConstants.COLLECTION,{'_id' : port_id}))

    @classmethod
    def get_by_email(cls, email):  # Retrieves portfolio(s) from MongoDB by user's email
        return [cls(**elem) for elem in Database.find(PortfolioConstants.COLLECTION, {'user_email': email})]
