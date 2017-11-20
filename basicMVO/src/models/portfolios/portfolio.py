import datetime
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    def __init__(self, user_email, description, amount,initial_deposit, years, importance, risk_appetite, start_time = None, tickers = None, weights = None, _id=None):
        self.user_email = user_email
        self.description = description
        self.amount = amount # target amount of money to achieve by the terminal date of this goal
        self.initial_deposit = initial_deposit
        self.years = years
        self.importance = importance
        self.risk_appetite = risk_appetite
        self.start_time = time.time() if start_time is None else start_time # this stores the starting time of the user's portfolio in the format of seconds
        self.tickers = PortfolioConstants.TICKERS if tickers is None else tickers
        self.weights = weights
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

    def runMVO(self, samples = 100):
        '''
        (A) Gets Portfolio characteristics (exp. return & cov. matrix)
        (B) Runs basic MVO for different risk aversion values (gamma)
        (C) Portfolio weights given current portfolio's risk aversion are extracted and assigned to
            current portfolio instance which is then updated in MongoDB
        (D) A figure of the efficient frontier for all risk aversion parameters is created along with
            the individual assets used.

        :param start_date: time-series start date as string format (ex: YYYY-MM-DD '2006-01-01')
        :param end_date: time-series start date as string format (ex: YYYY-MM-DD '2016-12-31')
        :param samples: number of portfolios to compute for comparison/ efficient frontier

        :return: Matplotlib figure object of efficient frontier
        '''

        #        (A)        #
        #####################
        mu, cov = self.get_Params()
        std = np.sqrt(np.diag(cov))
        n = len(mu)

        #        (B)        #
        #####################
        w = cvx.Variable(n)
        gamma = cvx.Parameter(sign='positive')
        exp_ret = mu.T * w
        risk = cvx.quad_form(w, cov)

        prob = cvx.Problem(cvx.Maximize(exp_ret - gamma * risk),
                           [cvx.sum_entries(w) == 1,
                            w >= 0])

        SAMPLES = samples
        risk_data = np.zeros(SAMPLES)
        ret_data = np.zeros(SAMPLES)
        ws = []
        gamma_vals = np.logspace(-2, 4, num=SAMPLES)

        for i in range(SAMPLES):
            gamma.value = gamma_vals[i]
            prob.solve()
            risk_data[i] = cvx.sqrt(risk).value
            ws.append(np.array(w.value).T.tolist()[0])
            ret_data[i] = exp_ret.value

        w_minVar = cvx.Variable(n)
        exp_ret_minVar = mu.T * w_minVar

        risk_minVar = cvx.quad_form(w_minVar, cov)
        prob_minVar = cvx.Problem(cvx.Minimize(risk_minVar),
                            [cvx.sum_entries(w_minVar) == 1,
                             w_minVar >= 0])
        prob_minVar.solve()
        risk_data_minVar = cvx.sqrt(risk_minVar).value
        ret_data_minVar = exp_ret_minVar.value

        #        (C)        #
        #####################
        gam = PortfolioConstants.RISK_APP_DICT.get(self.risk_appetite)

        port_weights = ws[gam]
        self.weights = port_weights
        self.save_to_mongo()

        #        (D)        #
        #####################
        fig = self.plot_comparison(risk_data, ret_data, gamma_vals, risk_data_minVar, ret_data_minVar, std, mu)

        return fig

    def plot_comparison(self, risk_data, ret_data, gamma_vals, risk_data_minVar, ret_data_minVar, std, mu):
        '''
        Plots a figure of the efficient frontier for all risk aversion parameters along with the
        individual assets used.

        :param risk_data: (list) portfolio variances for different risk aversion parameters
        :param ret_data: (list) portfolio expected returns for different risk aversion parameters
        :param gamma_vals: (list) portfolio risk aversion parameters
        :param risk_data_minVar: (float) portfolio variance for Minimum Variance portfolio
        :param ret_data_minVar: (float) portfolio expected return for Minimum Variance portfolio
        :param std: (np.array) standard deviation of assets used
        :param mu: (list) expected returns of assets used

        :return: Matplotlib figure object of efficient frontier
        '''
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        plt.plot(risk_data, ret_data, 'g-')

        for i in PortfolioConstants.RISK_APP_DICT:
            marker = PortfolioConstants.RISK_APP_DICT.get(i)
            plt.plot(risk_data[marker], ret_data[marker], 'bs')
            if i == 'low':
                y = 0
            else:
                y = -0.0015

            if i == self.risk_appetite:
                indicator = "--> Your Portfolio"
            else:
                indicator = ""

            ax.annotate(i + " ($\gamma = %.2f$)" % gamma_vals[marker], xy=(risk_data[marker] + .0015, ret_data[marker] + y))
            ax.annotate(indicator, xy=(risk_data[marker] + .02, ret_data[marker] + y))

        plt.plot(risk_data_minVar, ret_data_minVar, 'rs')
        ax.annotate('minimum Variance', xy=(risk_data_minVar - .015, risk_data_minVar - .00015))
        n = len(PortfolioConstants.TICKERS)
        for i in range(n):
            plt.plot(std[i], mu[i], 'o')
            ax.annotate(PortfolioConstants.TICKERS[i], xy=(std[i] - .0005, mu[i] - .001))
        plt.xlabel('Standard deviation')
        plt.ylabel('Return')
        plt.xlim([0, 0.15])

        return fig

    def get_one_year_index(self, start_year):
        #this function creates the index
        #since we retrieved financial data from 2012 to today
        '''
        imagine one user created his goal on the first day of 2014, and 2 years have past. Our model should be able to return the weight allocation
        by the end of each season. Therefore, our model should provided 7 weight allocations by now (2017 Nov 18th).
        '''
        market_prices = []
        market_index = Stock.get_by_ticker('SNP')
        prices = pd.read_json(market_index.prices)
        ''' this is for actual use of the model
        start_date = self.start_time
        '''
        '''
        set start_date to be the time
        '''
        str_time = '{0}-01-01'.format(start_year)
        datetime_time = datetime.datetime.strptime(str_time, "%Y-%m-%d")
        start_date = datetime_time.timestamp()
        diff_in_seconds = datetime.timedelta(4*365/12).total_seconds()
        # s1, s2, s3, s4 are the four timestamps for first, second, third quarters
        s1 = start_date + diff_in_seconds
        s2 = s1 + diff_in_seconds
        s3 = s2 + diff_in_seconds
        s4 = s3 + diff_in_seconds
        time_stamps = [s1,s2,s3,s4]
        for i in time_stamps:
            time_key = datetime.datetime.fromtimestamp(i).strftime('%Y-%m-%d')
            try:
                market_price = prices.loc[time_key]
                print(market_price)
                market_prices.append(market_price)
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
                date_list = random.sample(range(1, 27), 26)
                for j in date_list:
                    try:
                        day = datetime.datetime.fromtimestamp(i).day + j
                        year = datetime.datetime.fromtimestamp(i).year
                        month = datetime.datetime.fromtimestamp(i).month
                        time_key = '{0}-{1}-{2}'.format(year, month, day)
                        market_price = prices.loc[time_key]
                        market_prices.append(market_price)
                    except KeyError:
                        continue
        #print(market_prices)
        return market_prices # market_prices returns a list which contains 4 prices

    '''
    def compute_r_q(self):
        if self.importance ==
    '''


    def get_market_index(self,start_year, diff_year):
        market_index = []
        for i in range(diff_year):
            market_index.append(self.get_one_year_index(start_year))
            start_year += 1
        return market_index

    def run_logic(self):
        str_time = "2014-01-01"
        datetime_time = datetime.datetime.strptime(str_time, "%Y-%m-%d")
        start_year = datetime_time.year
        diff_year = 2
        market_index = self.get_market_index(start_year,diff_year)
        tickers = self.tickers.copy()
        urls = PortfolioConstants.URLS
        rets = []

        for i, ticker in enumerate(tickers):
            if ticker != "SNP":
                try:
                    stock = Stock.get_by_ticker(stock_ticker=ticker)  # Check if stock exists in db
                    # if yes, update information in stock
                    stock.update_data()
                except:  # If not, get params from Quandl
                    stock = Stock.get_Params(ticker=ticker, url=urls[i])

                rets.append(pd.read_json(stock.returns, orient='index'))
                print(type(rets))
                #prices = pd.concat(prices, axis=1)
        rets = pd.concat(rets, axis=1)
        print(rets)
        #port_stochastic = stochastic(float(self.initial_deposit), int(self.years), rets, float(self.amount), 1, 3, market_index)
        #print(port_stochastic.strategy())
        #self.weights = port_stochastic
        #self.save_to_mongo()

        return





    def plot_portfolio(self):
        '''
        Plots pie chart of portfolio constituents
        :return: matplotlib matplotlib.figure.Figure object
        '''

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        plt.pie(self.weights, labels=self.tickers, explode=[0.01]*len(self.weights), autopct='%1.1f%%')

        return fig

    def save_to_mongo(self):
        Database.update(PortfolioConstants.COLLECTION,{'_id':self._id}, self.json())

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
            "weights": self.weights
        }

    @classmethod
    def get_by_id(cls, port_id):        # Retrieves portfolio from MongoDB by its unique id
        return cls(**Database.find_one(PortfolioConstants.COLLECTION,{'_id' : port_id}))

    @classmethod
    def get_by_email(cls, email):  # Retrieves portfolio(s) from MongoDB by user's email
        return [cls(**elem) for elem in Database.find(PortfolioConstants.COLLECTION, {'user_email': email})]
