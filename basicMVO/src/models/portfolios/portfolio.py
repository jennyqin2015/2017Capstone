import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')
import cvxpy as cvx
from src.common.database import Database
import src.models.portfolios.constants as PortfolioConstants
from src.models.stocks.stock import Stock



class Portfolio(object):
    # Portfolio class creates portfolio instances for user portfolios using stocks in Stock class

    def __init__(self, user_email, risk_appetite, tickers = None, weights = None, _id=None):
        self.user_email = user_email
        self.risk_appetite = risk_appetite
        self.tickers = PortfolioConstants.TICKERS if tickers is None else tickers
        self.weights = weights
        self._id = uuid.uuid4().hex if _id is None else _id

    def __repr__(self):
        return "<Portfolio for user {}>".format(self.user_email)


    def get_Params(self, start_date = PortfolioConstants.START_DATE, end_date = PortfolioConstants.END_DATE):
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

        n = len(tickers)
        rets = []
        mu = []
        for ticker in tickers:
            try:
                stock = Stock.get_by_ticker(stock_ticker = ticker)    # Check if stock exists in db
            except:                                                   # If not, get params from Quandl
                stock = Stock.get_Params(ticker = ticker,start_date = start_date, end_date = end_date)

            rets.append(pd.read_json(stock.returns, orient='index'))
            mu.append(stock.mu)
            returns = pd.concat(rets, axis=1)

        mu = np.array(mu).reshape([n, 1])
        cov = returns.cov()
        cov = cov.values

        return mu, cov

    def runMVO(self, start_date = PortfolioConstants.START_DATE, end_date = PortfolioConstants.END_DATE, samples = 100):
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
        mu, cov = self.get_Params(start_date, end_date)
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
        Database.update(PortfolioConstants.COLLECTION,{'_id':self._id},self.json())

    def json(self):     # Creates JSON representation of portfolio instance
        return{
            "_id" : self._id,
            "user_email" : self.user_email,
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
