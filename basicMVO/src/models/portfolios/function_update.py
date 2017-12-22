def run_logic(self):
    # user = cls(**Database.find_one('users', {'email': email}))
    str_time = self.start_time

    datetime_time = datetime.datetime.strptime(str_time, "%Y-%m-%d")
    datetime_terminal_date = datetime.datetime(datetime_time.year + int(self.years), datetime_time.month,
                                               datetime_time.day)
    self.terminal_date = datetime.datetime.strftime(datetime_terminal_date, "%Y-%m-%d")
    now_date = datetime.datetime.now()
    diff = now_date - datetime_time

    # diff_year = int(diff.days/365)
    diff_month = int(diff.days / 31)
    int(self.years) / 5

    # if self.years <= 5:
    # change diff_year to diff_month, and multiplied year with 12

    if diff_month > int(self.years) * 12:
        diff_month = int(self.years) * 12
        print(diff_month)
    else:
        diff_month = diff_month
    market_list, timestamps = self.get_month_index(datetime_time,
                                                   diff_month)  # this returns a list with market indexes for all months
    diff_quarters = int(diff_month / 3)
    market_index = []

    start_i = 0
    while start_i < len(market_list):
        # print("this is added element: {}".format(market_list[i*3]))
        market_index.append(market_list[start_i])
        start_i += 3

    print(market_index)
    # print(len(market_index))
    tickers = self.tickers.copy()
    urls = PortfolioConstants.URLS
    info_urls = PortfolioConstants.Info_URLs
    rets = []

    for i, ticker in enumerate(tickers):

        try:
            stock = Stock.get_by_ticker(stock_ticker=ticker)  # Check if stock exists in db
            # if yes, update information in stock
        except:  # If not, get params from Quandl
            stock = Stock.get_Params(ticker=ticker, url=urls[i], info_url=info_urls[i])
        # print(stock.prices)
        rets.append(pd.read_json(stock.returns, orient='index'))
        # print(type(rets))
        # prices = pd.concat(prices, axis=1)
    rets = pd.concat(rets, axis=1)
    rets = rets.dropna()
    rets = rets.ix[:datetime_time]
    # rets should stop

    # print(market_index)
    market_index = np.array(market_index)

    # print(float(self.initial_deposit),int(self.years), float(self.amount), market_index)
    r = self.get_r_q()
    if float(self.initial_deposit) > 1000 and float(self.amount) > 1000:
        first_deposit = float(self.initial_deposit) / 1000
        target = float(self.amount) / 1000
    else:
        first_deposit = float(self.initial_deposit)
        target = float(self.amount)