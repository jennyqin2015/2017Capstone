import pandas as pd
COLLECTION = 'portfolios'                                   # MongoDB collection
# Tickers could be modified in file "tickers.csv"
# We assign tickers to a list in this python file
URLS = ['http://finance.google.com/finance/historical?q=NASDAQ:QQQ', 'http://finance.google.com/finance/historical?q=NASDAQ:QTEC','http://finance.google.com/finance/historical?q=NYSEARCA:VDC', 'http://finance.google.com/finance/historical?q=NYSEARCA:IHI','http://finance.google.com/finance/historical?q=NASDAQ:IEI', 'http://finance.google.com/finance/historical?q=NYSEARCA:PSI', 'http://finance.google.com/finance/historical?q=NYSEARCA:PBJ', 'http://finance.google.com/finance/historical?q=NYSEARCA:XLY']#, "http://finance.google.com/finance/historical?q=NASDAQ:IEF", "http://finance.google.com/finance/historical?q=NASDAQ:TLT", "http://finance.google.com/finance/historical?q=NYSEARCA:XLP", "https://finance.google.com/finance/historical?q=NYSE:BDX", "https://finance.google.com/finance/historical?q=NYSEARCA:IAU", "https://finance.google.com/finance/historical?q=NYSEARCA:GLD"]

TICKERS = ['QQQ',"QTEC", "VDC","IHI", 'IEI', "PSI", "PBJ", "XLY"]
Info_URLs = ["https://finance.yahoo.com/quote/QQQ/profile?p=QQQ","http://www.etf.com/QTEC","https://finance.yahoo.com/quote/ihi?ltr=1","http://www.nasdaq.com/symbol/iei", "https://finance.yahoo.com/quote/PSI/","https://finance.yahoo.com/quote/VDC/", "https://finance.yahoo.com/quote/PBJ/","https://finance.yahoo.com/quote/xly/"]
#TICKERS = ['AAPL','AMZN','MMM','T','KO']                    # List of tickers used
#NYSEARCA: IWV
RISK_PROFILE_INDECES = [32, 39, 60]                         # Gamma values corresponding to risk appetites
RISK_LABELS = ['high', 'medium','low']                      # Different portfolio risk levels
RISK_APP_DICT = dict(zip(RISK_LABELS,RISK_PROFILE_INDECES)) # Dict tying gamma values to risk levels
START_DATE = '2012-01-01'                                   # Start date for data collection
END_DATE = '2016-12-31'                                     # End date for data collection

