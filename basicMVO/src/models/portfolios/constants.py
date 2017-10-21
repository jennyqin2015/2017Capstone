COLLECTION = 'portfolios'                                   # MongoDB collection
TICKERS = ['AAPL','AMZN','MMM','T','KO']                    # List of tickers used
RISK_PROFILE_INDECES = [32, 39, 60]                         # Gamma values corresponding to risk appetites
RISK_LABELS = ['high', 'medium','low']                      # Different portfolio risk levels
RISK_APP_DICT = dict(zip(RISK_LABELS,RISK_PROFILE_INDECES)) # Dict tying gamma values to risk levels
START_DATE = '2006-01-01'                                   # Start date for data collection
END_DATE = '2016-12-31'                                     # End date for data collection

