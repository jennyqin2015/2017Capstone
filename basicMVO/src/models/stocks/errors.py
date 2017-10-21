class StockError(Exception):
    def __init__(self, message):
        self.message = message

class IncorrectTickerError(StockError):     # Raised if ticker entered does not match any in Quandl table
    pass
