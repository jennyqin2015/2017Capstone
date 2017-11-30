from flask import Flask, render_template
from src.common.database import Database
import os
import pandas as pd
import datetime
# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config')
app.secret_key = "123"
import time
#update SNP_price

# today = datetime.datetime.now()
# snp_url = "http://finance.google.com/finance/historical?q=INDEXSP:.INX"
# url = "{0}&startdate=Jan+01%2C+2002&enddate={1}+{2}%2C+{3}&output=csv".format(snp_url, today.strftime("%b"), today.day, today.year)
# data = pd.read_csv(url)
# os.remove("common/SNP_Price.csv")
# Initialize Database before running any other command
@app.before_first_request
def init_db():
    Database.initialize()

# Render home page
@app.route('/')
def home():
    return render_template("index.jinja2")

# Import all views
from src.models.users.views import user_blueprint
from src.models.portfolios.views import portfolio_blueprint
from src.models.stocks.views import stock_blueprint

# Register views in Flask app
app.register_blueprint(user_blueprint, url_prefix = '/users')
app.register_blueprint(portfolio_blueprint, url_prefix = '/portfolios')
app.register_blueprint(stock_blueprint, url_prefix = '/stocks')
