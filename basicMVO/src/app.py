from flask import Flask, render_template
from src.common.database import Database

# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config')
app.secret_key = "123"

# Initialize Database before running any other command
@app.before_first_request
def init_db():
    Database.initialize()

# Render home page
@app.route('/')
def home():
    return render_template('home.jinja2')

# Import all views
from src.models.users.views import user_blueprint
from src.models.portfolios.views import portfolio_blueprint
from src.models.stocks.views import stock_blueprint

# Register views in Flask app
app.register_blueprint(user_blueprint, url_prefix = '/users')
app.register_blueprint(portfolio_blueprint, url_prefix = '/portfolios')
app.register_blueprint(stock_blueprint, url_prefix = '/stocks')
