from flask import Blueprint, render_template, request, redirect, url_for, json

from src.common.database import Database
from src.models.stocks.stock import Stock
import src.models.users.decorators as user_decorators
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64

stock_blueprint = Blueprint('stocks', __name__)

@stock_blueprint.route('/')
def index():    # Views list of available/stored stocks
    stocks = Stock.all()
    return render_template('stocks/stock_index.jinja2', stocks = stocks)

'''
@stock_blueprint.route('/stock/<string:stock_ticker>')
def stock_page(stock_ticker):   # Renders unique stock page
    stock = Stock.get_by_ticker(stock_ticker)
    return render_template('stocks/stock.jinja2', stock = stock)
'''
@stock_blueprint.route('/stock/<string:stock_ticker>')
def stock_page(stock_ticker):   # Renders unique portfolio page
    stock = Stock.get_by_ticker(stock_ticker)
    fig = stock.plot_stock()
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()

    return render_template('stocks/stock.jinja2', stock = stock, plot_url=plot_data)
'''
@stock_blueprint.route('/new', methods=['GET','POST'])
@user_decorators.requires_admin_permissions
def create_stock(): 

    if request.method == 'POST':
        pass

    return render_template('stocks/new_stock.jinja2')


@stock_blueprint.route('/delete/<string:stock_id>', methods=['GET'])
@user_decorators.requires_admin_permissions
def delete_stock(stock_id):
    Stock.get_by_ticker(stock_id).remove()

    return redirect(url_for('.index'))

'''
