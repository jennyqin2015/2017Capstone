from flask import Blueprint, render_template

from src.models.stocks.stock import Stock

stock_blueprint = Blueprint('stocks', __name__)

@stock_blueprint.route('/')
def index():    # Views list of available/stored stocks
    stocks = Stock.all()
    return render_template('stocks/stock_index.jinja2', stocks = stocks)


@stock_blueprint.route('/stock/<string:stock_ticker>')
def stock_page(stock_ticker):   # Renders unique stock page
    stock = Stock.get_by_ticker(stock_ticker)
    return render_template('stocks/stock.jinja2', stock = stock)


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
