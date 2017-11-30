from flask import Blueprint, request, session, redirect, url_for, render_template

from src.models.users.user import User
import src.models.users.errors as UserErrors
import src.models.users.decorators as user_decorators
from src.common.database import Database
from src.models.portfolios.portfolio import Portfolio
from src.models.portfolios.constants import TICKERS as PortTickers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import urllib
import base64
import numpy as np
import math
import datetime
from src.models.stocks.stock import Stock
portfolio_blueprint = Blueprint('portfolios', __name__)

@portfolio_blueprint.route('/portfolio/<string:portfolio_id>')
@user_decorators.requires_login
def get_portfolio_page(portfolio_id):   # Renders unique portfolio page
    port = Portfolio.get_by_id(portfolio_id)
    weights = port.weights
    n_weights = len(weights)
    n_tickers = len(PortTickers)
    start_ls = np.arange(0,n_weights,n_tickers)
    print(n_tickers,n_weights)
    #years = math.ceil(n_weights/(n_tickers*4))
    years = int(port.years)
    print(years)
    start_time = datetime.datetime.strptime(port.start_time, "%Y-%m-%d")
    start_year = start_time.year

    year_list = [start_year+x for x in range(years)]
    print(year_list)

    return render_template('/portfolios/portfolio_land_page.jinja2', portfolio = port, year_count = years, year_list = year_list)

@portfolio_blueprint.route('/portfolio/<string:portfolio_id>/<string:year_index>')
@user_decorators.requires_login
def get_individual_portfolio(portfolio_id, year_index):   # Renders unique portfolio page
    port = Portfolio.get_by_id(portfolio_id)
    #port.run_logic()

    weights = port.weights
    start_ls = np.arange(0,len(weights),len(PortTickers)*4) #we store weights for each year
    #print(weights)
    #print(start_ls)
    #print(PortTickers)
    #stock = Stock.get_by_ticker()
    year_index = int(year_index)-1
    plot_data_list = []
    i = start_ls[year_index]
    end = i + len(PortTickers) * 4
    weights_i = weights[i:end]
    start_i = np.arange(0, len(weights_i), len(PortTickers))
    print(weights_i, start_i)


    while i < end:
        weights_each = weights[i:i+len(PortTickers)]
        if weights_each == []:
            break
        fig = port.plot_portfolio(weights_each)
        canvas = FigureCanvas(fig)
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        plot_data_list.append(plot_data)
        print(len(plot_data_list))

        i += len(PortTickers)


    return render_template('/portfolios/portfolio.jinja2', portfolio = port, plot_url_list=plot_data_list, weights = weights_i, start_ls = start_i, year_index = year_index)

@portfolio_blueprint.route('/portfolio/<string:portfolio_id>/performance')
@user_decorators.requires_login
def get_performance_by_year(portfolio_id):   # Renders unique portfolio page
    port = Portfolio.get_by_id(portfolio_id)

    achieved_rets, account_balance_all = port.cal_actual_rets()

    fig = port.plot_performance(account_balance_all)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()
    #plot_data_list.append(plot_data)
    #i += len(PortTickers)


    return render_template('/portfolios/performance.jinja2', portfolio = port, plot_url_list=plot_data)






@portfolio_blueprint.route('/edit/<string:portfolio_id>', methods=['GET','POST'])
@user_decorators.requires_login
def change_risk(portfolio_id):         # Views form to change portfolio's associated risk aversion parameter
    port = Portfolio.get_by_id(portfolio_id)
    if request.method == "POST":
        risk_appetite = request.form['risk_appetite']
        port.description = request.form['goal_description']
        port.amount = request.form['Amount_for_goal']
        port.initial_deposit = request.form['initial_deposit']
        port.years = request.form['years_to_achieve']
        port.importance = request.form['importance']
        port.start_time = request.form['start_time']
        port.risk_appetite = risk_appetite
        port.run_logic()
        #port.save_to_mongo()


        return redirect(url_for(".get_portfolio_page", portfolio_id=port._id))

        #return render_template('/portfolios/optimal_portfolio.jinja2', portfolio = port, plot_url=plot_data)

    return render_template('/portfolios/edit_portfolio.jinja2')

@portfolio_blueprint.route('/new', methods=['GET','POST'])
@user_decorators.requires_login
def create_portfolio():            # Views form to create portfolio associated with active/ loggedin user
    if request.method == "POST":
        risk_appetite = request.form['risk_appetite']
        desc = request.form['goal_description']
        amount = request.form['Amount_for_goal']
        initial_deposit = request.form['initial_deposit']
        years = request.form['years_to_achieve']
        importance = request.form['importance']
        start_time = request.form['start_time']
        port = Portfolio(session['email'], desc, amount, initial_deposit,years, importance, risk_appetite= risk_appetite, start_time = start_time)
        #port.get_Params()
        #description, amount, initial_deposit, years, importance, risk_appetite
        port.run_logic()
        #print(port.weights)

        '''
        fig = port.runMVO()
        canvas = FigureCanvas(fig)
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        '''
        return redirect(url_for(".get_portfolio_page", portfolio_id = port._id))
        #return render_template('/portfolios/optimal_portfolio.jinja2', portfolio=port, plot_url=plot_data)

    return render_template('/portfolios/new_portfolio.jinja2')

@portfolio_blueprint.route('/delete/<string:portfolio_id>')
@user_decorators.requires_login
def delete_portfolio(portfolio_id):            # Views form to create portfolio associated with active/ loggedin user
    Database.remove("portfolios",{"_id":portfolio_id})
    user = User.find_by_email(session['email'])
    portfolios = user.get_portfolios()
    return render_template('/users/portfolios.jinja2', portfolios = portfolios)

@portfolio_blueprint.route('/test/<string:portfolio_id>')
@user_decorators.requires_login
def test_portfolio(portfolio_id):            # Views form to create portfolio associated with active/ loggedin user
    port = Portfolio.get_by_id(portfolio_id)
    port.run_logic()
    return render_template('/portfolios/weight_table.jinja2', portfolio= port)