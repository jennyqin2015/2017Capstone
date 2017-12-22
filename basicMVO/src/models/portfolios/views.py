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
    port.cal_actual_rets()
    weights = port.weights
    print(weights[0:6])
    account_balance_all = port.account_balance
    print(account_balance_all)
    rets = port.rets
    print(rets)
    n_weights = len(weights)
    n_tickers = len(PortTickers)
    start_ls = np.arange(0,n_weights,n_tickers)


    years = int(port.years)

    start_time = datetime.datetime.strptime(port.start_time, "%Y-%m-%d")
    start_year = start_time.year
    plot_url_list = []
    year_list = [start_year+x for x in range(years)]

    #plot portfolio balance
    fig = port.plot_performance(account_balance_all)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()
    plot_url_list.append(plot_data)
    # plot returns during these quarters
    fig_2 = port.plot_return(rets)
    canvas = FigureCanvas(fig_2)
    img = BytesIO()
    fig_2.savefig(img)
    img.seek(0)
    plot_data_2 = base64.b64encode(img.read()).decode()
    plot_url_list.append(plot_data_2)
    return render_template('/portfolios/portfolio_land_page.jinja2', portfolio = port, year_count = years, year_list = year_list, plot_url_list=plot_url_list)

@portfolio_blueprint.route('/portfolio/<string:portfolio_id>/<string:year_index>')
@user_decorators.requires_login
def get_individual_portfolio(portfolio_id, year_index):   # Renders unique portfolio page
    port = Portfolio.get_by_id(portfolio_id)
    #port.run_logic()
    port.years

    weights = port.weights
    wealth_allocated = port.wealth_allocated

    start_ls = np.arange(0,len(weights),len(PortTickers)*4) #we store weights for each year
    if len(start_ls) > 1 or len(weights)>0:
    #print(weights)
    #print(start_ls)
    #print(PortTickers)
    #stock = Stock.get_by_ticker()
        year_index = int(year_index)


        plot_data_list = []
        if year_index>len(start_ls)-1:
            return render_template('/portfolios/future.jinja2', portfolio=port)
        i = start_ls[year_index]
        end = i + len(PortTickers) * 4
        weights_i = weights[i:end]


        start_i = np.arange(0, len(weights_i), len(PortTickers))
        print(weights_i, start_i)


        while i < end:
            weights_each = weights[i:i+len(PortTickers)]
            wealth_each= wealth_allocated[i:i + len(PortTickers)]
            print(wealth_each)
            if len(wealth_each) == 0:
                print("hi")
                return render_template('/portfolios/portfolio.jinja2', portfolio=port, plot_url_list=plot_data_list, weights=weights_i, start_ls=start_i, year_index=year_index)

            quarter_index = int(i/len(PortTickers))
            fig = port.plot_portfolio(wealth_each, quarter_index)
            canvas = FigureCanvas(fig)
            img = BytesIO()
            fig.savefig(img)
            img.seek(0)
            plot_data = base64.b64encode(img.read()).decode()
            plot_data_list.append(plot_data)
            print(len(plot_data_list))

            i += len(PortTickers)


        return render_template('/portfolios/portfolio.jinja2', portfolio = port, plot_url_list=plot_data_list, weights = weights_i, start_ls = start_i, year_index = year_index)

    else:

        return render_template('/portfolios/future.jinja2', portfolio= port)

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
def update_portfolio(portfolio_id):         # Views form to change portfolio's associated risk aversion parameter
    port = Portfolio.get_by_id(portfolio_id)
    if request.method == "POST":
        #risk_appetite = request.form['risk_appetite']
        port.description = request.form['goal_description']
        port.amount = request.form['Amount_for_goal']
        port.initial_deposit = request.form['initial_deposit']
        port.years = request.form['years_to_achieve']
        port.importance = request.form['importance']
        port.start_time = request.form['start_time']
        #port.risk_appetite = risk_appetite
        port.run_logic()
        #port.save_to_mongo()


        return redirect(url_for(".get_portfolio_page", portfolio_id=port._id))

        #return render_template('/portfolios/optimal_portfolio.jinja2', portfolio = port, plot_url=plot_data)

    return render_template('/portfolios/edit_portfolio.jinja2', portfolio=port)

@portfolio_blueprint.route('/new', methods=['GET','POST'])
@user_decorators.requires_login
def create_portfolio():            # Views form to create portfolio associated with active/ loggedin user
    if request.method == "POST":
        desc = request.form['goal_description']
        amount = request.form['Amount_for_goal']
        goal_type = request.form['goal_type']
        initial_deposit = request.form['initial_deposit']
        years = request.form['years_to_achieve']
        importance = request.form['importance']
        start_time = request.form['start_time']
        injection = request.form["injection"]
        port = Portfolio(session['email'], desc, goal_type, amount, initial_deposit,years, importance, injection, start_time = start_time)
        user = User.find_by_email(session['email'])
        user.number_goals+=1
        user.save_to_mongo()
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
        return redirect(url_for(".create_success"))
        #return render_template('/portfolios/optimal_portfolio.jinja2', portfolio=port, plot_url=plot_data)

    return render_template('/portfolios/new_portfolio.jinja2')

@portfolio_blueprint.route('/create_success')
@user_decorators.requires_login
def create_success():            # Views form to create portfolio associated with active/ loggedin user

        #return render_template('/portfolios/optimal_portfolio.jinja2', portfolio=port, plot_url=plot_data)

    return render_template('/portfolios/created_successful.jinja2')





@portfolio_blueprint.route('/delete/<string:portfolio_id>')
@user_decorators.requires_login
def delete_portfolio(portfolio_id):            # Views form to create portfolio associated with active/ loggedin user
    Database.remove("portfolios",{"_id":portfolio_id})
    user = User.find_by_email(session['email'])
    portfolios = user.get_portfolios()
    return render_template('/users/portfolios.jinja2', portfolios = portfolios, user=user)

@portfolio_blueprint.route('/test/<string:portfolio_id>')
@user_decorators.requires_login
def test_portfolio(portfolio_id):            # Views form to create portfolio associated with active/ loggedin user
    port = Portfolio.get_by_id(portfolio_id)
    port.run_logic()

    return render_template('/portfolios/weight_table.jinja2', portfolio= port)
'''
@portfolio_blueprint.route('/test/<string:portfolio_id>')
@user_decorators.requires_login
def delete_all_portfolios(portfolio_id):            # Views form to create portfolio associated with active/ loggedin user
    port = Portfolio.get_by_id(portfolio_id)
    Database.remove("portfolios", {"_id": port.user_email})
    user = User.find_by_email(session['email'])
    portfolios = user.get_portfolios()
    return render_template('/users/portfolios.jinja2', portfolios=portfolios)
'''