# Welcome to the Portfolio Backtest Project built by James Pavlicek. 
# This code is free use and anyone can use it.
# If you have any questions or inquiries feel free to reach out to me at https://www.jamespavlicek.com/ 
# To start I have a summary below of what this project will do and the code will follow below.


#0. Import all necessary python packages.
#1. Set up Streamlit app and user inputs. Start Calculations via button.
#2. Use yfinance to get stock data such as returns and standard deviation.
#3. Generate random portfolios and utilize portfolio diversification benefit to calculate portfolio standard deviation.
#4. Use Modern Portfolio Theory to identify key portfolios with the Efficient Frontier and Capital Market Line.
#5. Calculate Portfolio and Benchmark investment values and clean relevant data.
#6. Build Graphs and other resources to display to the user.
#7. Build and output a PDF of all relevant information for the end user.



#-----------------------------------------------------------#
#------------------STEP 0: Import Packages------------------#
#-----------------------------------------------------------#

import yfinance as yf
import datetime
import pytz
import streamlit as st
import pandas as pd
import numpy as np
import time
import dataframe_image as dfi
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import base64
import os
import requests



#-----------------------------------------------------------#
#----------------STEP 1: Setup Streamit App-----------------#
#-----------------------------------------------------------#

#Page Title and Description
st.title("""
Portfolio Backtest
""")
st.write("Enter your information below to get the best portfolio allocation based your age and risk tolerance.")
st.write('')
st.subheader("""
User Information      
""")

#Age and risk sliders
age = st.slider('Age', min_value=18, max_value=100, value=25, step=1) 
risk = st.slider('Risk', min_value=0, max_value=10, value=5, step=1)  
st.write('')

#Sub Title
st.subheader(""" 
Investment Information
""")

#Duration of Investment
duration = st.number_input("Years", min_value = 1, max_value = 22, value = 5, step = 1)

#Initial Investment Amount
initial_investment = st.number_input("Initial Amount", min_value = 1, max_value = 1000000000000000, value = 100000, step = 1000)

#Recurring investment
add_investment = st.toggle('Recurring Investment?')

periodic_investment_amount = None
periodicity = "Yearly"

if add_investment:
    periodic_investment_amount = st.number_input("Recurring Amount", min_value = 0, max_value = 1000000000000, value = 1000, step = 100)
    periodicity = st.selectbox(
    f'How often do you want to reinvest ${periodic_investment_amount}?',
    ('Monthly', 'Yearly'))
else:
    st.write('')

#Start the Calculations
if st.button('Calculate Results',type="primary"):

    #Notify the user that the calculations have started
    with st.spinner('Information received, ideal portfolio will be displayed shortly.'):
        time.sleep(5)

    #Convert user inputs to integers
    initial_investment = float(initial_investment)

    if periodic_investment_amount is None:
        periodic_investment_amount = 0
    else:
        periodic_investment_amount = float(periodic_investment_amount)

    #Calculate Risk score
    risk_score = ( ((118 - age)*.5) + ((risk*10)*.5) )

    # Start and end dates
    end_date = datetime.datetime.now(pytz.timezone('America/New_York'))
    start_date = (end_date - datetime.timedelta(days=duration*365))
    


    #-----------------------------------------------------------#
    #--------------STEP 2: Get Stock Information----------------#
    #-----------------------------------------------------------#

    #Set stocks based on investment duration
    if duration <=12:
        Ticker1 = "VTI"    #US Total Stock Market
        Ticker2 = "VTV"    #US Large Cap Value
        Ticker3 = "QQQ"    #US Large Cap Growth
        Ticker4 = "VO"     #US Mid Cap 
        Ticker5 = "VXUS"   #International 
        Ticker6 = "VNQ"    #REIT
        Ticker7 = "BND"    #High Yield Corporate bonds
        Ticker8 = "BIL"    #1-3 Month T-Bill

        tickers = [Ticker1,Ticker2,Ticker3,Ticker4,Ticker5,Ticker6,Ticker7,Ticker8]

        #Get stock data
        stock_data = {}
        for ticker in tickers:
            stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date)

        annual_returns = {}
        annual_std_dev = {}

        #Get daily returns, mean, and standard deviation
        for ticker, data in stock_data.items():
            
            data['Daily Returns'] = data['Adj Close'].pct_change()
            
            annual_returns[ticker] = data['Daily Returns'].mean() * 252
            annual_std_dev[ticker] = data['Daily Returns'].std() * np.sqrt(252)

         #Generate correlation matrix for the stocks
        corr_matrix = pd.DataFrame({ticker: data['Daily Returns'] for ticker, data in stock_data.items()}).corr()

    else:
    
        Ticker1 = "VTSMX"    #US Total Stock Market
        Ticker2 = "VIVAX"    #US Large Cap Value
        Ticker3 = "QQQ"      #US Large Cap Growth
        Ticker4 = "VIMSX"    #US Mid Cap 
        Ticker5 = "VGTSX"    #International 
        Ticker6 = "VGSIX"    #REIT
        Ticker7 = "VWEHX"    #High Yield Corporate bonds
        Ticker8 = "VFISX"    #1-3 Month T-Bill

        tickers = [Ticker1,Ticker2,Ticker3,Ticker4,Ticker5,Ticker6,Ticker7,Ticker8]

        #Get stock data
        stock_data = {}
        for ticker in tickers:
            stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date)

        annual_returns = {}
        annual_std_dev = {}

        #Get daily returns, mean, and standard deviation
        for ticker, data in stock_data.items():
  
            data['Daily Returns'] = data['Adj Close'].pct_change()

            annual_returns[ticker] = data['Daily Returns'].mean() * 252
            annual_std_dev[ticker] = data['Daily Returns'].std() * np.sqrt(252)

         # Compute the correlation matrix for the stocks
        corr_matrix = pd.DataFrame({ticker: data['Daily Returns'] for ticker, data in stock_data.items()}).corr()

    #Compute two stock portfolio standard deviation for 75% stock / 25% bond portfolio 
    sevenfive_correlation = corr_matrix.iloc[corr_matrix.columns.get_loc(Ticker1), corr_matrix.columns.get_loc(Ticker7)]
    sevenfive_covariance = sevenfive_correlation * (annual_std_dev[Ticker1] * annual_std_dev[Ticker7])
    sevenfive_ret = ((annual_returns[Ticker1]*.75) + (annual_returns[Ticker7]*.25))
    sevenfive_std = ((.75)**2 * (annual_std_dev[Ticker1])**2) + ((.25)**2 * (annual_std_dev[Ticker7])**2) + (2 * .75 * .25 * sevenfive_covariance)
    sevenfive_std = np.sqrt(.75**2 * annual_std_dev[Ticker1]**2 + .25**2 * (annual_std_dev[Ticker7])**2 + 2*.75*.25*sevenfive_correlation*(annual_std_dev[Ticker7])*annual_std_dev[Ticker1])

    #Get annual returns and standard deviation for the total stock market benchmark
    benchmark_ret = annual_returns[Ticker1]
    benchmark_std = annual_std_dev[Ticker1]



    #-----------------------------------------------------------#
    #-------STEP 3: Generate Portfolios and Calculations--------#
    #-----------------------------------------------------------#   

    #Generate 10000 random portfolio allocations
    num_portfolios = 10000
    results = np.zeros((4, num_portfolios))
    weights_array = []

    for i in range(num_portfolios):
        weights = np.random.random(8)
        weights /= np.sum(weights)
        weights_array.append(weights)
        
        #Portfolio return
        portfolio_return = np.sum(weights * np.array(list(annual_returns.values())))
        
        #Portfolio volatility
        portfolio_stddev = 0
        for j in range(len(tickers)):
            for k in range(len(tickers)):
                portfolio_stddev += weights[j] * weights[k] * annual_std_dev[tickers[j]] * annual_std_dev[tickers[k]] * corr_matrix.iloc[j, k]
        portfolio_stddev = np.sqrt(portfolio_stddev)
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev

        #Sharpe ratio (assuming a risk-free rate of 3%)
        results[2,i] = (portfolio_return - 0.03) / portfolio_stddev



    #-----------------------------------------------------------#
    #-------------STEP 4: Identify Best Portfolio---------------#
    #-----------------------------------------------------------#   

    #Convert results to dataframe
    results_frame = pd.DataFrame(results.T, columns=['ret','stdev','sharpe', 'weights'])
    results_frame['weights'] = weights_array

    #Find the portfolio with the highest Sharpe ratio
    best_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]

    #Graph the CML using the risk-free rate and the Sharpe ratio of the market portfolio (y = mx + c)
    risk_free_rate = 0.03
    x = np.linspace(0, max(results_frame['stdev']), 100)
    y = (best_sharpe_port[2] * x) + risk_free_rate

    #Find min and max standard deviation 
    results_frame_min_stdev = results_frame['stdev'].min()
    results_frame_max_stdev = results_frame['stdev'].max()

    #Find all the returns in a range of standard deviation based on age and risk
    stdev_range = (results_frame_max_stdev - results_frame_min_stdev)
    desired_stdev = ((stdev_range * (risk_score/100)) + results_frame_min_stdev)
    desired_stdev_min = (desired_stdev * .98)
    desired_stdev_max = (desired_stdev * 1.02)
    results_frame_desired = results_frame[results_frame['stdev'] >= desired_stdev_min]
    results_frame_desired = results_frame_desired[results_frame['stdev'] <= desired_stdev_max]
    ranked_portfolios = results_frame_desired.sort_values('ret', ascending=False)

    #Create a column in the portfolio dataframe that shows weights of each portfolio
    ranked_portfolios['weights'] = ranked_portfolios['weights'].apply(lambda x: {tickers[i]: f"{weight*100:.2f}%" for i, weight in enumerate(x)})

    #Grab key statistics from the "best portfolio"
    best_portfolio = ranked_portfolios.iloc[0]
    port_return =  best_portfolio.iloc[0]
    port_standard_deviation =  best_portfolio.iloc[1]
    port_sharpe_ratio =  best_portfolio.iloc[2]
    stocks_and_returns =  best_portfolio.iloc[3]

    #Grab the weights of each stock in the portfolio
    Stock1, Stock2, Stock3, Stock4, Stock5, Stock6, Stock7, Stock8= stocks_and_returns.keys()
    stocks_and_returns = {k: v[:-1] for k, v in stocks_and_returns.items()}
    W1, W2, W3, W4, W5, W6, W7, W8 = stocks_and_returns.values()

    W1 = float(W1)/100
    W2 = float(W2)/100
    W3 = float(W3)/100
    W4 = float(W4)/100
    W5 = float(W5)/100
    W6 = float(W6)/100
    W7 = float(W7)/100
    W8 = float(W8)/100

    #Find current price to calculate stock shopping list
    def get_current_price(ticker_symbol):
        ticker = yf.Ticker(ticker_symbol)
        todays_data = ticker.history(period='1d')
        return todays_data['Close'][0]

    Ticker11 = "VTI"    #US Total Stock Market
    Ticker21 = "VTV"    #US Large Cap Value
    Ticker31 = "QQQ"    #US Large Cap Growth
    Ticker41 = "VO"     #US Mid Cap 
    Ticker51 = "VXUS"   #International 
    Ticker61 = "VNQ"    #REIT
    Ticker71 = "BND"    #High Yield Corporate Bonds
    Ticker81 = "BIL"    #1-3 Month T-Bill

    Amount1 = W1*initial_investment
    Amount2 = W2*initial_investment
    Amount3 = W3*initial_investment
    Amount4 = W4*initial_investment
    Amount5 = W5*initial_investment
    Amount6 = W6*initial_investment
    Amount7 = W7*initial_investment
    Amount8 = W8*initial_investment

    #Builds a dataframe of the amount of each stock the user should buy
    shopping_list = pd.DataFrame({
        "Ticker" : [Ticker11,Ticker21,Ticker31,Ticker41,Ticker51,Ticker61,Ticker71,Ticker81],
        'Amount ($)': ["${:,.2f}".format(Amount1),
                      "${:,.2f}".format(Amount2),
                      "${:,.2f}".format(Amount3),
                      "${:,.2f}".format(Amount4),
                      "${:,.2f}".format(Amount5),
                      "${:,.2f}".format(Amount6),
                      "${:,.2f}".format(Amount7),
                      "${:,.2f}".format(Amount8)],
        'Amount (Shares)': [(Amount1/get_current_price(Ticker11)).round(2), 
                           (Amount2/get_current_price(Ticker21)).round(2), 
                           (Amount3/get_current_price(Ticker31)).round(2),
                           (Amount4/get_current_price(Ticker41)).round(2),
                           (Amount5/get_current_price(Ticker51)).round(2),
                           (Amount6/get_current_price(Ticker61)).round(2),
                           (Amount7/get_current_price(Ticker71)).round(2),
                           (Amount8/get_current_price(Ticker81)).round(2)]
    })

    shopping_list = shopping_list.T
    shopping_list.columns = ["1","2","3","4","5","6","7","8"]

    #Builds a recurring shopping list dataframe if there is periodic investments
    if periodic_investment_amount > 0:
        Amount11 = W1*periodic_investment_amount
        Amount21 = W2*periodic_investment_amount
        Amount31 = W3*periodic_investment_amount
        Amount41 = W4*periodic_investment_amount
        Amount51 = W5*periodic_investment_amount
        Amount61 = W6*periodic_investment_amount
        Amount71 = W7*periodic_investment_amount
        Amount81 = W8*periodic_investment_amount

        repurchase_list = pd.DataFrame({
            "Ticker" : [Ticker11,Ticker21,Ticker31,Ticker41,Ticker51,Ticker61,Ticker71,Ticker81],
            'Amount ($)': ["${:,.2f}".format(Amount11),
                        "${:,.2f}".format(Amount21),
                        "${:,.2f}".format(Amount31),
                        "${:,.2f}".format(Amount41),
                        "${:,.2f}".format(Amount51),
                        "${:,.2f}".format(Amount61),
                        "${:,.2f}".format(Amount71),
                        "${:,.2f}".format(Amount81)]})
        
        repurchase_list = repurchase_list.T
        repurchase_list.columns = ["1","2","3","4","5","6","7","8"]
    else:
        pass

    #Coverts stocks and their weights to stings 
    W1 = str(W1)
    W2 = str(W2)
    W3 = str(W3)
    W4 = str(W4)
    W5 = str(W5)
    W6 = str(W6)
    W7 = str(W7)
    W8 = str(W8)

    weight = {Stock1:[W1], Stock2:[W2], Stock3:[W3], Stock4:[W4], Stock5:[W5], Stock6:[W6], Stock7:[W7], Stock8:[W8]}

    str(Stock1)
    str(Stock2)
    str(Stock3)
    str(Stock4)
    str(Stock5)
    str(Stock6)
    str(Stock7)
    str(Stock8)

    Portfolio = [Stock1, Stock2, Stock3, Stock4, Stock5, Stock6, Stock7, Stock8]

    all_data = pd.DataFrame()



    #-----------------------------------------------------------#
    #-----STEP 5: Calculate Portfolio and Benchmark Values------#
    #-----------------------------------------------------------#   

    #Function that calculates the value of the portfolio over specified duration and investment amount 
    def PortfolioCalc(weight, data, name, initial_investment, periodic_investment_amount, periodicity):
        
        #Calculate the portfolio return percentages
        data[name] = sum([float(weight[x][0]) * data[x] for x in list(weight.keys())])
        
        #Sets the initial value as the initial investment
        data[name] = initial_investment * data[name]
        
        #Adds recurring investments as time goes on
        if periodicity == "Monthly":
            for i in range(1, len(data)):
                if data["Date"].iloc[i].month != data["Date"].iloc[i-1].month:
                    data[name].iloc[i:] += periodic_investment_amount
        elif periodicity == "Yearly":
            for i in range(1, len(data)):
                if data["Date"].iloc[i].year != data["Date"].iloc[i-1].year:
                    data[name].iloc[i:] += periodic_investment_amount
        return data
    
    #set weight of benchmark and 75/25 portfolios
    weight_100 = {Ticker1: ['1.00']}
    weight_75 = {Ticker1: ['0.75'], Ticker7 : ['0.25']}

    #Function that calculates the value of total stock market benchmark over specified duration and investment amount 
    def BenchmarkCalc(weight_100, data, name, initial_investment, periodic_investment_amount, periodicity):
        
        #Calculate the portfolio return percentages
        data[name] = sum([float(weight_100[x][0]) * data[x] for x in list(weight_100.keys())])
        
        #Sets the initial value as the initial investment
        data[name] = initial_investment * data[name]
        
        #Adds recurring investments as time goes on
        if periodicity == "Monthly":
            for i in range(1, len(data)):
                if data["Date"].iloc[i].month != data["Date"].iloc[i-1].month:
                    data[name].iloc[i:] += periodic_investment_amount
        elif periodicity == "Yearly":
            for i in range(1, len(data)):
                if data["Date"].iloc[i].year != data["Date"].iloc[i-1].year:
                    data[name].iloc[i:] += periodic_investment_amount
        return data

    #Function that calculates the value of a 75% stock and 25% bond portfolio over specified duration and investment amount 
    def SevenFiveCalc(weight_75, data, name, initial_investment, periodic_investment_amount, periodicity):
        
        # Calculate the portfolio return percentages
        data[name] = sum([float(weight_75[x][0]) * data[x] for x in list(weight_75.keys())])

        #Sets the initial value as the initial investment
        data[name] = initial_investment * data[name]

        #Adds recurring investments as time goes on
        if periodicity == "Monthly":
            for i in range(1, len(data)):
                if data["Date"].iloc[i].month != data["Date"].iloc[i-1].month:
                    data[name].iloc[i:] += periodic_investment_amount
        elif periodicity == "Yearly":
            for i in range(1, len(data)):
                if data["Date"].iloc[i].year != data["Date"].iloc[i-1].year:
                    data[name].iloc[i:] += periodic_investment_amount
        return data


    #Portfolio data cleaning
    for stock in Portfolio:
        basedata = yf.Ticker(stock).history(period="max").reset_index()[["Date", "Open"]]
        basedata["Date"] = pd.to_datetime(basedata["Date"])
        basedata = basedata.rename(columns={"Open": stock})

        if all_data.empty:
            all_data = basedata
        else:
            all_data = pd.merge(all_data, basedata, on="Date", how="outer")

    final_data = all_data[all_data["Date"] >= start_date]
    final_data = final_data[final_data["Date"] < end_date]

    #Benchmark data cleaning
    benchmark = yf.Ticker(Ticker1).history(period="max").reset_index()[["Date", "Open"]]
    benchmark["Date"] = pd.to_datetime(benchmark["Date"])
    benchmark = benchmark.rename(columns={"Open": Ticker1})
    
    benchmark = benchmark[benchmark["Date"] >= start_date]
    benchmark = benchmark[benchmark["Date"] < end_date]
    
    bench_first_value = benchmark[Ticker1].iloc[0]
    benchmark[Ticker1] = benchmark[Ticker1] / bench_first_value
    

    #75% stock / 25% bond data cleaning
    data75_25_final = pd.DataFrame()

    data75_25_1 = yf.Ticker(Ticker1).history(period="max").reset_index()[["Date", "Open"]]
    data75_25_1["Date"] = pd.to_datetime(data75_25_1["Date"])
    data75_25_1 = data75_25_1.rename(columns={"Open": Ticker1})

    data75_25_2 = yf.Ticker(Ticker7).history(period="max").reset_index()[["Date", "Open"]]
    data75_25_2["Date"] = pd.to_datetime(data75_25_2["Date"])
    data75_25_2 = data75_25_2.rename(columns={"Open": Ticker7})

    data75_25_final = pd.merge(data75_25_1, data75_25_2, on="Date", how="outer")

    data75_25_final = data75_25_final[data75_25_final["Date"] >= start_date]
    data75_25_final = data75_25_final[data75_25_final["Date"] < end_date]
  
    data75_25_final_first_value = data75_25_final[Ticker1].iloc[0]
    data75_25_final[Ticker1] = data75_25_final[Ticker1] / data75_25_final_first_value

    data75_25_final_first_value = data75_25_final[Ticker7].iloc[0]
    data75_25_final[Ticker7] = data75_25_final[Ticker7] / data75_25_final_first_value

    #Normalizes all stock values to start at 1.0
    for stock in Portfolio:
        first_value = final_data[stock].iloc[0]
        final_data[stock] = final_data[stock] / first_value
    
    #Calculate Portfolio values overtime with user inputs
    final_data = PortfolioCalc(weight, final_data, "My_Portfolio", initial_investment, periodic_investment_amount, periodicity)
    benchmark = BenchmarkCalc(weight_100,benchmark,"Benchmark", initial_investment, periodic_investment_amount, periodicity)
    sevenfive = SevenFiveCalc(weight_75, data75_25_final, "75% Stock, 25% Bonds", initial_investment, periodic_investment_amount, periodicity)



    #-----------------------------------------------------------#
    #----------------STEP 6: Graphs and Resources---------------#
    #-----------------------------------------------------------#   

    # Create a scatter plot for the Efficient Frontier
    scatter = go.Scatter(
        x=results_frame['stdev'],
        y=results_frame['ret'],
        mode='markers',
        hoverinfo='text',
        name='Random Portfolios',
        showlegend=False,
        marker=dict(size=4,color=results_frame['sharpe'],colorscale='Blues'),
        text=[f'{Ticker1}: {weights[0]*100:.2f}%, {Ticker2}: {weights[1]*100:.2f}%, {Ticker3}: {weights[2]*100:.2f}%, {Ticker4}: {weights[3]*100:.2f}%, {Ticker5}: {weights[4]*100:.2f}%, {Ticker6}: {weights[5]*100:.2f}%, {Ticker7}: {weights[6]*100:.2f}%, {Ticker8}: {weights[7]*100:.2f}%'
           for weights in results_frame['weights']])

    #Builds the Capital Market Line for the Efficient Frontier graph
    line = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='grey'),
        name='Capital Market Line (CML)')

    layout = go.Layout(
        title='Efficient Frontier with Capital Market Line (CML)',
        xaxis=dict(title='Portfolio Risk (Standard Deviation)'),
        yaxis=dict(title='Portfolio Return'),
        showlegend=True)

    eff_fig = go.Figure(data=[scatter, line], layout=layout)
    

    #Creates points on the efficient frontier for "Your Portfolio", the total stock market benchmark and the 75% stock / 25% bond portfolio
    eff_fig.add_scatter(x=[port_standard_deviation],
                        y=[port_return],
                        marker=dict(color='yellow',size=15,symbol=217),name='Your Portfolio',line=dict(width=0,color='black'))

    eff_fig.add_scatter(x=[benchmark_std],
                        y=[benchmark_ret],
                        marker=dict(color='magenta',size=12, symbol='diamond'),name='Total Stock Market',line=dict(width=0,color='black')) 
    
    eff_fig.add_scatter(x=[sevenfive_std],
                        y=[sevenfive_ret],
                        marker=dict(color='red',size=12, symbol='diamond'),name='75% Stock / 25% Bonds',line=dict(width=0,color='black'))

    #Creates a graph that shows value over time for the three portfolios
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=final_data["Date"], y=final_data["My_Portfolio"],
                            mode='lines',name='Your Portfolio'))
    
    fig2.add_trace(go.Scatter(x=benchmark["Date"], y=benchmark["Benchmark"],
                            mode='lines', name='Total Stock Market',opacity=0.4))
    
    fig2.add_trace(go.Scatter(x=sevenfive["Date"], y=sevenfive["75% Stock, 25% Bonds"],
                            mode='lines', name='75% Stock / 25% Bonds',opacity=0.4))
    
    fig2.update_layout(template="plotly_white",
                    title="Portfolio Outlook",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    showlegend=True)

    #Shows Fig2 (value over time graph) to the user
    st.divider()
    st.subheader("""               
    Results
    """)
    st.plotly_chart(fig2)

    #Building a dataframe to show the user how their portfolio compares against the two benchmarks 
    compare_value_1 = final_data["My_Portfolio"].iloc[-2].round(2)
    compare_value_2 = benchmark["Benchmark"].iloc[-2].round(2)
    compare_value_3 = sevenfive["75% Stock, 25% Bonds"].iloc[-2].round(2)
    
    compare_ret_1 = ((compare_value_1 / initial_investment)**(1/duration))-1
    compare_ret_1 = "{:.2f}%".format(compare_ret_1*100)
    compare_ret_2 = ((compare_value_2 / initial_investment)**(1/duration))-1
    compare_ret_2 = "{:.2f}%".format(compare_ret_2*100) 
    compare_ret_3 = ((compare_value_3 / initial_investment)**(1/duration))-1
    compare_ret_3 = "{:.2f}%".format(compare_ret_3*100) 

    compare_std_1 = "{:.2f}%".format(port_standard_deviation * 100)
    compare_std_2 = "{:.2f}%".format(benchmark_std * 100)
    compare_std_3 = "{:.2f}%".format(sevenfive_std * 100)

    compare_sharpe_1 = port_sharpe_ratio.round(3)
    compare_sharpe_2 = (benchmark_ret - risk_free_rate ) / annual_std_dev[Ticker1]
    compare_sharpe_2 = compare_sharpe_2.round(3)
    compare_sharpe_3 = (sevenfive_ret - risk_free_rate ) / sevenfive_std
    compare_sharpe_3 = compare_sharpe_3.round(3)

    compare_min_1 = final_data["My_Portfolio"].min().round(2)
    compare_min_2 = benchmark["Benchmark"].min().round(2)
    compare_min_3 = sevenfive["75% Stock, 25% Bonds"].min().round(2)

    compare_max_1 = final_data["My_Portfolio"].max().round(2)
    compare_max_2 = benchmark["Benchmark"].max().round(2)
    compare_max_3 = sevenfive["75% Stock, 25% Bonds"].max().round(2)

    portfolio_names = ("Your Portfolio", "Total Stock Market", "75% Stock / 25% Bond")

    port_compare = pd.DataFrame({
        "Value" : ["${:,.2f}".format(compare_value_1), "${:,.2f}".format(compare_value_2), "${:,.2f}".format(compare_value_3)],
        'Return (CAGR)': [compare_ret_1, compare_ret_2, compare_ret_3],
        'Risk (St. Dev.)': [compare_std_1, compare_std_2, compare_std_3],
        'Sharpe Ratio': [compare_sharpe_1, compare_sharpe_2, compare_sharpe_3],
        'Min' : ["${:,.2f}".format(compare_min_1),  "${:,.2f}".format(compare_min_2), "${:,.2f}".format(compare_min_3)],
        'Max': ["${:,.2f}".format(compare_max_1),"${:,.2f}".format(compare_max_2),  "${:,.2f}".format(compare_max_3)]
    })
    port_compare.index = portfolio_names

    #Builds a pie graph of the portfolio's allocation to different asset classes and displays it to the user
    st.write("\n**Portfolio Comparison**\n", port_compare)

    xyz =list(stocks_and_returns.values()) 
    pie_chart_labels = (['US Total Stock Market', 'US Large Cap Value', 'US Large Cap Growth', 'US Mid Cap', 'International', 'REIT', 'Corporate Bonds', '1-3 Month T-Bill'])

    fig3 = px.pie(values=xyz, names=list(pie_chart_labels), color_discrete_sequence=px.colors.sequential.Blues_r)
    fig3.update_layout(title_text="Your Portfolio Allocation")

    st.plotly_chart(fig3)

    #Displays the Efficient Frontier chart and correlation matrix the user
    st.divider()
    st.subheader("Advanced Stats")
    st.plotly_chart(eff_fig)
    st.write("\n**Correlation Matrix:**\n", corr_matrix)

    #Gets the value over time graph ready for the pdf document
    fig2.update_layout(legend_font=dict(color='black'))
    fig2.update_layout(xaxis_tickfont=dict(color='black'), yaxis_tickfont=dict(color='black'))
    fig2.update_layout(xaxis_tickfont=dict(color='black'), yaxis_tickfont=dict(color='black'))
    fig2.update_layout(xaxis_title_font=dict(color='black'), yaxis_title_font=dict(color='black'))
    fig2.update_layout(title_text="")

    #Gets the  pie chart chart ready for the pdf document
    fig3.update_layout(legend_font=dict(color='black'))
    fig3.update_layout(title_x=0, title_xanchor='left')
    fig3.update_layout(title_text="")

    #saves both graphs as .png
    fig2.write_image("fig2.png", engine='kaleido')
    fig3.write_image("fig3.png", engine='kaleido')

    #saves 3 dataframes as .png
    dfi.export(port_compare, 'portfolio_compare.png',table_conversion='matplotlib')
    dfi.export(shopping_list, 'shopping_list.png',table_conversion='matplotlib')

    if periodic_investment_amount > 1:
        dfi.export(repurchase_list, 'repurchase_list.png',table_conversion='matplotlib')
    else:
        pass



    #-----------------------------------------------------------#
    #-------------STEP 7: PDF Creation and Output---------------#
    #-----------------------------------------------------------#  

    #Creates download link for the report
    def create_download_link(val, filename):
        b64 = base64.b64encode(val)
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download Report (PDF)</a>'
    
    #Starts a new PDF
    #Notes: PDF Width = 215.9, Length = 279.4
    pdf = FPDF('P', 'mm', 'letter')
    pdf.add_page()
    pdf.set_font('Helvetica', '', 14)

    #Download the background image for page 1 and place it on the PDF
    image_url = "https://i.ibb.co/Y8ynxvW/Blue-and-White-Professional-Letterhead-1.png"
    response = requests.get(image_url)
    temp_file = "temp_image.png"
    with open(temp_file, 'wb') as f:
        f.write(response.content)
    pdf.image(temp_file, x=0, y=0, w=215.9)
    os.remove(temp_file)

    #Print the portfolio comparison and stats on the PDF
    pdf.image("fig2.png",10,70,190)
    pdf.image('portfolio_compare.png',10,210,190)
    pdf.text(10,85, "Portfolio Performance")
    
    #Start of Page two on the PDF
    pdf.add_page()

    #Download the background image for page 2 and place it on the PDF
    image_url2 = "https://i.ibb.co/4g1Zdz1/Blue-and-White-Professional-Letterhead-2.png"
    response2 = requests.get(image_url2)
    temp_file2 = "temp_image2.png"
    with open(temp_file2, 'wb') as f:
        f.write(response2.content)
    pdf.image(temp_file2, x=0, y=0, w=215.9)
    os.remove(temp_file2)

    #Prints the shopping list and recurring investment list onto the pdf
    pdf.image('shopping_list.png',10,163,190)
    pdf.set_font('Helvetica', '', 14)
    pdf.text(10,158, "Stock Shopping List")

    if periodic_investment_amount > 1:
        pdf.text(10,210, f"Stock Repurchase List ({periodicity})")
        pdf.image('repurchase_list.png',10,215,150)
    else:
        pass

    #Prints the pie chart onto the PDF
    pdf.image("fig3.png",10,25,180)
    pdf.text(10,20, "Portfolio Allocation")

    #Cleans values and prints a user input summary
    initial_investment = int(initial_investment)
    periodic_investment_amount = int(periodic_investment_amount)
    pdf.set_font('Helvetica', '', 8)
    pdf.text(10,252,f"Investment Details: Age = {age}, Risk = {risk}, Duration = {duration} years, Initial Amount = {initial_investment}, Recurring Amount = {periodic_investment_amount}, Periodicity = {periodicity}" )
    
    #Creates a download link for the PDF
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "portfolio_backtest_report")

    #Notifies the user that the pdf report can be found below
    st.write('')
    st.write("**Download your Portfolio Report below:**") 
    st.markdown(html, unsafe_allow_html=True)
    
    #Writes version and builder's information
    st.divider()          
    st.write("Built by James Pavlicek. More info at jamespavlicek.com")
    st.write("Version 0.1.0")

#Finishes if/else "Calculate Results" button click
else:
    st.write('')


#  Functions of Version 1
#  1. Online application that receives user and investment information to create an ideal portfolio bases on risk, age, and relevant investment information.
#  2. Outputs a portfolio outlook that compares the performance of the portfolio to other common options.
#  3. Displays advance stats (Efficient Frontier and Correlation Matrix) for more in-depth explanation to the user.
#  4. Outputs a PDF so it is easy for the user to save the information and take it with them.

#  Ideas for Version 2
#  1. Monte Carlo Simulation (Retirement)
#  2. Custom Portfolio Simulation Amount (default set to 10,000)
#  3. Input custom stock variables
#  4. Display Income per period 
#  5. Geometric Mean, Downside Deviation and max drawdowns per year stats
#  6. Input your own options for Benchmarks
#  7. Increase investment duration to longer than 22 years
