import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import pandas as pd
import json 
import requests
from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
from scipy.integrate import odeint
from scipy.optimize import curve_fit

app = dash.Dash(__name__)

def countries_data():
    
    countries = []
    codes = []
    response= requests.get("https://corona.lmao.ninja/v2/countries?yesterday&sort")
    data = json.loads(response.text)

    for each in data:
        if each['countryInfo']['iso3'] != None:
            countries.append(each['country'])
            codes.append(each['countryInfo']['iso3'])
        else:
            countries.append(each['country'])
            codes.append(each['country'])
    return countries, codes  

def get_each_country_data(country):

    confirmed_data = []
    recovered_data = []
    death_data =[]
    dates = []

    
    response = requests.get(f"https://api.covid19api.com/total/country/{country}")
    if response.status_code != 200:
        response = requests.get(f"https://api.covid19api.com/total/country/Germany")
    data = json.loads(response.text)
    

    for each in data:
        confirmed_data.append(each['Confirmed'])
        recovered_data.append(each['Recovered'])
        death_data.append(each['Deaths'])
        dates.append(each['Date'])
    
    return confirmed_data, recovered_data, death_data, dates

def sir_simulations( confirmed_data, recovered_data, dates):
 
    length = 60 # duration for simulations 

    confirmed_data = confirmed_data[(len(confirmed_data)-length):]

    recovered_data = recovered_data[(len(recovered_data)-length):]

    dates = dates[ len(dates)-length: ]
    N = 1000000
    I_0 = confirmed_data[0]
    R_0 = recovered_data[0]
    S_0 = N - R_0 - I_0

    def SIR(y, t, beta, gamma):    
        S = y[0]
        I = y[1]
        R = y[2]
        return -beta*S*I/N, (beta*S*I)/N-(gamma*I), gamma*I

    def fit_odeint(t,beta, gamma):
        return odeint(SIR,(S_0,I_0,R_0), t, args = (beta,gamma))[:,1]

    t = np.arange(len(confirmed_data))
    params, cerr = curve_fit(fit_odeint,t, confirmed_data)
    beta,gamma = params
    prediction = list(fit_odeint(t,beta,gamma))


    fig = go.Figure()
    fig.add_trace(go.Scatter(x= dates, y= prediction,
                        mode='lines+markers',
                        name='Simulated'))
    fig.add_bar(x = dates, y= confirmed_data, name = "Actual")
    fig.update_layout(height = 800)

    return fig

countries, codes = countries_data()

app.layout = html.Div(children=[
    html.H3("COVID19 PANDEMIC",style = { "textAlign" : "Center"}),
    html.Div([
        dcc.Dropdown(
        id="country",
        options=[ {'label': country_name, 'value': code } for country_name, code in zip(countries, codes)],
        value='DEU'
    ),
    ]),
    html.Div([html.Table([
        html.Tr([

            html.Td( [html.H4("Confirmed"), html.Div(id = "total_confirmed"),]),
            
            html.Td( [html.H4("Recovered"), html.Div(id = "total_recovered")]),
            
            html.Td( [html.H4("Deaths"), html.Div(id = "total_deaths")]),
            
            html.Td( [html.H4("Active"), html.Div(id = "total_active")]),

        ]),], style = { "width" : "800px" , "marginLeft" : "auto", "marginRight" : "auto", "textAlign" : "Center"}


    )]),
    html.Div([dcc.Graph(id = 'country_confirmed')]),
    html.Div([dcc.Graph(id = 'country_recovered')]),
    html.Div([dcc.Graph(id = 'country_deaths')]),
    html.Div([dcc.Graph(id = 'sir')]),

])

@app.callback(
    [Output(component_id='country_confirmed', component_property='figure'),
    Output(component_id='country_recovered', component_property='figure'),
    Output(component_id='country_deaths', component_property='figure'),
    Output(component_id='total_confirmed', component_property='children'),
    Output(component_id='total_recovered', component_property='children'),
    Output(component_id='total_deaths', component_property='children'),
    Output(component_id='total_active', component_property='children'),
    Output(component_id='sir', component_property='figure')],

    [Input(component_id='country', component_property='value')]
)
def country_graphs(country):

    confirmed_data, recovered_data, death_data, dates = get_each_country_data(country)

    sir_figure = sir_simulations( confirmed_data, recovered_data, dates)

    confirmed = px.line(x = dates ,y = confirmed_data, labels = { "x" : "Date", "y" : "Confirmed cases"}, height = 800)
    confirmed.update_layout(title_text = " Confirmed cases" ,title_x=0.5)

    recovered = px.line(x = dates ,y = recovered_data, labels = { "x" : "Date", "y" : "Recovered cases"}, height = 800)
    recovered.update_layout(title_text = " Recovered cases" ,title_x=0.5)

    deaths = px.line(x = dates ,y = death_data,labels = { "x" : "Date", "y" : "Deaths"} ,height = 800)
    deaths.update_layout(title_text = " Deaths" ,title_x=0.5)

    active = confirmed_data[-1] - recovered_data[-1] - death_data[-1]

    return confirmed, recovered, deaths, confirmed_data[-1], recovered_data[-1], death_data[-1], active, sir_figure


if __name__ == '__main__':
    app.run_server(debug=True)
