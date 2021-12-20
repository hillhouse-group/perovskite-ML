#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 23:18:33 2020
@author: Preetham
"""

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
import psutil
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import shutil

filename = "electronic_measurement_data.csv"
ld_time = 16
t_step = 0.2
#dummy = "C:/Users/Preetham/Desktop"
n_ts = int(ld_time/t_step)
ts = np.linspace(0, ld_time-t_step, num=n_ts)

# Variables which change with users' input
data = None
data_arr = None
csv_file_path = None
file_status = False
settings = {
        'frequency' : 10,
        'machine' : 'SRS',
        'frac' : 0.75,
        'gradients_dropdown' : '0',
        'gradient_points_dropdown': '0',
        'slider': 1,
}

n_gradients = 0
n_gradpoints = 0
default_fontsize='15px'

font_family = ['Roboto Condensed', "Helvetica Neue", 'Helvetica' , 'Arial' , 'sans-serif']
external_stylesheets = [
                        'https://codepen.io/chriddyp/pen/bWLwgP.css',
                        #'https://codepen.io/Coffee2Code/pen/YeZGjR.css',
                        #'bootstrap.css',
                        #'styles.css',
                        #'main.css',
                        #'design.css'
                        ]
external_scripts=[
      'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML',
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)


app.layout = html.Div(children=[
    
    # Heading -----------------Title starts------------------------------------
    html.Div(id='main_title', children=[
    html.H1('Dynamic Visualization of Ld Measurements',
            style={'font-family': font_family,
                   'margin':'auto',
                   'color': '#FFFFFF',
                   'padding': '35px',
                   'text-align':'center',
                    }),

    ],
    style={'z-index':'0', 'background-color':'#40136B', 'width':'100%', 'padding':'0'}
    ),
    # -----------------------------Title ends----------------------------------

    # Folder path ----------------File open starts-----------------------------
    html.Div(children=[
    html.Div([
    html.Label('Folder Path : ', style={'font-size':'20px'}),
    dcc.Input(
        id='folder_path',
        placeholder='Enter the folder path here',
        style={'font-size':'20px', 'z-index':'1'}
    ),
    dcc.Loading(
        id="loading1",
        children=[
                html.Div(id='csv_found', style={'font-style':'italic'})
        ],
        type="circle"),
    ], style={'display':'inline-block', 'position':'relative',
                'top':'0px', 'z-index':'1'}),
    
    html.Div([
        html.Button(id='refresh_button', n_clicks=0, children='Refresh',
                style={'font-size':'15px', 'margin':'1px 20px 1px 20px',
                       'background-color':'#FFFFFF', 'z-index':'1',
                       'display':'inline'}
        ),
        dcc.Loading(
            id="loading_refresh",
            children=[
                    html.Div(id='refresh_done', children='Refreshed',
                             style={'font-style':'italic', 'visibility':'hidden', 'position':'relative',
                                    'left':'25px'})
            ],
            type="circle"),
    ], style={'display':'inline-block', 'position':'relative',
                'top':'1px', 'width':'300px', 'z-index':'1'}),
    
    
    html.Div([
    html.Label('Disk Usage', style={'font-size':'20px'}),
    html.Div([html.Label('Total : ', style={'display':'inline-block'}),
              html.Div(id='total_space', style={'display':'inline-block', 'font-color':'#28016b', 'font-size':'20px'})]),
    html.Div([html.Label('Free : ', style={'display':'inline-block'}),
              html.Div(id='free_space', style={'display':'inline-block', 'font-color':'#03610b', 'font-size':'20px'})]),
    dcc.Interval(
            id='disk_usage_update',
            interval=5*1000
    ),
    ], style={'display':'inline-block', 'left':'45%', 'position':'relative',
            'font-size':default_fontsize, 'top':'-13px'})
    ],

    style={'font-family': font_family, 'background-color':'#F6C93B',
           'padding':'25px', 'z-index':'0'}
    
    ),
    #------------------------------- File open ends ----------------------------    

    # MAIN BLOCK ---------------------Body section starts---------------------------
    html.Div(children=[
    
    # -------------------------------- Settings starts -------------------------
    html.Div(children=[
    
    html.Div(children=[
    html.Label('Frequency : ', style={'font-size':default_fontsize, 'font-style':'bold', 'display':'inline-block'}),
    dcc.RadioItems(
        id='frequency',
        options=[{'label': 'DC', 'value': 0}, {'label': '10 Hz', 'value': 10}, {'label': '400 Hz', 'value': 400}],
        value=settings['frequency'],
        labelStyle={'display': 'inline-block', 'font-size':default_fontsize, 'margin':'2px'},
        inputStyle={'top':'8px', 'border':'0px', 'width':'2em', 'height':'2em', 'position':'relative'},
        style={'padding':'10px', 'display': 'inline-block'}
    ),
    ]),
    
    html.Div(children=[
    html.Label('Measuring Device : ', style={'font-size':default_fontsize, 'font-style':'bold', 'display':'inline-block'}),
    dcc.RadioItems(
        id='machine',
        options=[{'label': 'Keithley', 'value': 'keithley'}, {'label': 'SRS', 'value': 'SRS'}],
        value=settings['machine'],
        labelStyle={'display': 'inline-block', 'font-size':default_fontsize, 'margin':'2px'},
        inputStyle={'top':'8px', 'border':'0px', 'width':'2em', 'height':'2em', 'position':'relative'},
        style={'padding':'10px', 'display': 'inline-block'}
    ),
    ]),
    
    html.Div([
    html.Label('Frac. of initial value required to visualize ',
               style={'font-size':default_fontsize, 'font-style':'bold', 'display':'inline-block'}),
    dcc.Input(
        id='frac',
        placeholder='Enter a number between 0 and 1',
        value=settings['frac'],
        style={'font-size':default_fontsize, 'padding':'10px',
               'margin':'10px', 'display':'inline-block', 'width':'25%', 'position':'relative'},
    ),
    ]),
    
    html.Div([
    html.Div([
    dcc.Slider(id='slider',
        min=0,
        max=1,
        step=None,
        marks={
            0: {'label':''},
            1: {'label':''}
        },
        value=settings['slider'],
    )], style={'display':'inline', 'width':'50%', 'height':'50px',
                'position':'relative', 'top':'18px'},
    ),
    html.Div(children='Choose positions manually',
             style={'box-color':'#40136B', 'text-align':'left', 'display':'inline-box',
                    'position':'relative', 'top' :'0px'}),
    html.Div(children='Show all positions',
             style={'box-color':'#40136B','text-align':'right', 'display':'inline-box',
                    'position':'relative', 'top' :'-24px'}),
    ], style={'padding':'10px', 'position':'relative',
                'border': '2px solid #FFFFFF', 'border-radius':'2px'},
    ),
    html.Hr(),
    
    # dropdown lists
    html.Div(id='dropdown_div', children=[
    html.Label('Gradient Index', style={'font-size':default_fontsize, 'font-style':'bold'}),
    dcc.Dropdown(
        id='gradients_dropdown',
        options=[{'label': str(i), 'value': i} for i in range(n_gradients)],
        value=settings['gradients_dropdown'],
    ),
    
    html.Div(id='gradpoints_div', children=[
    html.Label('Position Index', style={'font-size':default_fontsize, 'font-style':'bold'}),
    dcc.Dropdown(
        id='gradient_points_dropdown',
        options=[{'label': str(i), 'value': i} for i in range(n_gradpoints)],
        value=settings['gradient_points_dropdown'],
    )]),
    ], style={'display':'block'}),
    
    # Submit button        
    html.Button(id='submit_to_plot', n_clicks=0, children='Submit',
                style={'font-size':'15px', 'margin':'20px', 'background-color':'#FFFFFF'}
                ),
    ],
    
    style={'background-color':'#F6C93B', 'display' : 'inline-block',
           'padding':'25px', 'width':'30%', 'z-index':'0',
           'margin':'auto', 'height':'600px', 'top':'494px', 'position':'relative',
            },
    ),
    
    # ----------------------------- Settings ends ------------------------------
    

    # ------------------------------ Plot starts --------------------------------
    html.Div(children=[
    html.Div(id='plot_title_main_div',
             children=[html.Div(id='plot_title',
                        style={'padding':'5px',
                               'text_align':'center',
                               'display':'inline-block', 'width':'100%',
                               'position':'relative', 'font-size':'20px'})],
            style={'background-color':'#FFFFFF',}
    ),
    dcc.Loading(
        id="loading2",
        children=[
                dcc.Graph( id='graph', style={'padding':'5px'})
        ],
        type="circle"),
    ],
    style={'background-color':'#F6C93B', 'padding':'25px',
           'width':'60%', 'height':'600px', 'display' : 'inline-block',
           'margin':'5px'},
    )
    ],
    style={'padding' : '1px', 'margin':'auto', 'width':'100%', 'height':'600px',
           'position':'relative', 'top':'-488px', 'z-index':'0'}, 
    ),
    #--------------------------------- Plot ends -----------------------------------
    
    # ----------------------------Body section ends---------------------------------
    
    # Empty space
    html.Div(style={'height':'200px', 'width':'100%', 'background-color':'#FFFFFF'})
    
    
])


# CALLBACK FUNCTIONS BELOW ----------------------------------------------------

# Update the availabel space
@app.callback([Output('total_space', 'children'), Output('free_space', 'children')],
               [Input('disk_usage_update', 'n_intervals')])
def get_disk_space(dummy_arg):
    """Updates the disk space values"""
    usage = psutil.disk_usage('/')
    return str(round(usage[0]*1e-9, 2)) + ' Gb', str(round(usage[2]*1e-9, 2)) + ' Gb'


# read the input for folder path and update the data df, which is None otherwise
@app.callback([Output('csv_found', 'children'),
               Output('gradients_dropdown', 'options'),
               Output('gradient_points_dropdown', 'options')],
            [Input('refresh_button', 'n_clicks'),],
            [State('folder_path', 'value'),])
def read_csv(refresh_clicks, folder_path):
    """Reads the electronic csv file from the run's folder"""
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    global file_status, filename
        
    if 'folder_path' in changed_id or 'refresh_button' in changed_id:
        files = os.listdir(folder_path)
        global csv_file_path, data_arr
        
        if filename in files:
            print('file found')
            file_path = os.path.join(folder_path, filename)
            file_path_copy = os.path.join(folder_path, 'copy_'+filename)
            shutil.copyfile(file_path, file_path_copy)
            csv_file_path = file_path_copy
            file_status = True
            data_arr = None
    
    global data, n_gradients, n_gradpoints
    
    if file_status:
        df = pd.read_csv(csv_file_path)
        data = df.copy()
        n_gradients = get_n_gradients(df)
        n_gradpoints = get_n_gradpoints(df)
        
    # set Gradient options for dropdown
    grad_dropdown_options = [{'label': str(i), 'value': i} for i in range(n_gradients)]
    gradpoints_dropdown_options = [{'label': str(i), 'value': i} for i in range(n_gradpoints)]
    
    if file_status:
        return 'Electronic CSV file found.', grad_dropdown_options, gradpoints_dropdown_options
    else:
        return 'NO Electronic CSV file found.', grad_dropdown_options, gradpoints_dropdown_options

# Refresh the data array
@app.callback([Output('refresh_done', 'style'),
               Output('refresh_done', 'children')],
               [Input('refresh_button', 'n_clicks')]) 
def refresh_csv(refresh_clicks):
    """Refreshes the electronic csv data"""
    
    style={'font-style':'italic', 'visibility':'hidden', 'display':'inline-block',
           'position':'relative', 'left':'25px'}
    if refresh_clicks>0:
        style['visibility']='visible'
        return style, 'Refreshed'
    else:
        style['visibility']='hidden'
        return style, 'Not Refreshed'



# Make the plot
@app.callback([Output('graph', 'figure'), Output('plot_title', 'children')],
              [Input('submit_to_plot', 'n_clicks')],
              [State('frequency', 'value'), State('machine', 'value'), State('frac', 'value'),
               State('gradients_dropdown', 'value'), State('gradient_points_dropdown', 'value'),
               State('slider', 'value')])
def make_plot(n_clicks, freq, machine, frac, grad_num, loc_num, dropdown_status):
    """Makes the plot based on callback"""
    global data, settings
    fig_dict = {}
    title = 'For plot, click Submit'
    settings_changed = False
    
    if freq != settings['frequency']:
        settings['frequency'] = freq
        settings_changed = True
    if machine != settings['machine']:
        settings['machine'] = machine
        settings_changed = True
    if frac != settings['frac']:
        settings['frac'] = frac
        settings_changed = True
        
    if str(grad_num) != settings['gradients_dropdown']:
        settings['gradients_dropdown'] = str(grad_num)    
    if str(loc_num) != settings['gradient_points_dropdown']:
        settings['gradient_points_dropdown'] = str(loc_num)
    if dropdown_status != settings['slider']:
        settings['slider'] = dropdown_status
    
            
    if not data is None and n_clicks>0:
        if int(dropdown_status) == 0:
            fig_dict, title = make_sqrt_I_plot(data, freq=freq, machine=machine,
                                               grad_num=grad_num, loc_num=loc_num,
                                               frac=frac, settings_changed=settings_changed)
        else:
            fig_dict, title = make_sqrt_I_plot_allpos(data, freq=freq, machine=machine,
                                                      grad_num=grad_num,
                                                      frac=frac,
                                                      settings_changed=settings_changed)
    
    return fig_dict, title
    
# update dropdown list
@app.callback(Output('gradpoints_div', 'style'), [Input('slider', 'value')])
def update_dropdown(status):
    """Updates the dropdown lists' visibility"""
    if status == 1:
        return {'visibility':'hidden'}
        
    else:
        return {'display':'block'}
    

# UTILITY FUNCTIONS ------------------------------------------------------------
    
def get_sqrt_I_avg(df, cycle_num, freq=10, machine='SRS', grad_num=0, loc_num=0):
    """Square root of median SRS current reading from the cycle num.
    grad_num, loc_num and cycle_num must be indices starting from 0
    """
    freq = int(freq)
    
    if freq == -1: # Dark conductivity
        freq_ind = 0
    elif freq == 0: # DC photoconductivity
        freq_ind = 1
    elif freq == 10: # 10 Hz
        freq_ind = 2
    elif freq == 400: # 400 Hx
        freq_ind = 3
    
    ngrads = get_n_gradients(df)
    ngradpoints = get_n_gradpoints(df)
    
    # Number of gradients
    grad_num = int(grad_num)
    loc_num = int(loc_num)
    cycle_num = int(cycle_num)
    
    start = int(cycle_num*ngrads*ngradpoints*n_ts + grad_num*ngradpoints*n_ts + loc_num*n_ts + int(4/t_step)*freq_ind - 1)
    end = int(cycle_num*ngrads*ngradpoints*n_ts + grad_num*ngradpoints*n_ts + loc_num*n_ts + int(4/t_step)*(freq_ind+1) - 2)
    if machine == 'SRS':
        I_data = df.iloc[start:end, 5].values
    elif machine == 'keithley':
        I_data = np.abs(df.iloc[start:end, 4].interpolate().values)
        
    return np.sqrt(np.median(I_data))


def update_cycles(df):
    return "Current cycle : {}".format(get_current_cycle(df))

def get_n_gradients(df):
    """Returns the number of gradients if any"""
    return df.iloc[:, 1].unique().astype(int)[-1]+1

def get_n_gradpoints(df):
    """Returns the number of gradients if any"""
    return df.iloc[:, 2].unique().astype(int)[-1]+1

def update_global_gradpoints(df):
    """Updated the global variables"""
    global n_gradients, n_gradpoints
    n_gradients = get_n_gradients(df)
    n_gradpoints = get_n_gradpoints(df)

def get_current_cycle(df):
    """Returns the current cycle index"""
    return df.iloc[:,0].unique().astype(int)[-1]

def make_sqrt_I_plot(df, freq=10, machine='SRS', grad_num=0,
                     loc_num=0, frac=0.75,  settings_changed=False):
    """Makes the Ld plot"""
    last_cycle = get_current_cycle(df)
    x = np.arange(last_cycle+1).astype(int)
    grad_num = int(grad_num)
    loc_num=int(loc_num)
    freq = int(freq)
    
    if (not data_arr is None) and np.sum(data_arr[grad_num]) != 0 and not (settings_changed):
        y = data_arr[grad_num, loc_num, :]
    else:
        y = np.array([get_sqrt_I_avg(df, i, freq=freq, machine=machine, grad_num=grad_num, loc_num=loc_num) for i in x])
        y = weighted_mov_avg(y.reshape((None,1)))
    title = 'sqrt(I)' + ' ('+machine+')' + ' | Last cycle : ' + str(int(last_cycle))
    xlabel = 'Cycle'
    ylabel = '$\sqrt{I}$'
    
    frac = float(frac)
    frac_val = frac*y[0]
    fig_dict = {
            'data':[{'type':'scatter', 'x':x, 'y':y, 'name':ylabel},
                    {'x':[x[0], x[-1]],'y':[frac_val, frac_val], 'name':str(format(frac*100))+'% val.'}],
            'layout':{
                    'xaxis':{'title':xlabel},
                    'yaxis':{'title':ylabel},
            }
    }
    return fig_dict, title

def make_sqrt_I_plot_allpos(df, freq=10, machine='SRS',
                            grad_num=0, frac=0.75, settings_changed=True):
    """Makes the sqrt_I plot for all positions"""
    
    n_cycles = get_current_cycle(df)
    global data_arr
    grad_num = int(grad_num)
    freq = int(freq)
    frac = float(frac)
    
    if data_arr is None or settings_changed:
        data_arr = np.zeros((n_gradients, n_gradpoints, n_cycles))
        for i in range(n_gradpoints):
            for j in range(n_cycles): # only loops till the 2nd cycle from last
                data_arr[grad_num, i, j] = get_sqrt_I_avg(df, j, freq=freq, machine=machine, grad_num=grad_num, loc_num=i)
        data_arr[grad_num] = (weighted_mov_avg(data_arr[grad_num].transpose())).transpose()
    data_arr_plot = data_arr[grad_num] / data_arr[grad_num][:, 0, None]
    data_arr_plot = np.abs(data_arr_plot - frac)
    
    
    title = 'The dark blue (0) marks Ld'+str(int(100*frac))+ ' is hit ('+machine+')' + ' | Last cycle : ' + str(int(n_cycles))
    xlabel = 'Cycle'
    ylabel = 'Location index on gradient'
    colorbar_label = 'abs. dist. metric\nfrom Ld'+str(int(100*frac))
    
    fig = go.Figure(data=go.Heatmap(
                    z=data_arr_plot,
                    x=np.arange(n_cycles),
                    y=np.arange(n_gradpoints),
                    colorscale='thermal',
                    zmin=0,
                    colorbar={'title':colorbar_label}
    ))
    fig.update_layout(xaxis={'title':xlabel}, yaxis={'title':ylabel})
    
    return fig, title

def weighted_mov_avg(arr, weights=[1,2,1]):
    """Removes noise and smoothens the curves by weighted moving average
    Also, the number of weights must be odd."""
    
    df = pd.DataFrame(data=arr)
    
    if len(df.columns)<=n_gradpoints:
        df.insert(0, 'dummy', df.index)
    
    weights = np.array(weights)
    weights = weights/np.sum(weights)
    bin_rad = int(len(weights)/2)  #The bin radius for mov. avg
    
    data = df.values #data as np array
    
    start = bin_rad #starting index for mov. avg
    end = df.values.shape[0] - bin_rad - 1 #ending index for mov. avg
    
    for i in range(start, end+1):
        #All columns except time
        data[i, 1:] = np.matmul(weights, data[i-bin_rad:i+bin_rad+1, 1:])
        
    #For the ends, use linear fit extrapolation
    def lin_interpolate(x,y_cols):
        w = np.polynomial.polynomial.polyfit(x, y_cols, 1)
        return np.transpose(np.polynomial.polynomial.polyval(x, w))
    
    #beginning of df
    data[:start+bin_rad, 1:] = lin_interpolate(data[:start+bin_rad, 0],
                                                data[:start+bin_rad, 1:])
    #ending of df
    data[end-bin_rad:, 1:] = lin_interpolate(data[end-bin_rad:, 0],
                                                data[end-bin_rad:, 1:])
    
    if 'dummy' in df.columns:
        data = data[:,1:]
        df.drop(columns='dummy', inplace=True)
    
    return pd.DataFrame(data=data, columns=df.columns).values


if __name__ == '__main__':
    app.run_server(debug=False)
    
    
    
#%%
# TESTING
"""
read_csv(dummy)
fig, ax = plt.subplots(1,1, dpi=200, tight_layout=True)
make_sqrt_I_plot(ax, data, freq=10, machine='SRS', grad_num=0, loc_num=17, frac=0.75)
print(get_n_gradpoints(data))

"""