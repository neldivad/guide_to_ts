import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

def run():
  st.header('Guide to forecast')

  stocks = ['^GSPC', 'AAPL', 'WMT', 'XOM', 'JNJ', 'KO', 'AMZN', 'BAC', 'AAL']
  start_date = "2015-01-01"
  end_date = "2022-05-21"

  if st.checkbox('Load data'):
    if 'stocks' not in st.session_state:
      data = pd.DataFrame()
      for stock in stocks:
        d = yf.download(stock, start=start_date, end=end_date)
        d = d[ ['Close'] ]
        d.columns = [stock] # col rename
        data = pd.concat([data, d], axis=1)
      st.session_state['stocks'] = data
    df = st.session_state['stocks']
  
  if 'stocks' not in st.session_state:
    st.info('Data not yet loaded')
   else:
    st.success('Data loaded')
  
  with st.expander('Show data'):
    try:
      st.write(df)
    except:
      st.write(pd.DataFrame())
      st.info('Data not yet loaded')

  ###########################
  def make_line_chart(df, title=''):
    import plotly.express as px
    fig = px.line(df)
    fig.update_layout(title=title) 
    return fig

  with st.expander('Show line chart'):
    fig = make_line_chart(df)
    st.plotly_chart(fig, use_container_width=True)

  ##################
  def make_scatter_chart(df, xdata, ydata):
    import plotly.express as px 
    fig = px.scatter(df, x=xdata, y=ydata, trendline='ols', hover_name=df.index)
    return fig 

  def transform_df(df, transform='no'):
    choices = ['no', 'diff', 'log', 'logdiff']
    if transform not in choices:
      raise ValueError("'transform' only accepts ['no', 'log', 'logdiff']")
    if transform == 'no':
      return df
    data = df.copy()
    if transform == 'log':
      data = np.log(data)
    elif transform == 'logdiff':
      data = np.log(data).diff()
    data.dropna(inplace=True)
    return data

  with st.expander('Show scatter chart'):
    with st.form('Scatter'):
      col1, col2 = st.columns([1,1])
      with col1:
        xd = st.selectbox('Select X data', stocks, index=0)
        transform = st.selectbox('Select data transformation', ['no', 'log', 'logdiff'], key='t0')
      with col2:
        yd = st.selectbox('Select Y data', stocks, index=1)

      if st.form_submit_button('Get linear relationship'):
        sfig = make_scatter_chart( transform_df(df, transform=transform), xdata=xd, ydata=yd )
        st.plotly_chart(sfig, use_container_width=True)

        if 'diff' not in transform:
          st.error('This scatter plot is meaningless. You got ruined by autocorrelation')
        else:
          st.info('Log diff is the only way to draw any meaningful linear relationship between prices.')
        

  with st.expander('Show MA chart'): 
    with st.form('MA'):
      col1, col2 = st.columns([1,1])
      with col1:
        window = st.number_input('Select Moving Average Interval', 1, 252, 21)
        axis_scale = st.selectbox('Y-axis scale', ['no scale', 'standard', 'minmax'])
      with col2:
        transform1 = st.selectbox('Select data transformation', ['no', 'log', 'logdiff'], key='t1')

      if st.form_submit_button("Get moving averages"):
        rolling_mean = transform_df( df, transform1 ).rolling(window = window).mean()
        rolling_std  = transform_df( df, transform1 ).rolling(window = window).std() *100

        tdf = transform_df(df, transform1) 
        tdf_minus = tdf - rolling_mean
        tdf_minus.dropna(inplace=True)

        def scale_df(df, scaler='standard'):
          from sklearn import preprocessing

          scaler_choice = ['standard', 'minmax']
          if scaler not in scaler_choice:
            raise ValueError("Invalid scaler type. Expected one of: %s" % scaler_choice)
          if scaler == 'standard':
            scaler = preprocessing.StandardScaler()
          if scaler == 'minmax':
            scaler = preprocessing.MinMaxScaler()
          
          cols= df.columns
          idx = df.index
            # skl scaler will strip colname and index
          scaled_df = scaler.fit_transform(df)
          scaled_df = pd.DataFrame(scaled_df, columns=cols)
          scaled_df = scaled_df.set_index(idx)
          return scaled_df

        rt = df['^GSPC'].pct_change().dropna()
        import arch 
        am = arch.univariate.arch_model(
          df['^GSPC'].pct_change().dropna(), 
          x=None, mean='HARX', 
          lags=0, vol='Garch', 
          p=1, o=0, q=1, 
          dist='skewt', hold_back=None, rescale=True
          ) 
        volatility_model = am.fit()
        const, omega, alpha, beta, eta, lamb = volatility_model.params # Retrieve Model Parameters
        garch_vol = volatility_model.conditional_volatility.round(2) * np.sqrt(252) # Retrieve conditional volatility
        VL = omega / (1 - alpha - beta ) # long-term variance under GARCH
        sigma_L = np.sqrt(VL) * np.sqrt(252) # long-term volatility under GARCH (convert from variance)
        sample_sigma = rt.std() *np.sqrt(252) * 100 # sample volatility estimate
 
        VIX = yf.download('^VIX', start= start_date, end= end_date)
        vol_df = pd.concat([rolling_std['^GSPC'], VIX['Close'], garch_vol], axis=1)
        vol_df.columns=['Actual Vol', 'Implied Vol', 'Conditional Vol']
        vol_df = vol_df.dropna()

        if 'no' in axis_scale: 
          pass
        else:
          vol_df = scale_df(vol_df, scaler=axis_scale)
        vol_minus = pd.DataFrame()
        vol_minus['Implied minus Actual'] = vol_df['Implied Vol'] - vol_df['Actual Vol']
        vol_minus['Implied minus Conditional'] = vol_df['Implied Vol'] - vol_df['Conditional Vol']
        vol_minus['Actual minus Conditional'] = vol_df['Actual Vol'] - vol_df['Conditional Vol']
        vol_minus = vol_minus.dropna() 

        st.plotly_chart( make_line_chart(rolling_mean, 'MA mean'), use_container_width=True)
        st.plotly_chart( make_line_chart(rolling_std, 'MA std + VIX'), use_container_width=True)
        st.plotly_chart( make_line_chart(tdf_minus, 'Original - MA mean' ), use_container_width=True)
        st.info('If rolling mean and rolling standard deviation is not stable with time, time series is not stationary.')

        st.plotly_chart( make_line_chart(vol_df, f'Implied Vol vs Actual Vol (Y-axis={axis_scale})'), use_container_width=True)
        st.plotly_chart( make_line_chart(vol_minus, f'Volatility gap (Y-axis={axis_scale})'), use_container_width=True)
        st.plotly_chart( make_KDE_plot(vol_minus, remove_outliers=False), use_container_width=True )
        st.info('If implied minus actual is above 0, it means traders overpaid for insurance. If below 0, traders are under-insured.')
        st.info('If actual minus conditional is above 0, it means models are too optimistic. If below 0, risk models are too pessimistic.')

  with st.expander('Autocorrelation Function Plot'):
    with st.form('ACF'):
      col1, col2 = st.columns([1,1])
      with col1:
        tickersel2 = st.selectbox('Select ticker data', stocks, index=0)
      with col2:
        transform2 = st.selectbox('Select data transformation', ['no', 'log', 'logdiff'], key='t2')
      
      if st.form_submit_button('Get ACF plots'):
        acfp = make_acf_plot( transform_df( df[ tickersel2 ], transform=transform2 ), plot_pacf=False)
        pacfp= make_acf_plot( transform_df( df[ tickersel2 ], transform=transform2 ), plot_pacf=True)
        st.plotly_chart(acfp, use_container_width=True)
        st.plotly_chart(pacfp, use_container_width=True)
        st.info('If ACF plot have downward trending spikes, series is not stationary. After log-diff, ACF spike signals MA(n), PACF spikes signals AR(n)')

  with st.expander('Run forecast model'):
    st.info('Fuck forecasting in Streamlit. Fuck most Python forecasting libraries in general. Difficult to customize and bring up to the front-end')
    st.error('Dynamic forecasting wastes so much computational resources. Lack of customizibility means delivering low resolution info')
    st.error('Why are you not doing it on Jupyter Notebook?')
      
def make_KDE_plot(resid, remove_outliers=False):
  import plotly.figure_factory as ff
  import numpy as np
  from scipy import stats
  
  if remove_outliers == True:
    resid = resid[(np.abs(stats.zscore(resid)) < 3).all(axis=1)]
    title = f'Distribution of Residuals:<br>(Z-score cutoff=3)'
  elif remove_outliers != False:
    resid = resid[(np.abs(stats.zscore(resid)) < remove_outliers).all(axis=1)]
    title = f'Distribution of Residuals:<br>(Z-score cutoff={remove_outliers})'
  else:
    title = f'Distribution of Residuals: (With Outliers)'

  rug_text=[]
  for col in resid.columns:
    rug_text.append(resid.index)

  fig = ff.create_distplot(
    [resid[c] for c in resid.columns],
    group_labels=resid.columns, 
    rug_text= rug_text, 
    show_hist=False,
    show_rug=True
  )
  return fig
      


# @st.cache()
def make_acf_plot(series, plot_pacf=False):
  """
  Produces an ACF/PACF plot for a series. Lags = 42. Areas are shaded at 0.05 alpha. 
  
  series: A series (column) of a dataframe.
  plot_pacf: Default = False. If true, returns a PACF plot instead of ACF plot. 
  """
  from statsmodels.tsa.stattools import acf, pacf
  import plotly.graph_objects as go
  
  corr_array = pacf(series.dropna(), alpha=0.05) if plot_pacf else acf(series.dropna(), alpha=0.05)
  
  lower_y = corr_array[1][:,0] - corr_array[0]
  upper_y = corr_array[1][:,1] - corr_array[0]

  fig = go.Figure()
  [fig.add_scatter(
    x=(x,x),
    y=(0,corr_array[0][x]), 
    mode='lines',
    line_color='#3f3f3f'
  )\
  for x in range(len(corr_array[0])) ]
  
  fig.add_scatter(
    x=np.arange(len(corr_array[0])),
    y=corr_array[0], 
    mode='markers', 
    marker_color='#1f77b4',
    marker_size=8
  )
  
  fig.add_scatter(
    x=np.arange(len(corr_array[0])), 
    y=upper_y, 
    mode='lines', 
    line_color='rgba(255,255,255,0)'
  )
  
  fig.add_scatter(
    x=np.arange(len(corr_array[0])), 
    y=lower_y, mode='lines',
    fillcolor='rgba(32, 146, 230,0.6)',
    fill='tonexty', 
    line_color='rgba(255,255,255,0)'
  )
  
  fig.update_traces(showlegend=False)
  fig.update_xaxes(
    range=[-1,42],
    # rangeslider_visible=True
  )
  fig.update_yaxes(zerolinecolor='#000000', autorange= True, fixedrange = False)
  #fig.update_xaxes(showgrid=True, gridwidth=0.2, gridcolor='darkslategrey')
  #fig.update_yaxes(showgrid=True, gridwidth=0.2, gridcolor='darkslategrey')
  
  yaxis_title = 'Partial Autocorrelation' if plot_pacf else 'Autocorrelation'
  
  title=f'Partial Autocorrelation (PACF) for "{series.name}"' if plot_pacf \
  else f'Autocorrelation (ACF) for "{series.name}"'
  
  fig.update_layout(
    xaxis_title= 'Lag',
    yaxis_title= yaxis_title,
    title=title,
    height= 300
  )
  return fig

#--------------------
# Run app
#--------------------
run()
