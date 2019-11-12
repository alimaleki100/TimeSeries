# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import fbprophet as Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('mv.xlsx')

df.head(5)

#e carrying capacity---> maximum size of each day
df['cap']=50

df['Date'] = pd.DatetimeIndex(df['Date'])


df = df.rename(columns={'Date': 'ds',
                        'Value': 'y'})

df.head(5)

df.plot(x='ds',y='y',kind='hist')
#df = df.set_index('ds')


#df['ds'] = pd.to_datetime(df['ds'], format = '%Y%M%D')

#Set Date as index
#df = df.set_index('ds')



plt.figure(figsize=(20,10))
plt.plot(df.ds, df.y)
plt.title('AirPassengers')

#plt.savefig('test2png.png', dpi=100)


###Prophet Model

# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet.Prophet(interval_width=0.95,weekly_seasonality=True,daily_seasonality=True,growth='logistic')


#fit df to model
my_model.fit(df)

"""
In order to obtain forecasts of our time series, 
we must provide Prophet with a new DataFrame containing a ds column that holds 
the dates for which we want predictions. Conveniently, we do not have to concern 
ourselves with manually creating this DataFrame,
 as Prophet provides the make_future_dataframe helper function:"""
 
 
future_dates = my_model.make_future_dataframe(periods=90, freq='D')
future_dates['cap'] = 50
future_dates.tail()

forecast = my_model.predict(future_dates)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())



my_model.plot(forecast,uncertainty=True)

my_model.plot_components(forecast)


"""
writer=pd.ExcelWriter("C:/Users/session1/Desktop/mvcrsResult.xlsx",engine='xlsxwriter')
forecast.to_excel(writer, sheet_name='new')
writer.close()"""


#cross validation

"""
    horizon the forecast horizon
    initial the size of the initial training period
    period the spacing between cutoff dates
    """
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(my_model,  horizon = '90 days')
df_cv.head()


"""
Obtaining the Performance Metrics

We use the performance_metrics utility to compute the Mean Squared Error(MSE),
 Root Mean Squared Error(RMSE), Mean Absolute Error(MAE), Mean Absolute Percentage Error(MAPE)
 and the coverage of the yhat_lower and yhat_upper estimates."""

from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
print(df_p.head())


"""
The performance Metrics can be visualized using the plot_cross_validation_metric utility.
 Let’s visualize the RMSE below."""
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='rmse')



####Simple anomaly Detection


#Create anomaly index
anomaly_df=pd.merge(forecast,df,on='ds')
anomaly_df['anomaly']=0


#set anomaly condition
anomaly_df.loc[anomaly_df['y'] > anomaly_df['yhat_upper'], 'anomaly'] = 1
anomaly_df.loc[anomaly_df['y'] < anomaly_df['yhat_lower'], 'anomaly'] = 1

#Visualize Anomaly
ax = plt.gca()
ax.plot(anomaly_df['ds'].values, anomaly_df['y'].values, 'b-')
ax.scatter(anomaly_df[anomaly_df['anomaly'] == 1]['ds'].values,
            anomaly_df[anomaly_df['anomaly'] == 1]['y'].values, color='red')
ax.fill_between(anomaly_df['ds'].values, anomaly_df['yhat_lower'].values, anomaly_df['yhat_upper'].values,
                    alpha=0.3, facecolor='r')
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()

#Generate Anomaly Positive df
positive_anomaly_df=anomaly_df.loc[anomaly_df['anomaly']==1]
positive_anomaly_df=positive_anomaly_df[['ds', 'y','yhat', 'yhat_lower', 'yhat_upper']]
print(positive_anomaly_df)



