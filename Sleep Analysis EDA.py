
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas_profiling
import math

import plotly.graph_objects as go
import plotly.express as px


# In[2]:


df = pd.read_csv("Sleep Analysis.csv")


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


print('Number of records where the Minutes asleep column is filled with zeros:', 
      len(df[df['Number of sleep periods'] == 0]))


# In[6]:


#pandas_profiling.ProfileReport(df)


# In[7]:


df['In bed start'] = pd.to_datetime(df['In bed start'],infer_datetime_format=True)
df['In bed Finish'] = pd.to_datetime(df['In bed Finish'],infer_datetime_format=True)


# In[8]:


df['In bed start Date'] = df['In bed start'].dt.date
df['In bed Finish Date'] = df['In bed Finish'].dt.date


# In[9]:


fig = go.Figure([go.Scatter(x=df['In bed start'], y=df['Minutes in bed'])])
fig.show()


# In[10]:


fig = go.Figure([go.Scatter(x=df[df['In bed start'] > '2017-10-11']['In bed start'], y=df[df['In bed start'] > '2017-10-11']['Minutes in bed'])])
fig.show()


# In[11]:


df[(df['In bed start'] > '2019-03-29') & (df['In bed start'] < '2019-04-02')]


# In[12]:


print("Nr of duplicated days: ", len(df[df['In bed Finish Date'].duplicated()]))


# In[13]:


aggregation_functions = {'In bed start': 'first', 'In bed Finish': 'last', 'Minutes in bed': 'sum', 
                        'Minutes asleep': 'sum', 'Time to fall sleep': 'sum'}
df_merged = df.groupby(df['In bed Finish Date']).aggregate(aggregation_functions)


# In[14]:


fig = go.Figure([go.Scatter(x=df_merged['In bed Finish'], y=df_merged['Minutes in bed'])])
fig.show()


# In[15]:


# Arrived in Kazakhstan
max_sleeping_time = df_merged['Minutes asleep'].max()
df_merged[df_merged['Minutes asleep'] == max_sleeping_time]


# In[16]:


# Went to the midnight premiere of the Fantastic Beasts 2
min_sleeping_time = df_merged[(df_merged['In bed start'] > '2017-10-11') & (df_merged['In bed Finish'].dt.hour < 13)]['Minutes in bed'].min()
df_merged[(df_merged['Minutes in bed'] == min_sleeping_time )]


# In[17]:


def deal_with_missing_sleep(row):
    if row['Minutes asleep'] == 0:
        return round(row['Minutes in bed'] / 60, 1)
    else:
        return round(row['Minutes asleep'] / 60, 1)
    
def round_to_closest_half(number):
    return round(number * 2) / 2


# In[18]:


df_merged['Hours asleep'] = df_merged.apply(deal_with_missing_sleep, axis=1)
df_merged['Asleep label'] = df_merged['Hours asleep'].apply(round_to_closest_half)


# In[19]:


df_merged['Asleep label EMA'] = df_merged.iloc[:,6].ewm(span=5,adjust=False).mean()


# In[20]:


fig = go.Figure([go.Scatter(x=df_merged[df_merged['In bed start'] > '2017-10-11']['In bed Finish'], 
                            y=df_merged[df_merged['In bed start'] > '2017-10-11']['Asleep label'], name = 'Hours asleep')])
fig.add_trace(go.Scatter(x=df_merged[df_merged['In bed start'] > '2017-10-11']['In bed Finish'], 
                            y=df_merged[df_merged['In bed start'] > '2017-10-11']['Asleep label EMA'], name= 'EMA'))
fig.show()


# In[21]:


df_merged['Year'] = df_merged['In bed Finish'].dt.year


# In[22]:


fig = px.histogram(df_merged, x="Asleep label", color = 'Year')
fig.show()


# In[23]:


fig = px.line(df_merged[df_merged['Year'].isin([2016, 2018, 2019])], y='Asleep label', color='Year')
fig.show()


# In[24]:


df_merged['Year'] = df_merged['In bed Finish'].dt.year
df_merged['Month'] = df_merged['In bed Finish'].dt.month
sleep_per_month_mean = pd.DataFrame(df_merged[df_merged['In bed start'] > '2017-10-11']
                                    .groupby(['Month', 'Year'])['Hours asleep'].mean()).reset_index()
sleep_per_month_mean['Month'] = sleep_per_month_mean['Month'].astype('category')
sleep_per_month_mean['Year'] = sleep_per_month_mean['Year'].astype('category')


# In[25]:


import calendar
month_names = dict(enumerate(calendar.month_abbr))
sleep_per_month_mean['Month'] = sleep_per_month_mean['Month'].map(month_names)


# In[26]:


fig = px.line(sleep_per_month_mean[(sleep_per_month_mean['Year'] == 2018) | (sleep_per_month_mean['Year'] == 2019)]
              , x="Month", y="Hours asleep", color='Year')
fig.show()


# In[27]:


sleep_per_month_mean = pd.DataFrame(df_merged[df_merged['In bed start'] > '2017-10-11'].groupby(['Month', 'Year'])['Asleep label'].mean()).reset_index()
sleep_per_month_mean['Month'] = sleep_per_month_mean['Month'].astype('category')
sleep_per_month_mean['Year'] = sleep_per_month_mean['Year'].astype('category')
sleep_per_month_mean = sleep_per_month_mean.set_index(['Month', 'Year'])
sleep_df = sleep_per_month_mean.unstack().unstack().to_frame().reset_index()
sleep_df = sleep_df.rename(columns={0: 'Sleep Mean'})
sleep_df = sleep_df.drop(columns=['level_0'])
month_names = dict(enumerate(calendar.month_abbr))
sleep_df['Month'] = sleep_df['Month'].map(month_names)


# In[28]:


fig = px.bar(sleep_df, x='Month', y="Sleep Mean", barmode='group', color = 'Year')
fig.show()

