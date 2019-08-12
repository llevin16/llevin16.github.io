#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import scipy as sp
import math
from matplotlib import pyplot
import datetime as dt
from scipy import stats
from mapsplotlib import mapsplot as mplt

mplt.register_api_key('AIzaSyBtL1MGTwvBlZnDNJCmT4nKQ5nO7feM78I')

get_ipython().magic(u'matplotlib inline')


# In[3]:


df_input=pd.read_csv('C:/Users/Lenny Levin/Downloads/CMM297FinalList.csv')
print df_input.shape
df_input.head()


# In[93]:


df_input.dtypes


# In[2]:


df_input.iloc[19900:20000].Distance.hist()


# In[5]:


df_ces=pd.read_excel('C:/Users/Lenny Levin/Downloads/ces3results_ll.xlsx')
print df_ces.shape
df_ces.head()


# In[81]:


df_ces.dtypes


# In[7]:


def haversine(coord1, coord2):
    R = 3958.756  # Earth radius in miles
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 +         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))


# In[60]:


sample_1=33.791455,-118.23866
sample_2=33.792327,-118.23728
print sample_1, sample_2
haversine(sample_1,sample_2)


# In[480]:


df_input.Distance[df_input.Distance>.15].hist()


# In[692]:


df_clean['Date_Time']=pd.to_datetime(df_clean['Date_Time'])
df_clean=df_clean[df_clean['DummyPtMode']!='Error']
df_clean.loc[df_clean.DummyPtMode=='Diesel','DummyPtMode']='Hybrid'
df_clean['Shift']=df_clean['Date_Time'].dt.date
print df_clean.describe()
print df_clean.DummyPtMode.value_counts()
print df_clean.DummySpeed.value_counts()


# In[612]:


df_clean.head()


# In[98]:


#df_clean=df_input.sample(frac=.25,replace=False,random_state=16)
#df_clean=df_input.iloc[0:10000]
df_clean=df_clean.reset_index(drop=True)
#df_clean=df_input
print df_clean.shape
"""
for la,lo,i in zip(df_clean['gps_Latitude'],df_clean['gps_Longitude'],df_clean.index):
    coord_1=la,lo
    try:
        coord_2=df_clean.iloc[i+1,1],df_clean.iloc[i+1,2]
        dist=haversine(coord_1,coord_2)
        df_clean.iloc[i,7]=dist
    except:
        df_clean.iloc[i,7]=float('nan')
"""
distance=[]
for la,lo,i in zip(df_clean['gps_Latitude'],df_clean['gps_Longitude'],df_clean.index):
    coord_1=la,lo 
    try:
        coord_2=df_clean.iloc[i+1,1],df_clean.iloc[i+1,2]
        distance.append(haversine(coord_1,coord_2))
    except:
        distance.append(0)
df_clean['Distance']=distance
df_clean.head()


# In[693]:


df_clean.Distance[df_clean.Distance>.15].hist()


# In[ ]:


df_clean.loc[df_clean.Distance>.1,'Distance']=0


# In[614]:


df_clean.iloc[19680:19700]


# In[102]:


df_clean.sort_values(by='Distance',ascending=False).head()


# In[103]:


df_clean.iloc[97300:97330]


# In[694]:


df_clean['VehicleSpeed(mph)'].hist()
df_clean.shape


# In[695]:


df_clean=df_clean.loc[df_clean['VehicleSpeed(mph)']<100]
df_clean.reset_index(drop=True,inplace=True)
df_clean['VehicleSpeed(mph)'].hist()
df_clean.shape


# In[452]:


df_clean.tail()


# In[128]:


df_ces_cl=df_ces[['Census Tract','ZIP','Longitude','Latitude','CES 3.0 Score']]
df_ces_cl.is_copy=None
df_ces_cl['Closest City']=df_ces.iloc[:,4]
df_ces_cl['DAC_Binary']=[bool(x=='Yes') for x in df_ces['SB 535 Disadvantaged Community']]
df_ces_cl.head()


# In[129]:


df_ces_cl.loc[(df_ces_cl['Census Tract']==6037980031)|(df_ces_cl['Census Tract']==6037980015),'CES 3.0 Score']=0
df_ces_cl.loc[(df_ces_cl['Census Tract']==6037980002)|(df_ces_cl['Census Tract']==6037980014),'CES 3.0 Score']=0
df_ces_cl.loc[(df_ces_cl['Census Tract']==6037575500)|(df_ces_cl['Census Tract']==6037980033),'CES 3.0 Score']=0
df_ces_cl.dropna(axis=0,inplace=True)
df_ces_cl['CES 3.0 Score'].isna().value_counts()


# In[257]:


df_ces_cl=df_ces_cl.reset_index(drop=True)
df_ces_cl[(df_ces_cl['Census Tract']==6037980031)|(df_ces_cl['Census Tract']==6037980033)].head()


# In[133]:


df_ces_cl.tail()


# In[134]:


#sample=df_clean.iloc[0:10000]
#sample.is_copy=None
cities=[]
zips=[]
scores=[]
dacs=[]
min_distances=[]
for la,lo in zip(df_clean['gps_Latitude'],df_clean['gps_Longitude']):
    distance=[]
    vlv_coor=la,lo
    for la2,lo2,i in zip(df_ces_cl['Latitude'],df_ces_cl['Longitude'],df_ces_cl.index):
        ces_coor=la2,lo2
        d=haversine(vlv_coor,ces_coor)
        temp=d,i
        distance.append(temp)
    minimum=min(distance)
    cities.append(df_ces_cl.iloc[minimum[1],5])
    zips.append(df_ces_cl.iloc[minimum[1],1])
    scores.append(df_ces_cl.iloc[minimum[1],4])
    dacs.append(df_ces_cl.iloc[minimum[1],6])
    min_distances.append(minimum[0])
df_clean['City']=cities
df_clean['Zip Code']=zips
df_clean['Score']=scores
df_clean['DAC']=dacs
df_clean['Distance to DAC']=min_distances
df_clean.head()


# In[696]:


df_clean.shape


# In[143]:


df_clean.to_csv('CMM297_cleaned.csv',index=False)


# In[690]:


df_clean=test


# In[697]:


print 'Overall Distances'
print 'Total Distance driven: %d miles' %df_clean.Distance.sum()
print 'Total Distance driven in Electric: %d miles' %df_clean.Distance[df_clean.DummyPtMode=='Electric'].sum()
print 'Total Distance driven in Hybrid: %d miles' %df_clean.Distance[df_clean.DummyPtMode=='Hybrid'].sum()
util=(np.divide(df_clean.Distance[df_clean.DummyPtMode=='Electric'].sum(),df_clean.Distance.sum()))*100
print 'Percent of Total Distance driven in Electric: %.2f%%' %util

bins = np.linspace(0, df_clean.Distance.max(), 50)
pyplot.hist(df_clean.Distance, bins, alpha=.5, label='Total')
pyplot.hist(df_clean.Distance[df_clean.DummyPtMode=='Electric'], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean.Distance[df_clean.DummyPtMode=='Hybrid'], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[619]:


print 'DAC Distances'
print 'Total Distance driven: %d miles' %df_clean.Distance[df_clean.DAC==True].sum()
print 'Total Distance driven in Electric: %d miles' %df_clean.Distance[(df_clean.DAC==True)&(df_clean.DummyPtMode=='Electric')].sum()
print 'Total Distance driven in Hybrid: %d miles' %df_clean.Distance[(df_clean.DAC==True)&(df_clean.DummyPtMode=='Hybrid')].sum()
util=(np.divide(df_clean.Distance[(df_clean.DAC==True)&(df_clean.DummyPtMode=='Electric')].sum(),df_clean.Distance[df_clean.DAC==True].sum()))*100
print 'Percent of Total Distance driven in Electric: %.2f%%' %util

bins = np.linspace(0, df_clean.Distance[df_clean.DAC==True].max(), 50)
pyplot.hist(df_clean.Distance[df_clean.DAC==True], bins, alpha=.5, label='Total')
pyplot.hist(df_clean.Distance[(df_clean.DAC==True)&(df_clean.DummyPtMode=='Electric')], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean.Distance[(df_clean.DAC==True)&(df_clean.DummyPtMode=='Hybrid')], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[620]:


print 'DAC Distances w/o Highway Driving'
t_condition=(df_clean.DAC==True)&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean.DAC==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Hybrid')
print 'Total Distance driven: %d miles' %df_clean.Distance[t_condition].sum()
print 'Total Distance driven in Electric: %d miles' %df_clean.Distance[e_condition].sum()
print 'Total Distance driven in Hybrid: %d miles' %df_clean.Distance[h_condition].sum()
util=(np.divide(df_clean.Distance[e_condition].sum(),df_clean.Distance[t_condition].sum()))*100
print 'Percent of Total Distance driven in Electric: %.2f%%' %util

bins = np.linspace(0, df_clean.Distance[t_condition].max(), 50)
pyplot.hist(df_clean.Distance[t_condition], bins, alpha=.5, label='Total')
pyplot.hist(df_clean.Distance[e_condition], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean.Distance[h_condition], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[621]:


print df_clean.DummySpeed[h_condition].value_counts()
print df_clean.DummySpeed[e_condition].value_counts()


# In[622]:


print 'Overall Speeds'
print 'Average Speed: %.2f mph' %df_clean['VehicleSpeed(mph)'].mean()
print 'Average Speed in Electric: %.2f mph' %df_clean['VehicleSpeed(mph)'][df_clean.DummyPtMode=='Electric'].mean()
print 'Average Speed in Hybrid: %.2f mph' %df_clean['VehicleSpeed(mph)'][df_clean.DummyPtMode=='Hybrid'].mean()

bins = np.linspace(0, df_clean['VehicleSpeed(mph)'].max(), 50)
pyplot.hist(df_clean['VehicleSpeed(mph)'], bins, alpha=.5, label='Overall')
pyplot.hist(df_clean['VehicleSpeed(mph)'][df_clean.DummyPtMode=='Electric'], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean['VehicleSpeed(mph)'][df_clean.DummyPtMode=='Hybrid'], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[623]:


print 'DAC Speeds'
t_condition=df_clean.DAC==True
e_condition=(df_clean.DAC==True)&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC==True)&(df_clean.DummyPtMode=='Hybrid')
print 'Average Speed: %.2f mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Average Speed in Electric: %.2f mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Average Speed in Hybrid: %.2f mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

bins = np.linspace(0, df_clean['VehicleSpeed(mph)'][t_condition].max(), 50)
pyplot.hist(df_clean['VehicleSpeed(mph)'][t_condition], bins, alpha=.5, label='Overall')
pyplot.hist(df_clean['VehicleSpeed(mph)'][e_condition], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean['VehicleSpeed(mph)'][h_condition], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[624]:


print 'DAC Speeds w/o Highway Driving'
t_condition=(df_clean.DAC==True)&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean.DAC==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Hybrid')
print 'Average Speed: %.2f mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Average Speed in Electric: %.2f mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Average Speed in Hybrid: %.2f mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

bins = np.linspace(0, df_clean['VehicleSpeed(mph)'][t_condition].max(), 50)
pyplot.hist(df_clean['VehicleSpeed(mph)'][t_condition], bins, alpha=.5, label='Overall')
pyplot.hist(df_clean['VehicleSpeed(mph)'][e_condition], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean['VehicleSpeed(mph)'][h_condition], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[625]:


print df_clean['VehicleSpeed(mph)'][df_clean['VehicleSpeed(mph)']==0].count()
print df_clean['VehicleSpeed(mph)'].count()
perc=np.divide(float(df_clean['VehicleSpeed(mph)'][df_clean['VehicleSpeed(mph)']==0].count()),
                df_clean['VehicleSpeed(mph)'].count())*100
print '%.2f%%' %perc


# In[699]:


df_clean['DAC_adj']=[bool(x>75) for x in df_clean['Score']]
df_clean['DAC_adj'].value_counts()


# In[627]:


print 'Adjusted DAC Distances'
print 'Total Distance driven: %d miles' %df_clean.Distance[df_clean.DAC_adj==True].sum()
print 'Total Distance driven in Electric: %d miles' %df_clean.Distance[(df_clean.DAC_adj==True)&(df_clean.DummyPtMode=='Electric')].sum()
print 'Total Distance driven in Hybrid: %d miles' %df_clean.Distance[(df_clean.DAC_adj==True)&(df_clean.DummyPtMode=='Hybrid')].sum()
util=(np.divide(df_clean.Distance[(df_clean.DAC_adj==True)&(df_clean.DummyPtMode=='Electric')].sum(),
                df_clean.Distance[df_clean.DAC_adj==True].sum()))*100
print 'Percent of Total Distance driven in Electric: %.2f%%' %util

bins = np.linspace(0, df_clean.Distance[df_clean.DAC_adj==True].max(), 50)
pyplot.hist(df_clean.Distance[df_clean.DAC_adj==True], bins, alpha=.5, label='Total')
pyplot.hist(df_clean.Distance[(df_clean.DAC_adj==True)&(df_clean.DummyPtMode=='Electric')], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean.Distance[(df_clean.DAC_adj==True)&(df_clean.DummyPtMode=='Hybrid')], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[628]:


print 'Adjusted DAC Distances w/o Highway Driving'
t_condition=(df_clean.DAC_adj==True)&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean.DAC_adj==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC_adj==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Hybrid')
print 'Total Distance driven: %d miles' %df_clean.Distance[t_condition].sum()
print 'Total Distance driven in Electric: %d miles' %df_clean.Distance[e_condition].sum()
print 'Total Distance driven in Hybrid: %d miles' %df_clean.Distance[h_condition].sum()
util=(np.divide(df_clean.Distance[e_condition].sum(),df_clean.Distance[t_condition].sum()))*100
print 'Percent of Total Distance driven in Electric: %.2f%%' %util

bins = np.linspace(0, df_clean.Distance[t_condition].max(), 50)
pyplot.hist(df_clean.Distance[t_condition], bins, alpha=.5, label='Total')
pyplot.hist(df_clean.Distance[e_condition], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean.Distance[h_condition], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[629]:


print 'Adjusted DAC Speeds'
t_condition=df_clean.DAC_adj==True
e_condition=(df_clean.DAC_adj==True)&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC_adj==True)&(df_clean.DummyPtMode=='Hybrid')
print 'Average Speed: %.2f mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Average Speed in Electric: %.2f mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Average Speed in Hybrid: %.2f mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

bins = np.linspace(0, df_clean['VehicleSpeed(mph)'][t_condition].max(), 50)
pyplot.hist(df_clean['VehicleSpeed(mph)'][t_condition], bins, alpha=.5, label='Overall')
pyplot.hist(df_clean['VehicleSpeed(mph)'][e_condition], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean['VehicleSpeed(mph)'][h_condition], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[630]:


print 'Adjusted DAC Speeds w/o Highway Driving'
t_condition=(df_clean.DAC_adj==True)&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean.DAC_adj==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC_adj==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Hybrid')
print 'Average Speed: %.2f mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Average Speed in Electric: %.2f mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Average Speed in Hybrid: %.2f mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

bins = np.linspace(0, df_clean['VehicleSpeed(mph)'][t_condition].max(), 50)
pyplot.hist(df_clean['VehicleSpeed(mph)'][t_condition], bins, alpha=.5, label='Overall')
pyplot.hist(df_clean['VehicleSpeed(mph)'][e_condition], bins, alpha=.5, label='Electric')
pyplot.hist(df_clean['VehicleSpeed(mph)'][h_condition], bins, alpha=0.5, label='Hybrid')
pyplot.legend(loc='upper right')
pyplot.show()


# In[774]:


df_clean['Electric_Bool']=[bool(x==2) for x in df_clean['currentPtMode_PT_X_T']]
df_clean['Speed_Bool']=[bool(x>0) for x in df_clean['VehicleSpeed(mph)']]
df_clean['Dist_Bool']=[bool(x>0) for x in df_clean['Distance']]


# In[775]:


print df_clean.DummySpeed.value_counts()
print df_clean.Dist_Bool.value_counts()


# In[317]:


gps_condition=((df_clean['Shift']==dt.date(2017,7,31))|(df_clean['Shift']==dt.date(2017,8,15))|
    (df_clean['Shift']==dt.date(2017,9,14))|(df_clean['Shift']==dt.date(2017,11,24))|(df_clean['Shift']==dt.date(2017,12,15)))
df_gps=df_clean[['Date_Time','gps_Latitude','gps_Longitude','VehicleSpeed(mph)']][gps_condition]
print df_gps.shape
df_gps.head()


# In[318]:


df_gps.to_csv('CMM297_GPS.csv',index=False)


# In[703]:


#sample=df_clean.sample(n=1000,replace=False,random_state=100)
#sample=sample.reset_index(drop=True)
df_clean=df_clean.sort_values(by='Date_Time')
df_clean.reset_index(inplace=True,drop=True)
interval=[]
for time,i in zip(df_clean['Date_Time'],df_clean.index):
    try:
        int_temp=df_clean.iloc[i+1,0]-time
        interval.append(int_temp)
    except:
        interval.append(0)
df_clean.loc[:,'Interval']=interval
df_clean['Interval']=pd.to_timedelta(df_clean.Interval)
df_clean.head()


# In[704]:


df_clean[df_clean['Shift']==dt.date(2017,7,28)].Interval.dt.total_seconds().sum()


# In[713]:


df_clean[df_clean.Interval>pd.Timedelta('2 hour')].count()


# In[714]:


df_clean.dtypes


# In[712]:


df_clean.loc[df_clean.Interval>pd.Timedelta('3 hour'),'Interval']=pd.Timedelta('0 seconds')


# In[731]:


print 'Overall Times'
print 'Total Time driven: %d hours' %np.divide(df_clean.Interval.dt.total_seconds().sum(),3600)
print 'Total Time driven in Electric: %d hours' %np.divide(df_clean.Interval[df_clean.DummyPtMode=='Electric'].
                                                           dt.total_seconds().sum(),3600)
print 'Total Time driven in Hybrid: %d hours' %np.divide(df_clean.Interval[df_clean.DummyPtMode=='Hybrid'].
                                                           dt.total_seconds().sum(),3600)
util=(np.divide(df_clean.Interval[df_clean.DummyPtMode=='Electric'].dt.total_seconds().sum(),
                df_clean.Interval.dt.total_seconds().sum()))*100
print 'Percent of Total Time driven in Electric: %.2f%%' %util


# In[734]:


print 'DAC Times'
t_condition=(df_clean.DAC==True)
e_condition=(df_clean.DAC==True)&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC==True)&(df_clean.DummyPtMode=='Hybrid')
print 'Total Time driven: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Total Time driven in Electric: %d hours' %np.divide(df_clean.Interval[e_condition].
                                                           dt.total_seconds().sum(),3600)
print 'Total Time driven in Hybrid: %d hours' %np.divide(df_clean.Interval[h_condition].
                                                           dt.total_seconds().sum(),3600)
util=(np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
                df_clean.Interval[t_condition].dt.total_seconds().sum()))*100
print 'Percent of Total Time driven in Electric: %.2f%%' %util


# In[735]:


print 'DAC Times w/o Highway Driving'
t_condition=(df_clean.DAC==True)&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean.DAC==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Hybrid')
print 'Total Time driven: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Total Time driven in Electric: %d hours' %np.divide(df_clean.Interval[e_condition].
                                                           dt.total_seconds().sum(),3600)
print 'Total Time driven in Hybrid: %d hours' %np.divide(df_clean.Interval[h_condition].
                                                           dt.total_seconds().sum(),3600)
util=(np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
                df_clean.Interval[t_condition].dt.total_seconds().sum()))*100
print 'Percent of Total Time driven in Electric: %.2f%%' %util


# In[736]:


print 'Adjusted DAC Times'
t_condition=(df_clean.DAC_adj==True)
e_condition=(df_clean.DAC_adj==True)&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC_adj==True)&(df_clean.DummyPtMode=='Hybrid')
print 'Total Time driven: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Total Time driven in Electric: %d hours' %np.divide(df_clean.Interval[e_condition].
                                                           dt.total_seconds().sum(),3600)
print 'Total Time driven in Hybrid: %d hours' %np.divide(df_clean.Interval[h_condition].
                                                           dt.total_seconds().sum(),3600)
util=(np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
                df_clean.Interval[t_condition].dt.total_seconds().sum()))*100
print 'Percent of Total Time driven in Electric: %.2f%%' %util


# In[737]:


print 'Adjusted DAC Times w/o Highway Driving'
t_condition=(df_clean.DAC_adj==True)&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean.DAC_adj==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean.DAC_adj==True)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean.DummyPtMode=='Hybrid')
print 'Total Time driven: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Total Time driven in Electric: %d hours' %np.divide(df_clean.Interval[e_condition].
                                                           dt.total_seconds().sum(),3600)
print 'Total Time driven in Hybrid: %d hours' %np.divide(df_clean.Interval[h_condition].
                                                           dt.total_seconds().sum(),3600)
util=(np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
                df_clean.Interval[t_condition].dt.total_seconds().sum()))*100
print 'Percent of Total Time driven in Electric: %.2f%%' %util


# In[810]:


print 'DISTANCES'

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')
print '\nPre Software Change: %d miles' %df_clean['Distance'][t_condition].sum()
print 'Electric Pre Software Change: %d miles' %df_clean['Distance'][e_condition].sum()
print 'Hybrid Pre Software Change: %d miles' %df_clean['Distance'][h_condition].sum()
util=np.divide(df_clean['Distance'][e_condition].sum(),df_clean['Distance'][t_condition].sum())*100
print 'Percentage of Distance in Electric Pre Software Change: %.2f%%' %util

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')
print '\nPost Software Change: %d miles' %df_clean['Distance'][t_condition].sum()
print 'Electric Post Software Change: %d miles' %df_clean['Distance'][e_condition].sum()
print 'Hybrid Post Software Change: %d miles' %df_clean['Distance'][h_condition].sum()
util=np.divide(df_clean['Distance'][e_condition].sum(),df_clean['Distance'][t_condition].sum())*100
print 'Percentage of Distance in Electric Post Software Change: %.2f%%' %util

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean.DummySpeed!='Moving on Highway')
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean.DummySpeed!='Moving on Highway')
print '\nPre Software Change w/o Highway: %d miles' %df_clean['Distance'][t_condition].sum()
print 'Electric Pre Software Change w/o Highway: %d miles' %df_clean['Distance'][e_condition].sum()
print 'Hybrid Pre Software Change w/o Highway: %d miles' %df_clean['Distance'][h_condition].sum()
util=np.divide(df_clean['Distance'][e_condition].sum(),df_clean['Distance'][t_condition].sum())*100
print 'Percentage of Distance in Electric Pre Software Change w/o Highway: %.2f%%' %util

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean.DummySpeed!='Moving on Highway')
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean.DummySpeed!='Moving on Highway')
print '\nPost Software Change w/o Highway: %d miles' %df_clean['Distance'][t_condition].sum()
print 'Electric Post Software Change w/o Highway: %d miles' %df_clean['Distance'][e_condition].sum()
print 'Hybrid Post Software Change w/o Highway: %d miles' %df_clean['Distance'][h_condition].sum()
util=np.divide(df_clean['Distance'][e_condition].sum(),df_clean['Distance'][t_condition].sum())*100
print 'Percentage of Distance in Electric Post Software Change w/o Highway: %.2f%%' %util

pre=df_clean['Electric_Bool'][(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean['Distance']>0)].sample(n=250000,replace=False,random_state=1)
post=df_clean['Electric_Bool'][(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean['Distance']>0)].sample(n=250000,replace=False,random_state=1)
print '\n%d %d' %(pre.shape[0],post.shape[0])
stats.ttest_ind(pre,post,equal_var=False)


# In[809]:


print 'TIME'

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')
print '\nPre Software Change: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Electric Pre Software Change: %d hours' %np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),3600)
print 'Hybrid Pre Software Change: %d hours' %np.divide(df_clean.Interval[h_condition].dt.total_seconds().sum(),3600)
util=np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
               df_clean.Interval[t_condition].dt.total_seconds().sum())*100
print 'Percentage of Time in Electric Pre Software Change: %.2f%%' %util

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')
print '\nPost Software Change: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Electric Post Software Change: %d hours' %np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),3600)
print 'Hybrid Post Software Change: %d hours' %np.divide(df_clean.Interval[h_condition].dt.total_seconds().sum(),3600)
util=np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
               df_clean.Interval[t_condition].dt.total_seconds().sum())*100
print 'Percentage of Time in Electric Post Software Change: %.2f%%' %util

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean.DummySpeed!='Moving on Highway')
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean.DummySpeed!='Moving on Highway')
print '\nPre Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Electric Pre Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),3600)
print 'Hybrid Pre Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[h_condition].dt.total_seconds().sum(),3600)
util=np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
               df_clean.Interval[t_condition].dt.total_seconds().sum())*100
print 'Percentage of Time in Electric Pre Software Change w/o Highway: %.2f%%' %util

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean.DummySpeed!='Moving on Highway')
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean.DummySpeed!='Moving on Highway')
print '\nPost Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Electric Post Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),3600)
print 'Hybrid Post Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[h_condition].dt.total_seconds().sum(),3600)
util=np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
               df_clean.Interval[t_condition].dt.total_seconds().sum())*100
print 'Percentage of Time in Electric Post Software Change w/o Highway: %.2f%%' %util

pre=df_clean['Electric_Bool'][(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean['Interval']>pd.Timedelta('0 seconds'))].sample(n=250000,replace=False,random_state=1)
post=df_clean['Electric_Bool'][(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean['Interval']>pd.Timedelta('0 seconds'))].sample(n=250000,replace=False,random_state=1)
print '\n%d %d' %(pre.shape[0],post.shape[0])
stats.ttest_ind(pre,post,equal_var=False)


# In[768]:


print 'AVERAGE MOVING SPEED'

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean['VehicleSpeed(mph)']>0)
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['VehicleSpeed(mph)']>0)
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['VehicleSpeed(mph)']>0)
print '\nPre Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Electric Pre Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Hybrid Pre Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean['VehicleSpeed(mph)']>0)
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['VehicleSpeed(mph)']>0)
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['VehicleSpeed(mph)']>0)
print '\nPost Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Electric Post Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Hybrid Post Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')
print '\nPre Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Electric Pre Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Hybrid Pre Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')
print '\nPost Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Electric Post Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Hybrid Post Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()


# In[769]:


print 'DAC DISTANCES'

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['DAC']==True)
print '\nPre Software Change: %d miles' %df_clean['Distance'][t_condition].sum()
print 'Electric Pre Software Change: %d miles' %df_clean['Distance'][e_condition].sum()
print 'Hybrid Pre Software Change: %d miles' %df_clean['Distance'][h_condition].sum()
util=np.divide(df_clean['Distance'][e_condition].sum(),df_clean['Distance'][t_condition].sum())*100
print 'Percentage of Distance in Electric Pre Software Change: %.2f%%' %util

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['DAC']==True)
print '\nPost Software Change: %d miles' %df_clean['Distance'][t_condition].sum()
print 'Electric Post Software Change: %d miles' %df_clean['Distance'][e_condition].sum()
print 'Hybrid Post Software Change: %d miles' %df_clean['Distance'][h_condition].sum()
util=np.divide(df_clean['Distance'][e_condition].sum(),df_clean['Distance'][t_condition].sum())*100
print 'Percentage of Distance in Electric Post Software Change: %.2f%%' %util

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
print '\nPre Software Change w/o Highway: %d miles' %df_clean['Distance'][t_condition].sum()
print 'Electric Pre Software Change w/o Highway: %d miles' %df_clean['Distance'][e_condition].sum()
print 'Hybrid Pre Software Change w/o Highway: %d miles' %df_clean['Distance'][h_condition].sum()
util=np.divide(df_clean['Distance'][e_condition].sum(),df_clean['Distance'][t_condition].sum())*100
print 'Percentage of Distance in Electric Pre Software Change w/o Highway: %.2f%%' %util

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
print '\nPost Software Change w/o Highway: %d miles' %df_clean['Distance'][t_condition].sum()
print 'Electric Post Software Change w/o Highway: %d miles' %df_clean['Distance'][e_condition].sum()
print 'Hybrid Post Software Change w/o Highway: %d miles' %df_clean['Distance'][h_condition].sum()
util=np.divide(df_clean['Distance'][e_condition].sum(),df_clean['Distance'][t_condition].sum())*100
print 'Percentage of Distance in Electric Post Software Change w/o Highway: %.2f%%' %util


# In[772]:


print 'DAC TIME'

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['DAC']==True)
print '\nPre Software Change: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Electric Pre Software Change: %d hours' %np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),3600)
print 'Hybrid Pre Software Change: %d hours' %np.divide(df_clean.Interval[h_condition].dt.total_seconds().sum(),3600)
util=np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
               df_clean.Interval[t_condition].dt.total_seconds().sum())*100
print 'Percentage of Time in Electric Pre Software Change: %.2f%%' %util

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['DAC']==True)
print '\nPost Software Change: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Electric Post Software Change: %d hours' %np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),3600)
print 'Hybrid Post Software Change: %d hours' %np.divide(df_clean.Interval[h_condition].dt.total_seconds().sum(),3600)
util=np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
               df_clean.Interval[t_condition].dt.total_seconds().sum())*100
print 'Percentage of Time in Electric Post Software Change: %.2f%%' %util

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
print '\nPre Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Electric Pre Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),3600)
print 'Hybrid Pre Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[h_condition].dt.total_seconds().sum(),3600)
util=np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
               df_clean.Interval[t_condition].dt.total_seconds().sum())*100
print 'Percentage of Time in Electric Pre Software Change w/o Highway: %.2f%%' %util

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
print '\nPost Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[t_condition].dt.total_seconds().sum(),3600)
print 'Electric Post Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),3600)
print 'Hybrid Post Software Change w/o Highway: %d hours' %np.divide(df_clean.Interval[h_condition].dt.total_seconds().sum(),3600)
util=np.divide(df_clean.Interval[e_condition].dt.total_seconds().sum(),
               df_clean.Interval[t_condition].dt.total_seconds().sum())*100
print 'Percentage of Time in Electric Post Software Change w/o Highway: %.2f%%' %util


# In[773]:


print 'DAC AVERAGE MOVING SPEED'

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean['VehicleSpeed(mph)']>0)&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean['DAC']==True)
print '\nPre Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Electric Pre Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Hybrid Pre Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean['VehicleSpeed(mph)']>0)&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean['DAC']==True)
print '\nPost Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Electric Post Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Hybrid Post Software Change: %d mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

t_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']<dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
print '\nPre Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Electric Pre Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Hybrid Pre Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()

t_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
e_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Electric')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
h_condition=(df_clean['Date_Time']>dt.datetime(2017,10,1))&(df_clean.DummyPtMode=='Hybrid')&(df_clean['VehicleSpeed(mph)']>0)&(df_clean.DummySpeed!='Moving on Highway')&(df_clean['DAC']==True)
print '\nPost Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][t_condition].mean()
print 'Electric Post Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][e_condition].mean()
print 'Hybrid Post Software Change w/o Highway: %d mph' %df_clean['VehicleSpeed(mph)'][h_condition].mean()


# In[4]:


df_shifts=df_clean[['Shift','VehicleSpeed(mph)','Score','DAC','DAC_adj','Electric_Bool',
                    'Speed_Bool','Dist_Bool']].groupby(by='Shift').mean()
df_shifts['Distance']=df_clean[['Shift','Distance']].groupby(by='Shift').sum()
print df_shifts.head(10)
bins = np.linspace(0, df_shifts['Distance'].max(), 50)
pyplot.hist(df_shifts['Distance'], bins, alpha=.5, label='Distance')
pyplot.legend(loc='upper right')
pyplot.show()
bins = np.linspace(0, df_shifts['Electric_Bool'].max(), 50)
pyplot.hist(df_shifts['Electric_Bool'], bins, alpha=.5, label='Electric')
pyplot.legend(loc='upper right')
pyplot.show()


# In[812]:


df_clean.to_csv('CMM297_Final_LL.csv',index=False)


# In[14]:


#df_clean=pd.read_csv('CMM297_Final_LL.csv')
df_clean['Date_Time']=pd.to_datetime(df_clean['Date_Time'])
df_clean['Shift']=df_clean['Date_Time'].dt.date


# In[164]:


df_clean[df_clean['Date_Time']<dt.datetime(2017,10,1)].count()


# In[163]:


df_clean.count()

