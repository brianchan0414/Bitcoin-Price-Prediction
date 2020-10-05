#!/usr/bin/env python
# coding: utf-8

##### Exploratory Analysis ######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from numpy import log
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/newuser/Downloads/data.csv",index_col="date")

df.head(5)


#### Central Tendency ####
sns.distplot( df["BTC_Price"], bins=100 )
plt.title("BTC_Price")
plt.show()


sns.distplot( df["BTC_network_hashrate"], bins=100 )
plt.title("BTC_network_hashrate")
plt.show()


sns.distplot( df["Estimated_TX_Volume_USD_BTC"], bins=100 )
plt.title("Estimated_TX_Volume_USD_BTC")
plt.show()


sns.distplot( df["Nasdaq_composite_index"], bins=100 )
plt.title("Nasdaq_composite_index")
plt.show()


sns.distplot( df["DJI"], bins=100 )
plt.title("DJI")
plt.show()


sns.distplot( df["Estimated_TX_Volume_USD_BTC"], bins=100 )
plt.title("Litecoin_Price")
plt.show()


sns.distplot( df["Gold_in_USD"], bins=100 )
plt.title("Gold_in_USD")
plt.show()


##### Trend #######

df["BTC_Price"].plot(figsize=(15,6))
df["BTC_Price"].rolling(window=50).mean().plot(figsize=(15,6),label='50 Day MA')
df["BTC_Price"].rolling(window=200).mean().plot(figsize=(15,6), label='200 Day MA')
plt.title("BTC_Price_Trend")
plt.legend(loc='upper left')


df["BTC_network_hashrate"].plot(figsize=(15,6))
df["BTC_network_hashrate"].rolling(window=50).mean().plot(figsize=(15,6),label='50 Day MA')
df["BTC_network_hashrate"].rolling(window=200).mean().plot(figsize=(15,6), label='200 Day MA')
plt.title("BTC_network_hashrate_Trend")
plt.legend(loc='upper left')


df["Estimated_TX_Volume_USD_BTC"].plot(figsize=(15,6))
df["Estimated_TX_Volume_USD_BTC"].rolling(window=50).mean().plot(figsize=(15,6),label='50 Day MA')
df["Estimated_TX_Volume_USD_BTC"].rolling(window=200).mean().plot(figsize=(15,6), label='200 Day MA')
plt.title("Estimated_TX_Volume_USD_BTC_Trend")
plt.legend(loc='upper left')


df["Nasdaq_composite_index"].plot(figsize=(15,6))
df["Nasdaq_composite_index"].rolling(window=50).mean().plot(figsize=(15,6),label='50 Day MA')
df["Nasdaq_composite_index"].rolling(window=200).mean().plot(figsize=(15,6), label='200 Day MA')
plt.title("Nasdaq_composite_index_Trend")
plt.legend(loc='upper left')



df["DJI"].plot(figsize=(15,6))
df["DJI"].rolling(window=50).mean().plot(figsize=(15,6),label='50 Day MA')
df["DJI"].rolling(window=200).mean().plot(figsize=(15,6), label='200 Day MA')
plt.title("DJI_Trend")
plt.legend(loc='upper left')


df["Litecoin_Price"].plot(figsize=(15,6))
df["Litecoin_Price"].rolling(window=50).mean().plot(figsize=(15,6),label='50 Day MA')
df["Litecoin_Price"].rolling(window=200).mean().plot(figsize=(15,6), label='200 Day MA')
plt.title("Litecoin_Price_Trend")
plt.legend(loc='upper left')


df["Gold_in_USD"].plot(figsize=(15,6))
df["Gold_in_USD"].rolling(window=50).mean().plot(figsize=(15,6),label='50 Day MA')
df["Gold_in_USD"].rolling(window=200).mean().plot(figsize=(15,6), label='200 Day MA')
plt.title("Gold_in_USD_Trend")
plt.legend(loc='upper right')


###### Spread ######

df_2013 = df.loc['29/04/2013':'31/12/2013']
df_2014 = df.loc['02/01/2014':'31/12/2014']
df_2015 = df.loc['02/01/2015':'31/12/2015']
df_2016 = df.loc['04/01/2016':'30/12/2016']
df_2017 = df.loc['03/01/2017':'21/07/2017']


box_plot_data=[df_2013["BTC_Price"], df_2014["BTC_Price"],df_2015["BTC_Price"],df_2016["BTC_Price"], df_2017["BTC_Price"]]
plt.boxplot(box_plot_data,patch_artist=True,labels=['2013','2014','2015','2016', '2017'])
plt.title("Bitcoin")
plt.xlabel('Years')
plt.ylabel('Bitcoin price')
plt.show()



df_2013["BTC_Price"].describe()

df_2014["BTC_Price"].describe()

df_2015["BTC_Price"].describe()

df_2016["BTC_Price"].describe()

df_2017["BTC_Price"].describe()


box_plot_data_1=[df_2013["DJI"], df_2014["DJI"],df_2015["DJI"],df_2016["DJI"], df_2017["DJI"]]
plt.boxplot(box_plot_data_1,patch_artist=True,labels=['2013','2014','2015','2016', '2017'])
plt.title("DJI")
plt.xlabel('Years')
plt.ylabel('DJI price')
plt.show()



df_2013["Litecoin_Price"].describe()

df_2014["Litecoin_Price"].describe()

df_2015["Litecoin_Price"].describe()

df_2016["Litecoin_Price"].describe()

df_2017["Litecoin_Price"].describe()



box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['2013','2014','2015','2016', '2017'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
 
plt.show()

box_plot_data=[df_2013["Estimated_TX_Volume_USD_BTC"], df_2014["Estimated_TX_Volume_USD_BTC"],df_2015["Estimated_TX_Volume_USD_BTC"],df_2016["Estimated_TX_Volume_USD_BTC"], df_2017["Estimated_TX_Volume_USD_BTC"]]
plt.boxplot(box_plot_data,patch_artist=True,labels=['2013','2014','2015','2016', '2017'])
plt.title("Estimated_TX_Volume_USD_BTC")
plt.show()


box_plot_data=[df_2013["BTC_Price"], df_2014["BTC_Price"],df_2015["BTC_Price"],df_2016["BTC_Price"], df_2017["BTC_Price"]]
plt.boxplot(box_plot_data,patch_artist=True,labels=['2013','2014','2015','2016', '2017'])
plt.title("Bitcoin_Price")
plt.show()

box=plt.boxplot(box_plot_data_1,vert=0,patch_artist=True,labels=['2013','2014','2015','2016', '2017'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
 
plt.show()


#### Correlation Coefficient ####

sns.plot(x=df["Difficulty_BTC"], y=df["BTC_network_hashrate"], kind="reg")


f, ax = plt.subplots(figsize=(8, 8))
sns.regplot(x=df["Difficulty_BTC"], y=df["BTC_network_hashrate"],ax=ax);


f, ax = plt.subplots(figsize=(8, 8))
sns.regplot(x=df["NUAU - BTC"], y=df["Number TX - BTC"],ax=ax);


f, ax = plt.subplots(figsize=(8, 8))
sns.regplot(x=df["Estimated_TX_Volume_USD_BTC"], y=df["BTC_Price"],ax=ax);



##### Seasonal Pattern ####


df_2014["Estimated_TX_Volume_USD_BTC"].plot(figsize=(15,6),label='2014')
df_2015["Estimated_TX_Volume_USD_BTC"].plot(figsize=(15,6),label='2015')
df_2016["Estimated_TX_Volume_USD_BTC"].plot(figsize=(15,6),label='2016')
plt.legend(loc='upper left')
plt.title("Bitcoin_Seasonal Pattern")



df_2014["BTC_network_hashrate"].plot(figsize=(15,6),label='2014')
df_2015["BTC_network_hashrate"].plot(figsize=(15,6),label='2015')
df_2016["BTC_network_hashrate"].plot(figsize=(15,6),label='2016')
plt.legend(loc='upper left')
plt.title("BTC_network_hashrate_Seasonal Pattern")


df_2014["DJI"].plot(figsize=(15,6),label='2014')
df_2015["DJI"].plot(figsize=(15,6),label='2015')
df_2016["DJI"].plot(figsize=(15,6),label='2016')
plt.legend(loc='upper left')
plt.title("DJI_index_Seasonal Pattern")


df_2014["Litecoin_Price"].plot(figsize=(15,6),label='2014')
df_2015["Litecoin_Price"].plot(figsize=(15,6),label='2015')
df_2016["Litecoin_Price"].plot(figsize=(15,6),label='2016')

plt.legend(loc='upper right')

plt.title("Litecoin_Price_Seasonal Pattern")

df_2014["Estimated_TX_Volume_USD_BTC"].plot(figsize=(15,6),label='2014')
df_2015["Estimated_TX_Volume_USD_BTC"].plot(figsize=(15,6),label='2015')
df_2016["Estimated_TX_Volume_USD_BTC"].plot(figsize=(15,6),label='2016')
plt.legend(loc='upper left')
plt.title("Estimated_TX_Volume_USD_BTC_Seasonal Pattern")

