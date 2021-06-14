#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:49:52 2020

@author: jbs
"""
#Group 1 - Boxberger, Murray, Mindlinm, Bruey-Sedano



import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.tools.eval_measures as sms_eval
import seaborn as sns
import matplotlib.pyplot as plt    
from scipy import stats 
pd.set_option('display.max_columns', 100)

#1) Import Sales.csv
sales = pd.read_csv("/Users/jbs/Downloads/Sale.csv", parse_dates = [3], low_memory=False)



#2) Make sure that sales price is numeric and document date is datetime
sales.dtypes
sales.head()

#Filter the data to only include rows with: 
#-	sales transactions with document dates in 2012 or later
#-	sales prices greater than 100,000
#-	sales instrument type equal to statutory warranty deeds1
#-	property class type equal to improved property1
#-	property type equal to household, single family unit1
#-	principal use type equal to residential, and1
#-	no sales warning (indicated using an empty space, i.e., ' ')
sales.DocumentDate = pd.to_datetime(sales.DocumentDate)  
sales_filtered = sales[(sales.DocumentDate.dt.year>=2012) & (sales.SalePrice > 100000) & (sales.SaleInstrument == 3) & (sales.PropertyClass == 8) & (sales.PropertyType == 11) & (sales.PrincipalUse == 6) & (sales.SaleWarning == ' ')] #(sales.SaleInstrument == 3) Statutory Warranty Deed  & sales.PropertyClass == 8 Improved property, PropertyType = 11 Household, Single Family Unit, sales.PrincipalUse = 6 residential, there is no sales warning (indicated by ' ')
sales.describe

#3) Drop all columns except 'Major', 'Minor', 'DocumentDate', 'SalePrice'
sales_filtered = sales_filtered[['Major', 'Minor', 'DocumentDate', 'SalePrice']]

#alternatively, use: sales_filtered = sales_filtered.drop(['ExciseTaxNbr', 'RecordingNbr', 'Volume', 'Page', 'PlatNbr', 'PlatType', 'PlatLot', 'PlatBlock', 'SellerName', 'BuyerName', 'PropertyType', 'PrincipalUse','SaleInstrument', 'AFForestLand', 'AFCurrentUseLand', 'AFNonProfitUse','AFHistoricProperty', 'SaleReason', 'PropertyClass', 'SaleWarning'], axis=1)

#4) Create a new field, TransactionYear defined as DocumentDate year.
sales_filtered['TransactionYear'] = sales_filtered.DocumentDate.dt.year

#5) Import ResidentialBuilding.csv.
houses = pd.read_csv("/Users/jbs/Downloads/ResidentialBuilding.csv", low_memory=False)

#6) NbrLivingUnits = 1 only
houses_filtered = houses[(houses.NbrLivingUnits == 1)]

#7) ZipCode, SqFtTotLiving is not null and do not contain an empty space
houses_filtered = houses_filtered.dropna(subset=['ZipCode', 'SqFtTotLiving'])
houses_filtered = houses_filtered[(houses_filtered.ZipCode != ' ') & (houses_filtered.SqFtTotLiving != ' ')]
houses_filtered.dtypes

#8) Remove four last digits and the dash of nine digit zip
houses_filtered['ZipCode'] = houses_filtered['ZipCode'].str[:5]

#9) Remove Address, BuildingNumber, BldgNbr, NbrLivingUnits, Fraction, StreetName, StreetType, DirectionSuffix, ViewUtilization, BldgGradeVar, PcntNetCondition, Obsolescence, PcntComplete, and PcntComplete
houses_filtered = houses_filtered.drop(['Address', 'BuildingNumber', 'BldgNbr', 'NbrLivingUnits', 'Fraction', 'StreetName', 'StreetType', 'DirectionSuffix', 'ViewUtilization', 'BldgGradeVar', 'PcntNetCondition', 'Obsolescence', 'PcntComplete', 'PcntComplete'], axis=1)

#10_ Replace year renovated 0s with YrBuilt values
houses_filtered.YrRenovated = np.where(houses_filtered.YrRenovated == 0, houses_filtered.YrBuilt, houses_filtered.YrRenovated)

#Or Boolean Indexing:
#houses_filtered.loc[houses_filtered.YrRenovated == 0, 'YrRenovated'] = houses_filtered.YrBuilt

#11 and 12) Create dummy variables for HeatSource, HeatSystem, and Transaction Year
#I commenented out this code below because of things I noted later, e.g., that HeatSource dummies
#are not very beneficial but that perhaps HeatSouce related to solar add value (though we are in Seattle...)
#Also, year appears to be linearly related to sales prices, so no need to create dummy variables.

#house_sales  = pd.get_dummies(house_sales , columns=['HeatSource', 'HeatSystem','TransactionYear'], drop_first=True)
#house_sales['TransactionYear'] = house_sales.DocumentDate.dt.year

#11) Merge with the Sales data. Only keep rows from the two tables with matching column values in Major and Minor.
houses_filtered[['Major','Minor']]=houses_filtered[['Major','Minor']].astype('str')
house_sales = pd.merge(sales_filtered, houses_filtered, how='inner', on=['Major', 'Minor'])

#12) Remove rows with the same Major, Minor, and DocumentDate values (do not remove rows with 
#the same values in only one or two of these three values).
house_sales.drop_duplicates(subset=['Major', 'Minor','DocumentDate'], keep='first', inplace=True)

#13) Create a new field, RenovationAge defined as DocumentDate year - YrRenovated.
#Note that YrRenovated can be after TransactionYear (YrRenovated is associated with the house, not the transaction).
#We only want the RenovationAge as of the transaction date and we as such can only include YrRenovated if it is earlier
#than the transaction date.  We have one problem... what if the transaction year = renovation year, we then do not know
#if the renovation was performed after or before the transaction.  Perhaps we can say that for all transaction with 
#transaction year = renovation year with a DocumentDate (1) before 7/1 of the year have not been renovated and (2) on or after 7/1
#have been renovated.  However, even with this we still have a problem... if the renovation year is after the document year (or classified by rule 1)
#then we do not know when the most recent renovation prior to the DocumentDate took place.  
#However, we only have 84 transactions with renovation year after the transaction year and removing 
#these observations are probably not going to change the model much and I therefore set these 
#observations to have Renovation Age = null (and if renovation age is included in the model, then these 
#observation are automatically excluded when we run the regression model 
house_sales['RenovationAge'] = np.where((house_sales.TransactionYear>house_sales.YrRenovated) | ((house_sales.TransactionYear == house_sales.YrRenovated) & (house_sales.DocumentDate.dt.month < 7)),house_sales.TransactionYear - house_sales.YrRenovated, np.NaN)

#14) Create a new field, Age defined as DocumentDate year - YrBuilt.
#There are, however, 43 observations with year built after the transaction year and 296 that are built in the same year that they
#are sold.  I will set the Age variable to null for the 43 observations, but I will assume that most houses are not demolished and built 
#in the same year that they are purchased and instead assume that these houses were built before the DocumentDate (we could have, similarly to YrRenovated,
#assume that houses with transaction year = year built that are purchased in the first n months are demolished and rebuilt).
house_sales['Age'] = np.where(house_sales.TransactionYear>=house_sales.YrBuilt, house_sales.TransactionYear - house_sales.YrBuilt, np.NaN)

#15) Remove YrRenovated and YrBuilt
house_sales = house_sales.drop(['YrRenovated', 'YrBuilt'], axis=1)

#16) Create a new variable, PricePerSqFoot defined as SalePrice/SqFtTotLiving 
house_sales['PricePerSqFoot'] = house_sales.SalePrice/house_sales.SqFtTotLiving

#17) Create a new variable, PriorAveragePricePerSqFoot_Zip defined as previous 6 months average PricePerSqFoot for homes sold in the same zip code (note that you cannot include sales prices, including PricePerSqFoot, from current or later sales as independent variables because this data will not be available when making predictions using your model).
house_sales = house_sales.set_index(['DocumentDate']).sort_index()

#Here is the rolling code for rolling the prior 180 days:
#house_sales_roll = house_sales.groupby('ZipCode', as_index=False).rolling('180D', closed='left').PricePerSqFoot.mean()

#closed='left' makes the rolling window exclude the current row.
#Here is an example adjusted based on the Pandas documentation to better understand rolling 
#and closed windows:
#df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4, 5, 6, 7, 8]},
 #                 index = [pd.Timestamp('20130101 09:00:00'),
  #                         pd.Timestamp('20130101 09:00:02'),
   #                        pd.Timestamp('20130101 09:00:03'),
    #                       pd.Timestamp('20130101 09:00:05'),
     #                      pd.Timestamp('20130101 09:00:06'),
      #                     pd.Timestamp('20130101 09:00:07'),
       #                    pd.Timestamp('20130101 09:00:08'),
        #                   pd.Timestamp('20130101 09:00:09'),
         #                  pd.Timestamp('20130101 09:00:10')])

#df['max_3s']=df.rolling('3s', closed='left').B.max()
#df['mean_3s']=df.rolling('3s', closed='left').B.mean()
#df['max_by_mean_3s']=df.rolling('3s', closed='left').B.max()/df.rolling('3s', closed='left').B.mean()
#df

#Here is the code for calculating the 180 average as
#(190 sum - 10 sum) / (190 count - 10 count). Note that this creates 
#some duplicates for zip code and document date and that Pandas is not 
#able to set the values in a new column in the original dataframe directly
#because of this (this generates an error). So I first save the rolled values
#to a new dataframe, remove the duplicates, and then manually merge the values.
#Also note that I do not need to exclude the left here as the first 10 days are
#excluded.
house_sales_roll =                                 \
   ((house_sales.groupby('ZipCode').rolling('190D').PricePerSqFoot.sum() -
   house_sales.groupby('ZipCode').rolling('10D').PricePerSqFoot.sum()) /                        
   (house_sales.groupby('ZipCode').rolling('190D').PricePerSqFoot.count() -
    house_sales.groupby('ZipCode').rolling('10D').PricePerSqFoot.count()))
         
#Rather than setting DocumentDate as the index and rolling on the index, we can also use on='documentdate', e.g., 
#house_sales.groupby('ZipCode', as_index=False).rolling('180D', on='DocumentDate')['PricePerSqFoot'].mean()

#Here is slower version that uses list comprehension for obtaining an average over -10 to -190. 
#house_sales = house_sales.set_index(['ZipCode', 'DocumentDate']).sort_index()
#house_sales['Average_PricePerSqFoot'] = [house_sales.loc[(house_sales.ZipCode == row['ZipCode'])].loc[i-timedelta(190):i-timedelta(10), 'PricePerSqFoot'].mean() for i, row in house_sales.iterrows()]

house_sales_roll=house_sales_roll.reset_index()
house_sales_roll.rename(columns={'PricePerSqFoot': 'Average_PricePerSqFoot'},inplace=True)
house_sales_roll = house_sales_roll.drop_duplicates(subset=['ZipCode', 'DocumentDate'], keep='last')

house_sales = pd.merge(house_sales, house_sales_roll, how='left', on=['ZipCode', 'DocumentDate'])
#house_sales[['ZipCode', 'DocumentDate', 'PricePerSqFoot', 'Average_PricePerSqFoot']].sort_values(by=['ZipCode', 'DocumentDate']).head(50) 

print(house_sales.dtypes)



#Data Exploration and Model Building Case 1

#Introducing House_Sales_Subset as our new DataFrame which we will be transforming
house_sales_subset = house_sales[(house_sales.TransactionYear<=2017)]

#SalePrice transformation
house_sales_subset.SalePrice = np.log(house_sales_subset.SalePrice)
sns.distplot(house_sales_subset.SalePrice, kde=False, fit=stats.norm)
plt.figure()

#Average_PricePerSqFt transformation
house_sales_subset.Average_PricePerSqFoot = np.log(house_sales_subset.Average_PricePerSqFoot)
sns.distplot(house_sales_subset.Average_PricePerSqFoot, kde=False, fit=stats.norm)
plt.figure()



print(house_sales_subset.describe())


#Print all displots of our variables

sns.distplot(house_sales_subset.SalePrice, kde=False, fit=stats.norm)
plt.figure()


sns.distplot(house_sales_subset.SqFt1stFloor, kde=False, fit=stats.norm)
plt.figure() 

sns.distplot(house_sales_subset.RenovationAge, kde=False, fit=stats.norm)
plt.figure() 

sns.distplot(house_sales_subset.SqFtTotLiving, kde=False, fit=stats.norm)
plt.figure() 

sns.distplot(house_sales_subset.Age, kde=False, fit=stats.norm)
plt.figure() 

sns.distplot(house_sales_subset.Average_PricePerSqFoot, kde=False, fit=stats.norm)
plt.figure() 

sns.distplot(house_sales_subset.BrickStone, kde=False, fit=stats.norm)
plt.figure() 



#Cleaning
house_sales_subset = house_sales_subset.replace(np.inf, np.nan)
house_sales_subset = house_sales_subset.replace(-np.inf, np.nan)
print(house_sales_subset.shape)

house_sales_subset.drop(['HeatSysytem', 'HeatSource'], axis=1, inplace=True)




house_sales_subset.dropna(subset=['Average_PricePerSqFoot'], inplace=True)



print(house_sales_subset.Average_PricePerSqFoot.describe())

print(house_sales_subset.groupby(['SqFtHalfFloor']).count())

#Binary Replacements 

house_sales_subset['HalfFllor'] = np.where(house_sales_subset.SqFtHalfFloor>0,1,0)

house_sales_subset['SqFtHalfFloor_Binary'] = np.where(house_sales_subset.SqFtHalfFloor>0,1,0)
house_sales_subset['SqFt2ndFloor_Binary'] = np.where(house_sales_subset.SqFt2ndFloor>0,1,0)
house_sales_subset['SqFtUpperFloor_Binary'] = np.where(house_sales_subset.SqFtUpperFloor>0,1,0)
house_sales_subset['SqFtUnfinFull_Binary'] = np.where(house_sales_subset.SqFtUnfinFull>0,1,0)
house_sales_subset['SqFtUnfinHalf_Binary'] = np.where(house_sales_subset.SqFtUnfinHalf>0,1,0)
house_sales_subset['SqFtTotBasement_Binary'] = np.where(house_sales_subset.SqFtTotBasement>0,1,0)
house_sales_subset['SqFtFinBasement_Binary'] = np.where(house_sales_subset.SqFtFinBasement>0,1,0)
house_sales_subset['FinBasementGrade_Binary'] = np.where(house_sales_subset.FinBasementGrade>0,1,0)
house_sales_subset['SqFtGarageBasement_Binary'] = np.where(house_sales_subset.SqFtGarageBasement>0,1,0)
house_sales_subset['SqFtGarageAttached_Binary'] = np.where(house_sales_subset.SqFtGarageAttached>0,1,0)
house_sales_subset['SqFtOpenPorch_Binary'] = np.where(house_sales_subset.SqFtOpenPorch>0,1,0)
house_sales_subset['SqFtEnclosedPorch_Binary'] = np.where(house_sales_subset.SqFtEnclosedPorch>0,1,0)
house_sales_subset['SqFtDeck_Binary'] = np.where(house_sales_subset.SqFtDeck>0,1,0)
house_sales_subset['HeatSystem_Binary'] = np.where(house_sales_subset.HeatSystem>0,1,0)
house_sales_subset['HeatSource_Binary'] = np.where(house_sales_subset.HeatSource>0,1,0)
house_sales_subset['BrickStone_Binary'] = np.where(house_sales_subset.BrickStone>0,1,0)
house_sales_subset['Bedrooms_Binary'] = np.where(house_sales_subset.Bedrooms>0,1,0)
house_sales_subset['BathHalfCount_Binary'] = np.where(house_sales_subset.BathHalfCount>0,1,0)
house_sales_subset['Bath3qtrCount_Binary'] = np.where(house_sales_subset.Bath3qtrCount>0,1,0)
house_sales_subset['BathFullCount_Binary'] = np.where(house_sales_subset.BathFullCount>0,1,0)





print(house_sales_subset.describe())


# OLS regression Model Prelim 


model_results = sm.ols(formula='SalePrice ~ TransactionYear + SqFtHalfFloor + SqFt2ndFloor + SqFtUpperFloor + SqFtTotBasement + SqFtGarageAttached + BrickStone + Condition + RenovationAge + Age + PricePerSqFoot', data=house_sales_subset).fit()




