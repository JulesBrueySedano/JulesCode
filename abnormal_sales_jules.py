#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 07:54:43 2020

@author: jbs
"""

import pandas as pd 
import statsmodels.formula.api as sm



financials = pd.read_csv("/Users/jbs/Downloads/Financials.csv")

#printing the first 20 rows
print("\na:")

print(financials.head(20)) 


#printing the last 20 rows

print(financials.tail(20))

print("\nb:")

# stats stuff

print(financials.describe())

# data types

print(financials.dtypes)

print("\nc:")


# give me first 20 rows of data date columns 

print(financials.datadate.head(20))
 
#printing daydate, gvkey column and give me 20 rows
print(financials[['datadate', 'gvkey']].head(20))

#select all rows that have sales under 100
print("\nd:")
print(financials[financials.SALE < 100].head(20))
#take away the negative values
print("\nd2:")
print(financials[(financials.SALE < 100) & (financials.SALE > 0)].head(20)) 

#create a new columns
print("\ne:")

financials['SalesToAssets'] = financials.SALE/financials.AT
print(financials.head(20))

#New Functions find all NULL Values in columns first and sum them
print("\nf:")

print(financials.isnull().sum())
# sum of null for each row!

print("\ng:")
print(financials.isnull().sum(axis=1).head(20))


#new column to find null values
print("\nh:")

financials['NumNull'] = financials.isnull().sum(axis=1)
print(financials.head(20))

print("\nh2:")

financials = financials.sort_values('NumNull', ascending=False)
print(financials.head(20))

print("\nI:")



# or like this 

financials = financials.drop(['IBE', 'OCF', 'PPE'], axis=1)
print(financials.isnull().sum())

print("\nJ:")

financials.dropna(subset=['SALE'], inplace=True)
print(financials.isnull().sum())


print("\nK:")
financials.dropna(thresh=11, inplace=True)
print(financials.isnull().sum())

print("\nL:")

#replace missing values in one or in one or more columns with global column means, on columns AP, REC and BV

financials['AP'].fillna(financials['AP'].mean(), inplace=True)

#Globalmeans option p2
financials.AP.fillna(financials['AP'].mean(), inplace=True)
financials.REC.fillna(financials['REC'].mean(), inplace=True)
financials.BV.fillna(financials['BV'].mean(), inplace=True)

#print(financials.isnull().sum())  // 122954 mean = 4625.18800849766

print(financials[(financials.AP.isnull())])
print(financials.AP.mean())

print(financials[financials.gvkey == 122954])


print("\nM:")

#replace missing values in one or more columbs with industry means (b ased on 2 digit sic_)

financials.MV.fillna(financials.groupby('2_digit_sic')['MV'].transform("mean"), inplace=True)
print(financials[(financials.MV.isnull())])

#create descript stats, and remove any negative values where they should have potivve values

print("\nN:")

print(financials.describe())
financials = financials[(financials.COGS >= 0)]
financials = financials[(financials.SALE > 0)]
financials = financials[(financials.XOPR >= 0)]
financials = financials[(financials.MV > 0)]
financials = financials[(financials.EMP > 0)]
print(financials.describe())

#standardize
print("\nO:")

financials.EMP = (financials.EMP - financials.EMP.mean())/financials.EMP.std()
print(financials.EMP.std())
print(financials.EMP.mean())

print("\nP:")

financials.COGS = (financials.COGS - financials.COGS.min())/(financials.COGS.max() - financials.COGS.min())
print(financials.COGS.min())
print(financials.COGS.max())

# binning
print("\nQ:")

financials['Binned_Sale']=pd.qcut(financials.SALE, 10, labels=False)

print(financials.describe())


#  winsorizing 

print(financials.SALE.quantile(q=0.02))

print(financials.SALE.quantile(q=0.98))

import numpy as np

financials.SALE = np.where(financials.SALE < financials.SALE.quantile(q=0.02), financials.SALE.quantile(q=0.02), financials.SALE)



financials.SALE = np.where(financials.SALE > financials.SALE.quantile(q=0.98), financials.SALE.quantile(q=0.98), financials.SALE)


financials.SALE = np.where(financials.SALE < financials.SALE.quantile(q=0.02), financials.SALE.quantile(q=0.02), financials.SALE)


print(financials.SALE.quantile(q=0.02))
print(financials.SALE.quantile(q=0.97))
print(financials.SALE.quantile(q=0.98))
print(financials.SALE.quantile(q=0.99))




print("\nR:")

#checking and correct data types

print(financials.dtypes)
print(financials.head())
financials.datadate = pd.to_datetime(financials.datadate,format='%Y%m%d')
print(financials.dtypes)
print(financials.head())

#creating new variables

financials.sort_values(by=['gvkey','datadate'], ascending=[True, True], inplace = True)
print(financials.head(20))

#shifting variables based on historical context

financials['prevSALE'] = financials.SALE.shift(1)
financials['prevEMP'] = financials.EMP.shift(1)
financials['prevCOGS'] = financials.COGS.shift(1)
financials['prevREC'] = financials.REC.shift(1)
financials['prevXOPR'] = financials.XOPR.shift(1)
financials['prevAP'] = financials.AP.shift(1)
financials['prevAT'] = financials.AT.shift(1)
print(financials.head(5))

#drop row with missing lagged data
print("\nS:")
print(financials.describe())
print("\nS2:")

financials['Year'] = financials.datadate.dt.year
financials = financials[((financials.Year -1) == financials.Year.shift(1)) & (financials.gvkey == financials.gvkey.shift(1))]
print(financials.describe())

#creating change variables

financials['Scaled_Sales'] = financials.SALE / financials.AT 

financials['Scaled_PrevSales'] = financials.prevSALE / financials.prevAT 

financials['Scaled_EMP'] = financials.EMP / financials.AT 

financials['Scaled_EMPChange'] = financials.EMP - financials.prevEMP / financials.AT 


financials['Scaled_COGS'] = financials.COGS / financials.AT 

financials['Scaled_COGSChange'] = (financials.COGS - financials.prevCOGS) / financials.AT 


financials['Scaled_REC'] = financials.REC / financials.AT 

financials['Scaled_RECChange'] = (financials.REC - financials.prevREC) / financials.AT 


financials['Scaled_XOPR'] = financials.XOPR / financials.AT 

financials['Scaled_XOPRChange'] = (financials.XOPR - financials.prevXOPR) / financials.AT 

financials['Scaled_AP'] = financials.AP / financials.AT 

financials['Scaled_APChange'] = (financials.AP - financials.prevAP)/ financials.AT 

financials['BookToMarket'] = financials.BV / financials.MV


#create boxplots and histograms


import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt



sns.distplot(financials.SALE, fit=stats.norm)
print("\nT1:")

sns.distplot(financials.SALE, kde=True, fit=stats.norm)
print("\nT2:")

financials.SALE = np.log(financials.SALE)
sns.distplot(financials.SALE, kde=True, fit=stats.norm)



#boxplots and violin plots
sns.boxplot(x=financials.SALE)

sns.violinplot(x=financials.SALE, color="0.25")

plt.figure()

# creating additional variables based on industry means 

print(financials.groupby(['2_digit_sic', 'Year'])['SALE'].mean())

#create a new dataframe that stores this info

industry_average_sales = financials.groupby(['2_digit_sic', 'Year'])['SALE'].mean()
industry_average_sales.name = 'average_sales'

print(industry_average_sales.head(15))
# want to join this new dataframe to the original data frame

print(financials[['gvkey','2_digit_sic','Year']].head(15))


#           merge - combine two tables based on column value
# pd.merge(financials, industry_average_sales, how= 'inner', on=['2_digit_sic', 'Year'])


#           join -  best for inex and table(dataframe)

financials = financials.join(industry_average_sales, how='inner', on=['2_digit_sic', 'Year'])

print(financials[['gvkey','2_digit_sic', 'Year','SALE', 'average_sales' ]].head(15))

financials.sort_values(by=['gvkey', 'Year'], ascending=[True,True], inplace=True)
print(financials[['gvkey','2_digit_sic', 'Year','SALE', 'average_sales' ]].head(15))


#        transform 

financials['SALE_Industry_Mean'] = industry_average_sales = financials.groupby(['2_digit_sic', 'Year'])['SALE'].transform("mean")


print(financials[['gvkey','2_digit_sic', 'Year','SALE', 'average_sales','SALE_Industry_Mean' ]].head(15))


#OLS regression

model_results = sm.ols(formula='Scaled_Sales ~ Scaled_PrevSales + Scaled_EMP + Scaled_EMPChange + Scaled_COGS + Scaled_COGSChange + Scaled_REC + Scaled_RECChange + Scaled_XOPR + Scaled_XOPRChange + Scaled_AP + Scaled_APChange + BookToMarket', data=financials).fit()

#robust regression w/ standard error
robust_result = model_results.get_robustcov_results(cov_type='cluster', use_t=None, goroups=financials['2_digit_sic'])
print(robust_result.summary())


print(model_results.summary())

#code of mice
from statsmodels.imputation import mice
financials_regression_columns = financials[['Scaled_Sales', 'Scaled_PrevSales', 'Scaled_EMP' ]]
imp = mice.MICEData(financials_regression_columns)
mice.MICE('Scaled_Sales ~ Scaled_PrevSales + Scaled_EMP', sm.OLS, imp).fit

#testing for homogeneity  of variamnce

financials["resials"] = model_results.resid
financials["predicted"] = model_results.fittedvalues
plt.scatter(financials.predicted, financials.residuals)
plt.title('Residuals by Predicted')
plt.xlabel('Predicted')
plt.ylabel('Residuals')


financials_subset = financials[(financials.predicted<20)]
plt.scatter(financials_subset.predicted, financials_subset.residuals)
plt.title('Residuals by Predicted')
plt.xlabel('Predicted')
plt.ylabel('Residuals')




plt.scatter(financials_subset.predicted, financials_subset.Scaled_Sales)
plt.title('Actual by Predicted')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show

#robust regression homogeneity

robust_results = model_results.get_robustcov_result(cov_type='HC3', use_t=None)
print(robust_results.summary())

new_results = sm.ols(formula='Scaled_Sales ~ Scaled_PrevSales + Scaled_EMP + Scaled_EMPChange + Scaled_COGS + Scaled_COGSChange + Scaled_REC + Scaled_RECChange + Scaled_XOPR + Scaled_XOPRChange + Scaled_AP + Scaled_APChange + BookToMarket', data=financials).fit(cov_type='HC3', use_t=None)

print(new_results.summary())


#normally distributed errprs

sns.distplot(financials.residuals, kde=False, fit=stats.norm)
plt.show()





#multicollienetity








