#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:02:02 2021

@author: jbs
"""


import numpy as np
import pandas as pd
import glob as glob
from pandas.tseries.holiday import USFederalHolidayCalendar
import seaborn as sns
import matplotlib.pyplot as plt    
from scipy import stats 
import statsmodels.formula.api as sm

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 50)

 
attendance_data_raw = pd.DataFrame()
for file_path in glob.glob(r'/Users/jbs/Desktop/Birch/*.xls*'):

    #For each workbook, create list of visible worksheets
    #First create a list of xlrd sheet objects
    sheets = pd.ExcelFile(file_path).book.sheets()

    #Then use list comprehension to create a list of only visible worksheet objects that have "Day" in cell B1 
    #and more than 9 columns in row 1.
    list_of_sheets_visible_and_day = [sheet for sheet in sheets if (sheet.visibility == 0) & (sheet.cell_value(0,1)=="Day")] 
#& (sheet.row_len(0) > 9)

    #Then based on the visibly sheets with day in cell A1, use list comprehension to create one lists of sheets for each table 
    #structure (that depend on the PrePay column).
    list_of_sheets_early = [sheet.name for sheet in list_of_sheets_visible_and_day if ((sheet.cell_value(0,4)=="PrePay") | (sheet.cell_value(0,4)=="Pre-Pay"))]
    list_of_sheets_late = [sheet.name for sheet in list_of_sheets_visible_and_day if ((sheet.cell_value(1,9)=="PrePay") | (sheet.cell_value(1,9)=="Pre-Pay"))]
    list_of_sheets_none = [sheet.name for sheet in list_of_sheets_visible_and_day if ((sheet.cell_value(0,4)!="PrePay") & (sheet.cell_value(0,4)!="Pre-Pay") & (sheet.cell_value(1,9)!="PrePay") & (sheet.cell_value(1,9)!="Pre-Pay"))]
         
    #Next import the data in the sheets that are in the three lists. The imported data is later concatenated with other dataframes
    #but the concat function only works if the objects that are concatenated are actual dataframes (even if empty), so first create empty
    #dataframes and at the same time clear the dataframes if the dataframe was populated in an earlier for loop iteration (and there is no
    #worksheets in this iteration that overwrites it)
    all_workbook_data_early = pd.DataFrame()
    all_workbook_data_late = pd.DataFrame()
    all_workbook_data_none = pd.DataFrame()
    if list_of_sheets_early:
        #Import all worksheets that have prepay early, but do not import the prepay column. Place these sheets in a data dictionary.
        #Do not import header information, instead set the column names to the values in a list from 0 to 15 (header=None) automatically
        #sets the headers to the Excel column index of the column, but since we removed one column we have a gap in the column names and the 
        #location of this gap depends on where (if at all) PrePay is located). Without consistent column headers, the concat function that is later
        #used to stack the data from the three different Excel sheet structures, will assume that the same name in different sheets means the same
        #column. To get around this I use the names argument and set it to range(16) to make sure that the columns are named 0, 1, 2,... without a gap.
        workbook_dictionary_early = pd.read_excel(file_path, sheet_name=list_of_sheets_early, skiprows=2, header=None, usecols="A:D,F:Q", nrows=33, names=[0,1,2,3,4,4,5,6,7,8,9,10,11,12,13,14,15])

        #stack all the worksheets in the data dictionary
        all_workbook_data_early = pd.concat(workbook_dictionary_early)
    if list_of_sheets_late:
        workbook_dictionary_late = pd.read_excel(file_path, sheet_name=list_of_sheets_late, skiprows=2, header=None, usecols="A:I,K:Q", nrows=33, names=[0,1,2,3,4,5,6,7,8,9,9,10,11,12,13,14,15])
        all_workbook_data_late = pd.concat(workbook_dictionary_late)

    if list_of_sheets_none:
        workbook_dictionary_none = pd.read_excel(file_path, sheet_name=list_of_sheets_none, skiprows=2, header=None, usecols="A:P", nrows=33)
        all_workbook_data_none = pd.concat(workbook_dictionary_none)

    #After importing all the sheets in the workbook that we want, concatenate (stack) these sheets and data imported (if any) from previous workbooks.
    attendance_data_raw = pd.concat([attendance_data_raw, all_workbook_data_early, all_workbook_data_late, all_workbook_data_none], axis=0, ignore_index=True)

#I next need to create the header labels (column names). 
#I obtain this information from one worksheet to then use as standard headers.
#First import worksheet 10, columns A:P, and only the first two rows.
#The use of 2014_2015 and worksheet 10 is arbitrary, I could have used
#any of the spreadsheets and any of the worksheets.
workbook_cols = pd.read_excel(r'/Users/jbs/Desktop/Birch/FY 2014 _ 2015.xls', sheet_name=10, header=None, usecols="A:D,F:P", nrows=2)

#Some headers in worksheet 10 have merged cells in row 1 and when importing these merged cells, the first cell will be 
#treated as containing the value while the other cells will be imported as containing nothing (i.e., NaNs). The second row 
#contains subcategories for merged cells (or NaN if row does not contain a merged cells). For example, if Excel contains the following header for the column Member, Guest Passes - Member Guest, and Guest Passes Pre-Paid:
#Row 1 --> Member |        Guest Passes      |
#Row 2 -->        | Member Guest | Pre-Paid  |

#Then Pandas will import this data as two rows with the following values:
#Row 1 --> Member, Guest Passes, NaN
#Row 2 --> NaN, Member Guest, Pre-Paid

#We thus loose the connection between Guest Passes and Pre-Paid.  To overcome this
#the code below fills NaN values in row 1 with the value from the left of the NaN value. All remaining
#NaN values are in row 2 are then replaced with empty strings as it is not possible to concatenate 
#NaN values with strings. The two lists containing the headers from row 1 and 2 are then concatenated
#(and a space is entered between concatenated words).  An leading or trailing spaces are then removed.
#Finally, the word Happenings is added as some other Excel tables contain an extra column with this information 
#(and the sheet we selected to use for the standard headers did not contain this column.

workbook_cols.iloc[0] = workbook_cols.iloc[0].ffill()
workbook_cols.iloc[1] = workbook_cols.iloc[1].fillna("")
df_headers = workbook_cols.iloc[0,:] + " " + workbook_cols.iloc[1,:]
df_headers = df_headers.str.strip()
df_headers = df_headers.tolist() + ['Happenings']

attendance_data_raw.columns = df_headers
####

#Data cleaning
attendance_data = attendance_data_raw.copy()

#Remove all rows with null values in Day (this removes some empty rows and rows with sheet totals and percentages)
attendance_data = attendance_data[pd.notnull(attendance_data.Day)]

#Found and fixed two dates that are incorrectly coded 
attendance_data.dtypes
attendance_data.Date = attendance_data.Date.astype(str)
print(attendance_data.Date.sort_values(ascending=True).head(20))
attendance_data.loc[attendance_data.Date=='2012-11-13 00:00:00','Date'] = '2014-11-13 00:00:00'
print(attendance_data.Date.sort_values(ascending=False).head(20))
attendance_data.loc[attendance_data.Date=='9/16/204','Date'] = '2014-09-16 00:00:00'

#Change the Date column to date time and set it to be the index (which will make resampling and rolling easier)
attendance_data.Date = pd.to_datetime(attendance_data.Date)
attendance_data = attendance_data.set_index('Date')

#Remove rows in 2019 as we only have weather day through 2019-01-01 (also takes care of some 
#empty rows.
attendance_data = attendance_data[:'2019-01-01'] 

#Check and change additional data types that are incorrect.
attendance_data.dtypes
attendance_data.College = attendance_data.College.astype(float)
attendance_data['Education Education Groups'] = attendance_data['Education Education Groups'].astype(float)

#Check that all rows have at least some full paid (the rows is otherwise odd). Confirm that all these 
#rows do not contain data we want and then remove them (I could have used dropna with an subset argument instead).
attendance_data[attendance_data['Full Paid'].isnull()]
attendance_data = attendance_data[attendance_data['Full Paid'].notnull()]

#Remaining null data means zero in that column
attendance_data.fillna(0, inplace=True)

#Find rows where Total On-Site Vistors is calculated incorrectly. If any found (I found two rows and verified 
#that they were incorrect by opening the raw excel data) then recalculate these values (I simply updated 
#all the values as the other values were correct and this updating therefore did not change them).
attendance_data[attendance_data['Total On-Site Visitors'] != attendance_data.iloc[:,1:13].sum(axis=1)]
attendance_data['Total On-Site Visitors'] = attendance_data.iloc[:,1:13].sum(axis=1)

#Calculate the DV and two IVs and only keep these three columns plus Happenings.
attendance_data['Attendance'] = attendance_data['Total On-Site Visitors'] -  attendance_data['Education Education Groups'] - attendance_data['Education Public Act.'] - attendance_data['Special Events Member Events'] - attendance_data['Special Events Paid Sp Events']
attendance_data['Education_Attendance'] = attendance_data['Education Education Groups'] + attendance_data['Education Public Act.']
attendance_data['Special_Event_Attendance'] = attendance_data['Special Events Member Events'] + attendance_data['Special Events Paid Sp Events']
attendance_data = attendance_data[['Attendance', 'Education_Attendance', 'Special_Event_Attendance', 'Happenings']]

#Examine the data to confirm that everything looks OK
attendance_data.describe()

#I saved the data here to have a copy stored that does not require rerunning all the code.
attendance_data.to_csv(r'/Users/jbs/Desktop/Birch/BirchImportClean.csv')

###################################################################
############### Importing and Cleaning Weather Data ###############
###################################################################

#First import and concatenate all weather data files. 
all_csv_files = glob.glob(r'/Users/jbs/Desktop/Birch/*.csv*')
weather_all = pd.concat((pd.read_csv(file) for file in all_csv_files), sort=False)

#Change ObservationTimeUtc to DateTime and then set the index to this variable.
weather_all['ObservationTimeUtc'] = pd.to_datetime(weather_all['ObservationTimeUtc'])
weather_all = weather_all.set_index(['ObservationTimeUtc']).sort_index()

#Then correct the time zone to have the time match pacific time correctly.
weather_all.index = weather_all.index.tz_convert('US/Pacific')

#This changes the time so that we can index and slice using Pacific Time rather than
#GMT. So if we wanted to know the weather at midnight before converting the 
#time zone we would need to use 08:00 (this is 8:00 AM in England, which is 
#midnight Pacific Time), but after the conversion we use 00:00 (which is
#correctly midnight Pacific Time.

#Only keep four variables that I expect are important (it would be even better to process more variables
#and determine statistically which variables are useful, but this make it a lot simpler to work with). 
#Based on later analysis of the data I also included the Humidity quality indicator - will remove this later.
weather_all = weather_all[['Humidity', 'TemperatureC', 'RainMillimetersRatePerHour', 'WindSpeedKph', 'Humidity-QcDataDescriptor']]


def create_box_and_dist_plots(col):
    plt.figure()
    sns.boxplot(x=col[pd.notnull(col)]) #only plot not null values (error if NaN values are plotted)
    plt.figure()    
    sns.distplot(col[pd.notnull(col)], kde=False, fit=stats.norm) 

#### Wind ####
create_box_and_dist_plots(weather_all.WindSpeedKph)
#There are some measures with very high wind, perhas incorrect data. 
#Let's isolate some of these and see when they occured.

print(weather_all.WindSpeedKph[weather_all.WindSpeedKph>55].sort_index)
#Early January 2016 and late January/early February 2016 were storms, so these observations look fine.

#Going back to the original graph - it also looks like there are a lot of 0 measures. 
#Perhaps the sensor is not sensitive enough to differentiate between very, very low wind speeds so 
#we end up with an abnormal amount of zero wind days. The graph is also indicating fairly heavy left skew
#I can perhaps include a dummy for zero and interact with a sqrt transformed wind variable. Let's what the 
#non-zero data looks like after this transformation.

create_box_and_dist_plots(np.sqrt(weather_all.WindSpeedKph[weather_all.WindSpeedKph>0]))

#This looks pretty good. So let's remmeber to later add a dummy and take the sqrt of wind.


#### Rain ####
create_box_and_dist_plots(weather_all.RainMillimetersRatePerHour)
#Most measures have no or very little rain (sunny San Diego), but there are a few observations with very heavy rain.
#Let's isolate these observations.

print(weather_all.RainMillimetersRatePerHour[weather_all.RainMillimetersRatePerHour>20].sort_index())
#2016-01-06 was a very wet day. These hourly rates appear a little high, but these were very 
#rainy days - this is also true for the other days in this list. I can try to winsorize to 
#see if it helps, but later after I have resampled the data to create daily values, I create
#a dummy variables to indicate no rain (or rain).

#### Temperature ####
create_box_and_dist_plots(weather_all.TemperatureC)
#Temps below 0 look odd
print(weather_all.TemperatureC[weather_all.TemperatureC<-70].sort_index)
#There are no values between 3.3 and -70 and 26 values below -70, there is clearly something wrong about these values
#They are on three different days. 

#Looking at 2017-03-15 a little closer, there are some extreme recordings and then no recordings (this is also
#true for all other measures). Something must have gone wrong here.
weather_all.TemperatureC['2017-03-15 12:00':'2017-03-15 19:20']
weather_all['2017-03-15 19:09':'2017-03-16 02:09:59']

#Similar thing happened two days later: 2017-03-17
weather_all.TemperatureC['2017-03-17 07':'2017-03-17 16']

#Similar thing happened on may 1st 2017
weather_all.TemperatureC['2017-05-01 11:19':'2017-05-01 13:26']

#Since the <-70 values are errors, I will replace them by NaN for now and 
#then after creating daily values below, I will replace any remaining daily 
#missing values with averages of the day before and after (there might be 
#some observations with non null values so that averages will still return some data)

#Setting all temparature errors to NaN:
weather_all.TemperatureC[weather_all.TemperatureC<-10] = np.NaN

#### Humidity ####
create_box_and_dist_plots(weather_all.Humidity)
#The low spike and the high spike look odd.  I do not know much about humidity apart from that
#between 30 and 60 percent is considered comfortable.  I assume we can have some variation in
#San Diego but it is surprising to see so much data at 80% and above, and then a really big 
#spike at 100%. The smaller spike towards the bottom that is at 5 points is even more surprising, 
#especially considering that the Sahara Desert has an average relative humidity of #25 percent.  
#That said, I do not know much about humidity and I have a feeling that a lot of 100% is not that
#odd (there is probably a ceiling effect going on). The spike at 5 percent is more concerning.
#I filtered the data to see some of the dates (2014-04-30, 2016-11-06, 2016-11-08, and 2016-11-15)
#and then went and looked up recorded weather information from other weather stations for these
#dates and while 2014-04-30 could possibly be correct, the other three dates had high humidity.
#I then went back to the raw data and noticed that these (and many other dates indentified below
#had a quality indicator for the humidty measure set to Q, which means that it passed level 1 but 
#failed level 2 or 3). I decided to code all values with Humidity-QcDataDescriptor == Q AND 
#(Humidity<10 OR Humidity=100) as NaN.
create_box_and_dist_plots(weather_all.Humidity[weather_all.Humidity<6])

#Code for looking at rows (I changed the integers for iloc to see different data).
weather_all.TemperatureC[weather_all.Humidity<6].iloc[200:220]

weather_all.Humidity[(weather_all['Humidity-QcDataDescriptor']=='Q')&((weather_all.Humidity<10)|(weather_all.Humidity==100))] = np.NaN
weather_all.drop('Humidity-QcDataDescriptor', inplace=True, axis=1)

#In addition to the two spikes, humidity also has a fairly strong left skew and what looks like a bi-modal 
#distribution.  Perhaps there is some differences between night and day or different seasons that explains the
#bi-modal distribution. While I should probably wait with exploring this further to the next section
#this made my curious so I decided to continue...

#To look at this, I created a line graph showing average hourly humidity each month, but first I 
#had to calculate average hourly humidity, which I do by first resampling (aggregating)
#every hour and taking the mean of Humidity. This creates a series with hourly humidity means. I 
#then convert this series to a dataframe and add two columns, one indicating the month and one 
#indicating the hour of day. I then create a line plot with the hour of day on the x-axis
#and humidity on the y-axis and with one line for each month.
hourly_humidity = weather_all.Humidity.resample('H').mean()
hourly_humidity = pd.DataFrame(hourly_humidity)
hourly_humidity['month'] = hourly_humidity.index.month
hourly_humidity['hour'] = hourly_humidity.index.hour

sns.lineplot(x="hour", y="Humidity", hue="month", data=hourly_humidity, ci=None)

#The graph shows two months (months 11 and 12) with much lower humidity, followed
#by months 1 and 2. This represents the winter months in San Diego. These four 
#months (and in particular months 11 and 12, and to some extent 1) also have different 
#points when humidity is the lowest compared to the other months. I have isolated these
#months in the graph below.
sns.lineplot(x="hour", y="Humidity", hue="month", data=hourly_humidity[hourly_humidity.month.isin([11,12,1,2])], ci=None)

#If humidity is included in the model, season and time of the day may be 
#omitted variables (the same is true for the other weather variables). Based 
#on this I will include month fixed effects. Further, the data is later aggregate 
#into daily values (e.g., morning average humidity, open hour average humidity, etc. 
#for each day) and differences in hourly fluctuations may as such be less of a problem. 
#In addition to including fixed effects, it is possible that humidity (and other weather
#measures) affect decisions about attending the aquarium differently depending on the month
#or in a non-linear fashion, e.g., a 10-point change in humidity within a comfortable range
#might be different than a 10-point change when the humidity is high. I will explore the effect 
#of this more later when looking at scatter plots with humidity and attendance.


#### Missing Rows ####
#Typically to find missing rows in a time series, you can create a datetime index with the expected 
#frequency and then compare this to your index to find datetime index rows that are missing in your 
#data. However, ObservationTimeUts is not measured in exact intervals (but roughly every 5 minutes) 
#so this method does not work (at least I do not think it does). Instead I simply created a new 
#column that contains the difference between the index value of the row and the index value of the 
#row above. I then used boolean indexing to return all rows with more than 360 seconds (6 minutes) 
#between two index values. Notes:
#1) .index returns a DatetimeIndex and to shift these values I first convert the DatetimeIndex to a series 
#with a datetime64 data type.
#2) To compare TimeDiff, which is a timedelta64 data type, to an integer I extract the number of seconds in the
#timedelta using .dt.second.

weather_all["TimeDiff"]=weather_all.index.to_series()-weather_all.index.to_series().shift(1)
weather_all.TimeDiff[weather_all.TimeDiff.dt.seconds>60*6]

#This returns a relatively large number of rows and it is somewhat difficult to examine this data further. 
#I instead use resample to create counts for each day. This counts the number of values for each day in each 
#column. To get a count for each day, I use mean(axis=1) to average these counts across the columns (to get 
#one average count per row). I then use boolean indexing to only show days where there are fewer than 100 observations.
 
dayss = weather_all.resample('D').count().mean(axis=1)
dayss[dayss<100]

#There are many days that are missing some measurements, which might be OK when we are later aggregating.  I assume the 
#weather station simply does not work all the time.  Assuming that this is fairly random, then this should not bias
#the results (this might actually not be the case if, for example, the weather station tends to fail under certain 
#conditions, e.g., when it rains a lot or when it is really windy). Given the assumption that they are missing at random,
#ignoring (i.e., some aggregate measures will be based on fewer observations than others) these missing values 
#should as such not bias the results and instead simply add noise to the data (which will make the results weaker). So for
#days when I have some values, I assume that aggregation will still work and simply make the measure less precise for these 
#random days.  Note that the earliest data (2014-01-01:2014-01-04) has exactly  24 measures per day (and so do some 
#other dates), it appears that this period for some reason only took measurements once an hour. However, since I 
#will later aggregate this data based on various daily #time periods (e.g., morning and and open hours), I do not 
#expect that this will have a significant negative effect on my analysis.
#However, the period between 2015-10-20 15:20:00 and 2015-10-28 13:20:00 has no data at all. I will fill these days
#with values by simply duplicating data from 10 days earlier.

#First create a new dataframe with data from 10 days before the missing records and reset the index (make the
#index a columns again). Add 10 days to the dates, which make the dates the same as the dates that the missing
#data would have had.  Then stack (vertically concatenate) these observations to the original data. 
weather_all_subset = weather_all['2015-10-10 15:20:00':'2015-10-18 13:20:00'].reset_index()
weather_all_subset.ObservationTimeUtc = weather_all_subset.ObservationTimeUtc + pd.DateOffset(days=10)
weather_all_subset.set_index('ObservationTimeUtc', inplace=True)
weather_all = pd.concat([weather_all, weather_all_subset], join='inner')


########### Resample ###########
#Resample the data to create daily variables that can then later be lagged.
#Store the daily values in different data frames to simplify processing.


#weather_all.to_csv('C:/Users/jperols/Documents/A Teaching Content/Assignment - Python/NEW/Birch/Birch_Weather_Cleaned.csv')

#### Morning Weather ####
morning_averages = weather_all.between_time('8:00', '11:00')[['Humidity', 'TemperatureC', 'RainMillimetersRatePerHour', 'WindSpeedKph']].resample('D').mean()
morning_averages.columns = morning_averages.columns + "_morning_means"

#### Open Hours Weather ####
open_averages = weather_all.between_time('9:00', '17:00')[['Humidity', 'TemperatureC', 'RainMillimetersRatePerHour', 'WindSpeedKph']].resample('D').agg(['mean', 'max'])

#Add open to all columns headers and also combine level 0 and level 1 of multi index values
open_averages.columns = open_averages.columns.get_level_values(0) + "_open_" + open_averages.columns.get_level_values(1)

#### Prior Evening Averages ####
evening_averages = weather_all.between_time('18:00', '23:00')[['Humidity', 'TemperatureC', 'RainMillimetersRatePerHour', 'WindSpeedKph']].resample('D').mean()
prior_evening_averages = evening_averages.shift(1, freq='D')
prior_evening_averages.columns = prior_evening_averages.columns + "_prior_evening_means"

#Merge all weather dataframes into one dataframe
all_weather = pd.merge(pd.merge(morning_averages, open_averages, left_index=True, right_index=True, how='inner'), prior_evening_averages, left_index=True, right_index=True, how='left')

#### Check for Missing Values and Replace With Prior Day Values ####
all_weather[all_weather.isnull().sum(axis=1)>0].isnull().count()
all_weather[all_weather.isnull().sum(axis=1)>0].isnull().sum()

#21 rows have one or more missing null values and humidity has the most null values (as expected
#given what we noticed earlier). 
#Let's check the raw data for a few of these:
all_weather[all_weather.isnull().sum(axis=1)>0]
weather_all['2014-04-09 05:46':'2014-04-10 11:15']

#Looks like there are some missing values in the data (as expected given our earlier findings 
#when looking at missing rows). As stated previosly, I will fill these values with the closest 
#values with the assumption that these are values that are missing at random and with the 
#assumption that the average of the closest prior and subsequent values are the best estimates 
#of the missing value, e.g., if I do not know the temparature on the 22nd, but it was 70 degrees
#on the 21st and 74 degrees on the 23rd then I am going to assume that it was 72 degrees on the 22nd.

all_weather.interpolate(method ='linear', axis=0, limit = 5, limit_direction ='both', inplace=True) 
all_weather[all_weather.isnull().sum(axis=1)>0].isnull().sum()

#The defaults for method and axis are linear and 0, respectively so I could have skipped these.
#Linear calcualtes the average of the values before and after. Axis=0 interpolates column by
#column. The argument limit = x puts a limited on how many consecutive NaN are filled, so if we
#have missing data on both the 22nd and the 23rd then both will be filled when using limit 2, but 
#if data were also missing on the 24th then they would not be filled. If two consecutive values 
#are missing then they will be evenly spaced between the two values around them. I set the limit
#argument to a fairly high number as this will only affect the filling of humidity values that were
#missing for about 10 days in december of 2016 (I additionaly do not expect the filling of humidity 
#for the winter to be a major concern as the humidity is fairly consistenly low in this month).  That 
#said, if it turns out that humidity is important in the regression then I need to confirm that the
#results are not sensitive to this interpolation. Limit_direction is set to 'both' to allow first and 
#last values to be filled.


#Change the weather index to contain dates without time to work easier with attendance dates
all_weather = all_weather.reset_index()
all_weather['Date'] = pd.to_datetime(all_weather.ObservationTimeUtc).dt.date
all_weather = all_weather.set_index('Date')
all_weather.drop('ObservationTimeUtc', axis=1, inplace = True)

#Do not keep any weather data prior to the earliest attendance date
all_weather = all_weather[attendance_data.index.min():]

#Now merge weather and attendance
weather_and_attendance = pd.merge(all_weather, attendance_data, left_index=True, right_index=True, how='inner')
#We lost 15 weather rows in the merge because attendance data is missing for Christmas Day, New Year's Day,
#and Thanksgiving Day.
###################################################################
####################### Feature Engineering #######################
###################################################################

#Before rolling, make sure that the index is sorted correctly:
weather_and_attendance.sort_index(inplace=True)

#Create prior 7-day averages using rolling (I am doing 7 days so that I get 
#the same number of weekends and business days in all windows). 
weekly = weather_and_attendance.rolling('7d', closed='left', min_periods=4).mean()

#Rolling aggregates rows based on a moving window. I am using 7 days ('7d') to
#take all records within the prior seven day period and group them together.
#I then use the aggregate function mean to take the average of these records.
#This is done for each row in the dataset, so the 7-day period is a moving 
#window.  The argument closed specifes which end points should be included. 
#Closed='left' is used to make the window include the left endpoint and exclude the
#current value (so with 7d it takes all prior values including the value 7 days
#ago but not the value of the current observation). With closed='both' the current 
#value and the earliest value matching the first date of the window are included. With
#closed = 'right', the right value (the current value) is included but not the left
#value (the earliest datetime in the window). The min_periods argument specifies
#the minimum number of observations in window required to return an aggregate value 
#(otherwise result is NA).

#Get weekly attendance a year ago
weekly_a_year_ago_centered = weekly.shift(1, pd.Timedelta('360d')) 
#364 takes us back to the same day of the week a year ago and 360 (364-4) gives 
#us the weekly average with the same day a year ago in the middle

#Note that the rolling procedure above creates rolling averages for each date except
#for the first 4 rows. The weekly a year ago then further shift the data by 360 days 
#so we now are missing 360 rows in the beginning of the sample period, but we have 
#instead 360 rows data added at the end that are outside the sample period.
#If we do a left join from the original weather_and_attendance data then we will
#instead end up with null values for the missing rows and the addred rows at the
#end will be removed - which is what we want. So we need to make sure to remember this.

#Create prior 28-day averages using rolling (I am doing 28 days so that we get the 
#same number of weekends and business days in all windows)
monthly = weather_and_attendance.rolling('28d', closed='left', min_periods=15).mean()
monthly_a_year_ago_centered = monthly.shift(1, pd.Timedelta('350d')) 
#364 takes us back to the same day of the week a year ago, 350 (364-14) gives us the monthly 
#average with the day of the week a year ago in the middle 
#Here is alternative code: monthly_a_year_ago_centered = monthly.shift(1, pd.DateOffset(days=350)) 

#Create indicators showing if a date is a holiday, weekend or business day,
#and the day of ther week (as a number - if I use this I need to create dummy variables)
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=weather_and_attendance.index.min(), end=weather_and_attendance.index.max())
weather_and_attendance['Holiday'] = weather_and_attendance.index.isin(holidays)
weather_and_attendance['Weekend'] = np.where(weather_and_attendance.index.weekday < 5, 0, 1)
weather_and_attendance['SchoolsOut'] = np.where((weather_and_attendance.Holiday + weather_and_attendance.Weekend+ weather_and_attendance.index.month.isin([7,8]) + ((weather_and_attendance.index.month==7) & (weather_and_attendance.index.day > 15)) + ((weather_and_attendance.index.month==12) & (weather_and_attendance.index.day > 22)))==0,0,1)
weather_and_attendance['SummerBreak'] = np.where((weather_and_attendance.index.month.isin([7,8]) + ((weather_and_attendance.index.month==7) & (weather_and_attendance.index.day > 15)))==0,0,1)
#This is a fairly rough estimate of when school is out in San Diego (it captures weekends, federal holidays, summer break, 
#and winter break with the assumption that they are always in same time period.

weather_and_attendance['Dayofweek'] = weather_and_attendance.index.weekday
#Find the average of the two most recent weekend or holiday days
two_weekend_days = weather_and_attendance[(weather_and_attendance.Weekend==1) | (weather_and_attendance.Holiday==1)].rolling(2, min_periods=2).mean().shift(1)

#Find the average of the five most recent business days
five_business_days = weather_and_attendance[(weather_and_attendance.Weekend==0) & (weather_and_attendance.Holiday==0)].rolling(5, min_periods=5).mean().shift(1)

#Since only weekend observations have values for two_weekend_days and only business day observations have 
#values for five_business_days I am stacking these so that all days have values
prior_weekend_business_day = pd.concat([two_weekend_days, five_business_days])
prior_weekend_business_day.sort_index(inplace=True)

#Find attendance of the same day a week ago
attendance_one_week_ago = weather_and_attendance.shift(1, pd.Timedelta('7d'))

#Create a list of all the dataframes that will be merged and then update the column names 
#to indicate which df they are part of.
merge_list = [weekly, monthly, weekly_a_year_ago_centered, monthly_a_year_ago_centered, prior_weekend_business_day, attendance_one_week_ago]
weekly.name = 'weekly'
monthly.name = 'monthly'
weekly_a_year_ago_centered.name = 'weekly_a_year_ago_centered'
monthly_a_year_ago_centered.name = 'monthly_a_year_ago_centered'
prior_weekend_business_day.name = 'prior_same_day_type'
attendance_one_week_ago.name = 'attendance_one_week_ago'

for data_frame in merge_list:
    data_frame.columns = data_frame.name + "_" + data_frame.columns 

#I am using join rather than merge here as join allows me to join the left
#table with a list of dataframes. I am using join instead of concat (which can
#also be usef for joining lists of dataframes) to be able to specify a left join
#rather than simply outer or inner. Note that I want outer so that I get all the
#original data and then the additional data when it exists (when it does not these
#columns will have null values). I want this because I do not know if I will be 
#using all the variables that I created and I am therefore better off keeping
#as many observations as possible. This however also means that I need to remove
#null values before I run regression analyses (it will create an error otherwise).

all_data = weather_and_attendance.join(merge_list, how='left')

#There are a bunch of Happenings and they are likely to affect attendance on that day.
#However, many of them are one off events (or only happens a few times) and a model that
#learns to adjust for a specific event will not be useful if that even does not happen
#again (instead it will likely result in over fitting). Below I clean up some spelling, 
#like capital vs. not in one of the words for the same happening, and then only keep
#hapennings that occur at least ten times. I convert all the other happenings to simply 
#say "Special Event".
all_data.Happenings = all_data.Happenings.str.upper().str.strip().str.slice(0,10)
CommonHappenings=all_data.groupby('Happenings')['Happenings'].count().sort_values()
CommonHappenings=CommonHappenings[CommonHappenings>=10]
all_data.Happenings=np.where(all_data.Happenings.isin(CommonHappenings.index.tolist()),all_data.Happenings,np.where(all_data.Happenings.notnull(),'SpecialEvent','NoEvent'))
all_data['AnyHappenings']=np.where(all_data.Happenings=='NoEvent',0,1)

#Then to see if the happenings tend to change Attendance, I create
#box-plots and also group (by using different hues) the data based on
#weekend (otherwise an event that is always on weekends may appear to have 
#higher attendance than an average day, but in reality it does not have
#higher average attedance than other weekend days

plt.figure()  
ax=sns.catplot(x='Happenings', y='Attendance', hue="Weekend", kind="box", dodge=False, data=all_data)
ax.set_xticklabels(rotation=30)
#Comparing the No Event boxes (on the left side of the graph) and comparing
#oranage to orange and blue to blue, it appears that BOFA IN PR might be 
#slighly higher than the others, but probably not significantly different.
#(Note that the special event attendance has been subtracted from attendance,
#so the only reason a special event would affect attendance is if other paying
#guests are more or less likely to go to attend because of the event. My
#guess would have been that people would be less likely to go if there was
#a special event, for example parking might be more difficult, it might be 
#more crowded, etc.  But there does not appear to be a relation between
#special events and attendance (though when controlling for other factors
#it could become significant). I will go ahead an keep the AnyHappenings
#variable, create a BOFA dummy, and drop the Happenings categorical 
#variable (also I did actually use the original happenings variable
#in regression models - I was curious - but as concluded here, they did
#not appear to help the model.
all_data['BOFA_Event']=np.where(all_data.Happenings=='BOFA IN PR',1,0)
all_data.drop('Happenings', axis=1, inplace = True)



pd.options.display.max_columns = all_data.shape[1] ##Show all columns when printing 'describe'
print(all_data.describe())





new_final_data= all_data[['Attendance','Humidity_open_mean','TemperatureC_open_mean','RainMillimetersRatePerHour_open_mean','WindSpeedKph_open_mean','Education_Attendance','Special_Event_Attendance','Weekend','SchoolsOut','SummerBreak','Dayofweek','weekly_Humidity_open_mean','weekly_TemperatureC_open_mean','weekly_TemperatureC_open_max','weekly_Attendance','weekly_Education_Attendance','weekly_Special_Event_Attendance','monthly_Humidity_open_max','monthly_TemperatureC_open_max','monthly_Special_Event_Attendance','weekly_a_year_ago_centered_WindSpeedKph_morning_means','weekly_a_year_ago_centered_Attendance','weekly_a_year_ago_centered_Special_Event_Attendance','prior_same_day_type_Attendance','prior_same_day_type_Education_Attendance','prior_same_day_type_Weekend','attendance_one_week_ago_Education_Attendance','attendance_one_week_ago_Dayofweek','attendance_one_week_ago_Weekend','AnyHappenings','BOFA_Event']].copy()            



sns.distplot(new_final_data.Attendance, kde=False, fit=stats.norm)

new_final_data['AttendanceSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.AttendanceSQRT, kde=False, fit=stats.norm)

new_final_data['Attendance_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.Attendance_Log, kde=False, fit=stats.norm)
##KEEP LOG





sns.distplot(new_final_data.Humidity_open_mean, kde=False, fit=stats.norm)

new_final_data['Humidity_open_meanSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.Humidity_open_meanSQRT, kde=False, fit=stats.norm)

new_final_data['Humidity_open_mean_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.Humidity_open_mean_Log, kde=False, fit=stats.norm)
##KEEP LOG




sns.distplot(new_final_data.TemperatureC_open_mean, kde=False, fit=stats.norm)

new_final_data['TemperatureC_open_meanSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.TemperatureC_open_meanSQRT, kde=False, fit=stats.norm)

new_final_data['TemperatureC_open_mean_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.TemperatureC_open_mean_Log, kde=False, fit=stats.norm)
##KEEP Regular





sns.distplot(new_final_data.WindSpeedKph_open_mean, kde=False, fit=stats.norm)

new_final_data['WindSpeedKph_open_meanSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.WindSpeedKph_open_meanSQRT, kde=False, fit=stats.norm)

new_final_data['WindSpeedKph_open_mean_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.WindSpeedKph_open_mean_Log, kde=False, fit=stats.norm)
##KEEP REGULAR ONE
 







sns.distplot(new_final_data.Education_Attendance, kde=False, fit=stats.norm)

new_final_data['Education_AttendanceSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.Education_AttendanceSQRT, kde=False, fit=stats.norm)


new_final_data['Education_Attendance_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.Education_Attendance_Log, kde=False, fit=stats.norm)
##KEEP Regular



sns.distplot(new_final_data.Special_Event_Attendance, kde=False, fit=stats.norm)

new_final_data['Special_Event_AttendanceSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.Special_Event_AttendanceSQRT, kde=False, fit=stats.norm)

new_final_data['Special_Event_Attendance_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.Special_Event_Attendance_Log, kde=False, fit=stats.norm)
##KEEP LOG




sns.distplot(new_final_data.Special_Event_Attendance, kde=False, fit=stats.norm)

new_final_data['Special_Event_AttendanceSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.Special_Event_AttendanceSQRT, kde=False, fit=stats.norm)

new_final_data['Special_Event_Attendance_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.Special_Event_Attendance_Log, kde=False, fit=stats.norm)
##KEEP LOG




sns.distplot(new_final_data.SchoolsOut, kde=False, fit=stats.norm)

new_final_data['SchoolsOutSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.SchoolsOutSQRT, kde=False, fit=stats.norm)

new_final_data['SchoolsOut_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.SchoolsOut_Log, kde=False, fit=stats.norm)
##KEEP LOG


sns.distplot(new_final_data.Dayofweek, kde=False, fit=stats.norm)

new_final_data['DayofweekSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.DayofweekSQRT, kde=False, fit=stats.norm)

new_final_data['Dayofweek_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.Dayofweek_Log, kde=False, fit=stats.norm)
##KEEP LOG



sns.distplot(new_final_data.weekly_Attendance, kde=False, fit=stats.norm)

new_final_data['weekly_AttendanceSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.weekly_AttendanceSQRT, kde=False, fit=stats.norm)

new_final_data['weekly_Attendance_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.weekly_Attendance_Log, kde=False, fit=stats.norm)
##KEEP LOG





sns.distplot(new_final_data.weekly_Special_Event_Attendance, kde=False, fit=stats.norm)

new_final_data['weekly_Special_Event_AttendanceSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.weekly_Special_Event_AttendanceSQRT, kde=False, fit=stats.norm)

new_final_data['weekly_Special_Event_Attendance_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.weekly_Special_Event_Attendance_Log, kde=False, fit=stats.norm)
##KEEP SQRT







sns.distplot(new_final_data.prior_same_day_type_Attendance, kde=False, fit=stats.norm)

new_final_data['prior_same_day_type_AttendanceSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.prior_same_day_type_AttendanceSQRT, kde=False, fit=stats.norm)

new_final_data['prior_same_day_type_Attendance_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.prior_same_day_type_Attendance_Log, kde=False, fit=stats.norm)
##KEEP LOG






sns.distplot(new_final_data.attendance_one_week_ago_Dayofweek, kde=False, fit=stats.norm)

new_final_data['attendance_one_week_ago_DayofweekSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.attendance_one_week_ago_DayofweekSQRT, kde=False, fit=stats.norm)

new_final_data['attendance_one_week_ago_Dayofweek_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.attendance_one_week_ago_Dayofweek_Log, kde=False, fit=stats.norm)

##KEEP LOG






sns.distplot(new_final_data.monthly_TemperatureC_open_max, kde=False, fit=stats.norm)

new_final_data['monthly_TemperatureC_open_maxSQRT'] = (new_final_data.Attendance) ** 0.5

sns.distplot(new_final_data.monthly_TemperatureC_open_maxSQRT, kde=False, fit=stats.norm)

new_final_data['monthly_TemperatureC_open_max_Log'] = np.log(new_final_data.Attendance)
sns.distplot(new_final_data.monthly_TemperatureC_open_max_Log, kde=False, fit=stats.norm)
##KEEP LOG







# Scatter Plots created to determine if additional transformations are neccessary 

def create_scatter_plots(col, lowess_arg = True, order_arg=1): 
    plt.figure()
    sns.regplot(x=col, y='Attendance_Log', data=new_final_data, lowess=lowess_arg, order=order_arg, ci=None, truncate=True, line_kws={'color':'black'})
    
plot_cols = ['Attendance', 'Humidity_open_mean', 'TemperatureC_open_mean', 'RainMillimetersRatePerHour_open_mean', 'WindSpeedKph_open_mean', 'Education_Attendance', 'Special_Event_Attendance', 'Weekend', 'SchoolsOut', 'SummerBreak', 'Dayofweek', 'weekly_Humidity_open_mean', 'weekly_TemperatureC_open_mean', 'weekly_TemperatureC_open_max', 'weekly_Attendance', 'weekly_Education_Attendance', 'weekly_Special_Event_Attendance', 'monthly_Humidity_open_max', 'monthly_TemperatureC_open_max',
       'monthly_Special_Event_Attendance', 'weekly_a_year_ago_centered_WindSpeedKph_morning_means', 'weekly_a_year_ago_centered_Attendance', 'weekly_a_year_ago_centered_Special_Event_Attendance', 'prior_same_day_type_Attendance', 'prior_same_day_type_Education_Attendance', 'prior_same_day_type_Weekend', 'attendance_one_week_ago_Education_Attendance', 'attendance_one_week_ago_Dayofweek', 'attendance_one_week_ago_Weekend', 'AnyHappenings', 'BOFA_Event', 'AttendanceSQRT', 'Attendance_Log',
       'Humidity_open_meanSQRT', 'Humidity_open_mean_Log', 'TemperatureC_open_meanSQRT', 'TemperatureC_open_mean_Log', 'WindSpeedKph_open_meanSQRT', 'WindSpeedKph_open_mean_Log', 'Education_AttendanceSQRT', 'Education_Attendance_Log', 'Special_Event_AttendanceSQRT', 'Special_Event_Attendance_Log', 'SchoolsOutSQRT', 'SchoolsOut_Log', 'DayofweekSQRT', 'Dayofweek_Log', 'weekly_AttendanceSQRT', 'weekly_Attendance_Log', 'weekly_Special_Event_AttendanceSQRT', 'weekly_Special_Event_Attendance_Log',
       'prior_same_day_type_AttendanceSQRT', 'prior_same_day_type_Attendance_Log', 'attendance_one_week_ago_DayofweekSQRT', 'attendance_one_week_ago_Dayofweek_Log', 'monthly_TemperatureC_open_maxSQRT', 'monthly_TemperatureC_open_max_Log']

for col in plot_cols:
    create_scatter_plots(col)
plt.close()

# Closer look at individual variables to determine tranformations
create_scatter_plots(np.sqrt(new_final_data.weekly_a_year_ago_centered_Attendance))
create_scatter_plots(np.log(new_final_data.weekly_a_year_ago_centered_Attendance))
create_scatter_plots(new_final_data.weekly_a_year_ago_centered_Attendance)
# Logging this data flattens the line, resulting in a more positive linear relationship

create_scatter_plots(np.sqrt(new_final_data.WindSpeedKph_open_mean))
create_scatter_plots(np.log(new_final_data.WindSpeedKph_open_mean))
create_scatter_plots(new_final_data.WindSpeedKph_open_mean)
#Logging this data would reduce outlier distance


# amodel_result = sm.ols(formula='Attendance_Log ~ SummerBreak+ Dayofweek+ weekly_Humidity_open_mean+  weekly_Attendance+ attendance_one_week_ago_Education_Attendance+ Humidity_open_mean_Log+ TemperatureC_open_mean_Log+ WindSpeedKph_open_mean_Log+ Education_Attendance_Log+ Special_Event_Attendance_Log+ SchoolsOut_Log+ Dayofweek_Log+ weekly_Attendance_Log+ weekly_Special_Event_Attendance_Log+ prior_same_day_type_Attendance_Log+ monthly_TemperatureC_open_max_Log', data=new_final_data).fit()
# print(amodel_result.summary())

#################### RESIDUALS VS PREDICTED #####################
plt.figure()
plt.title("Residuals by Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.scatter(model_results.fittedvalues,model_results.resid, s=1, alpha=0.1)

plt.figure()
plt.title("Residuals by Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.scatter(model_results1.fittedvalues,model_results1.resid, s=1, alpha=0.1)

plt.figure()
plt.title("Residuals by Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.scatter(model_results2.fittedvalues,model_results2.resid, s=1, alpha=0.1)

plt.figure()
plt.title("Residuals by Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.scatter(model_results3.fittedvalues,model_results3.resid, s=1, alpha=0.1)

# plt.figure()
# plt.title("Residuals by Predicted")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals")
# plt.scatter(amodel_result.fittedvalues,amodel_result.resid, s=1, alpha=0.1)

plt.figure()
sns.distplot(model_results2.resid, kde=False, fit=stats.norm)
plt.figure()
sns.distplot(model_results3.resid, kde=False, fit=stats.norm)




model_results = sm.ols(formula='Attendance ~ Education_Attendance + Education_Attendance + Special_Event_Attendance + Weekend + SchoolsOut + SummerBreak + Dayofweek + weekly_Humidity_open_mean + weekly_TemperatureC_open_mean + weekly_TemperatureC_open_max + weekly_Attendance + weekly_Education_Attendance + weekly_Special_Event_Attendance + monthly_Humidity_open_max + monthly_TemperatureC_open_max + monthly_Special_Event_Attendance + weekly_a_year_ago_centered_WindSpeedKph_morning_means', data=new_final_data).fit()



model_results1 = sm.ols(formula='Attendance ~ Education_Attendance + Education_Attendance + Special_Event_Attendance + Weekend + SchoolsOut + SummerBreak + Dayofweek + weekly_Humidity_open_mean + weekly_TemperatureC_open_mean + weekly_TemperatureC_open_max + weekly_Attendance + weekly_Education_Attendance + weekly_Special_Event_Attendance + monthly_Humidity_open_max + monthly_TemperatureC_open_max + monthly_Special_Event_Attendance + weekly_a_year_ago_centered_WindSpeedKph_morning_means', data=new_final_data).fit()


#With transformations


model_results2 = sm.ols(formula='Attendance_Log ~ Education_Attendance + Special_Event_Attendance_Log + Weekend + SchoolsOut_Log + SummerBreak + Dayofweek_Log + weekly_Humidity_open_mean + weekly_TemperatureC_open_mean + weekly_TemperatureC_open_max + weekly_Attendance_Log + weekly_Education_Attendance + attendance_one_week_ago_Dayofweek_Log + TemperatureC_open_mean + monthly_Special_Event_Attendance + Humidity_open_mean_Log + monthly_TemperatureC_open_max_Log', data=new_final_data).fit()

print(model_results2.summary())





model_results3 = sm.ols(formula='Attendance_Log ~ Education_Attendance + Special_Event_Attendance_Log + Weekend + SchoolsOut_Log + SummerBreak + Dayofweek_Log + weekly_Humidity_open_mean + weekly_TemperatureC_open_mean' , data=new_final_data).fit()


print(model_results3.summary())















