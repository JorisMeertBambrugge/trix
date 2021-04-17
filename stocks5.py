import pandas_datareader as pdr
import pandas as pd
import random
import numpy as np

from bokeh.plotting import figure
from bokeh.io import show,reset_output,output_file
from bokeh.models import Column,Row,ColumnDataSource,LinearAxis, Range1d, Band,Div,Quad
from bokeh.palettes import Spectral11
from bokeh.models import HoverTool

import urllib.request as urlRQ
from bs4 import BeautifulSoup as bs
from datetime import datetime

###############################################################################
#####################HELP FUNCTIONS############################################
###############################################################################

def crossing(a,b):
    """ returns the crossing indexes of two list-like series """
    if isinstance(a,(list,pd.Series)):
        crossing(np.array(a),b)
    if isinstance(b,(list,pd.Series)):
        crossing(a,np.array(b))
    
    crossingIndexes=np.where(np.diff(np.signbit(a-b)))[0]
    return crossingIndexes+1
# =============================================================================
# a = [-2, -1, 0, 1, 2,1,0,-1]
# b = pd.Series(data=[5,4,3,2,1,1,1,1])
# c=a-b
# print(c)
# crossingIndexes = crossing(a,b)
# print(crossingIndexes)
# =============================================================================

#a function that scrapes the dividend history from yahoo
def get_dividend(name,start='1/1/2013'):
    dividendList = []
    dividendDateList = []
    
    #calculate the time differences in seconds between 1-Jan-2000 and today
    #startTimeSeconds=1262300400 #1 Jan 2010 
    refDate=datetime.strptime(start, '%m/%d/%Y')
    startTimeSeconds=int((refDate-datetime(1970,1,1)).total_seconds())
    endTimeSeconds=int((datetime.today()-datetime(1970,1,1)).total_seconds())

    url=f"https://finance.yahoo.com/quote/{name}/history?period1={startTimeSeconds}&period2={endTimeSeconds}&interval=div%7Csplit&filter=div&frequency=1d"
    print(url)
    rows = bs(urlRQ.urlopen(url).read(),'lxml').findAll('table')[0].tbody.findAll('tr')

    for each_row in rows:
        divs = each_row.findAll('td')
        if divs[1].span.text  == 'Dividend': #use only the row from the table with dividend info
            dividendDateList.append(divs[0].span.text)
            dividendList.append(float(divs[1].strong.text.replace(',','')))
            
    dividendDateList=[datetime.strptime(i, '%b %d, %Y') for i in dividendDateList]#convert string list to datetime list

    return {'date':dividendDateList,'dividend':dividendList}
# =============================================================================
# dividendDict=get_dividend('TUB.BR',start)#scrape the dividend data from the yahoo website
# print(dividendDict)
# =============================================================================

def fill_missing_range(df, field, range_from, range_to, range_step=1, fill_with=0):
    """Function to transform a dataframe with missing rows because one column should be a stepwise range"""
    return df\
      .merge(how='right', on=field,
            right = pd.DataFrame({field:np.arange(range_from, range_to, range_step)}))\
      .sort_values(by=field).reset_index().fillna(fill_with).drop(['index'], axis=1)

def createDivPlot(dividendDict,data,start='1/1/2013'):
    dividendDF=pd.DataFrame(dividendDict)
    print(dividendDict)
    dividendDF['year']=dividendDF['date'].dt.year+1#create a year column
    dividendDF["yearDiv"] = dividendDF.groupby(["year"])["dividend"].transform(sum)#sum by year
    dividendDF['SP']=[data.loc[date]["Close"] for date in dividendDF['date']]
    dividendDF['divPercent']=[100*div/tub for div,tub in zip(dividendDF['dividend'],dividendDF["SP"])]
    dividendDF=dividendDF[['date','year','yearDiv','divPercent']]#keep only what matters
    dividendDF.columns=['date','year','dividend','divPercent']#rename
    dividendDF = dividendDF.drop_duplicates(subset=['year'], keep='first')#drop duplicates
    dividendDF=fill_missing_range(dividendDF, 'year', datetime.today().year, datetime.strptime(start, '%m/%d/%Y').year, range_step=-1, fill_with=0)#add a row with zero for each year where there was no dividend given
    dividendDF['date']=pd.to_datetime(dividendDF['year'].astype(str)+"-01-01",format="%Y-%m-%d",errors='raise')
    if dividendDict['dividend']!=[]:
        dividendSource = ColumnDataSource(data=dividendDict)
        dividendDFSource = ColumnDataSource(data=dividendDF)
        hover = HoverTool(tooltips=[("date","@date{%m/%d/%Y}"),("dividend","@dividend")],formatters={'@date': 'datetime'})
        tools=['pan','box_zoom','wheel_zoom',hover,'reset']
        divPlot=figure(width=1200,height=400,title='Historical dividend',x_axis_type='datetime',y_axis_label='Dividend',
                       y_range=(0,1.05*max(max(dividendDF['divPercent']),max(dividendDF['dividend']))),tools=tools)
        divPlot.scatter(x='date',y='dividend',line_color="red",fill_color='red',size=10,alpha=0.8,name='dividend',source=dividendSource,legend_label='Dividend coupons')
        divPlot.step(x='date',y='dividend',line_color="green",line_width=3,alpha=0.5,source=dividendDFSource,legend_label='Total dividend/year')  
        divPlot.step(x='date',y='divPercent',line_color="blue",line_width=3,alpha=0.5,source=dividendDFSource,legend_label='Dividend%')
        
        divPlot.legend.location = "top_left"
        return divPlot
    else:
        print(f'no dividend since {datetime.strptime(start, "%m/%d/%Y").year}!')
        divPlot=Div(text="no dividend since 2000! - Yahoo Finance")
        return divPlot

def createBoxPlot(Filter,yAxisFilter,source,title='Boxplot',width=1400):
    df=pd.DataFrame(source.data)
    # generate the category list
    catsColumn=list(source.data[Filter])
    cats =sorted(set(catsColumn))
    
    #get the x-axis for the dots and create jitter effect
    x_axis_value=[0.5]#bokeh plots categories on x-axis like this: 0.5,1.5,2.5,..
    for x in range (1,len(cats)):
        x_axis_value.append(x_axis_value[-1]+1)#make a list of the different category x-axis values
    x_axis=[]
    for x in catsColumn:
        index=cats.index(x)
        x_axis.append(x_axis_value[index]+random.uniform(-0.3,0.3))#make a jitter around the x-axis value of the catergory for each datapoint
    source.add(x_axis,'categorical_x_axis_value')#add a column to the datasource with the Jitter values 
    
    # find the quartiles and IQR for each category
    groups = df.groupby(Filter)
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    
    TOOLS="pan,wheel_zoom,lasso_select,reset,save"
    p = figure(tools=TOOLS, title=title, x_range=cats,width=width)
    
    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upperStem = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,yAxisFilter]),upper[yAxisFilter])]
    lowerStem = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,yAxisFilter]),lower[yAxisFilter])]
    
    # stems
    p.segment(cats, upperStem, cats, q3[yAxisFilter], line_color="black")
    p.segment(cats, lowerStem, cats, q1[yAxisFilter], line_color="black")
    
    #create the boxes boxes
    def createColorList(number=11):#create a color list for each category
        colorList=[]   
        for x in range(0,number):
            colorList.append(Spectral11[x%11])
            
        return colorList
    colorList=createColorList(number=len(cats))
    
    p.vbar(x=cats, width=0.7, bottom=q2[yAxisFilter], top=q3[yAxisFilter], line_color="black",color=colorList)
    p.vbar(cats, 0.7, q1[yAxisFilter], q2[yAxisFilter], line_color="black",color=colorList)
    
    #add data points
    #p.circle(source=source,x=Filter, y=yAxisFilter,size=5,color='black',alpha=0.3)
    p.circle(source=source,x='categorical_x_axis_value', y=yAxisFilter,size=5,line_color='black',fill_alpha=0)#with Jitter and via source
    
    # whiskers (almost-0 height rects simpler than segments)
    whiskerHeight=(max(qmax[yAxisFilter])-min(qmin[yAxisFilter]))/1000
    p.rect(x=cats, y=lowerStem, width=0.2, height=whiskerHeight, line_color="black",fill_color="black")
    p.rect(x=cats, y=upperStem, width=0.2, height=whiskerHeight, line_color="black",fill_color="black")
    
    return p

#get the index of the next value in a list equal to the value at startIndex
def findNextIndexOf(startIndex,timeList):
    while True:
        for time in timeList[startIndex+1:]:
            if timeList[startIndex]==time:
                index=timeList[startIndex+1:].index(time)
                #print(str(timeList[startIndex])+' is found at index '+str(index+startIndex+1))
                return index+startIndex+1
                break
        break
    return False

#find the previous index in timeList with value equal to the startIndex. StartIndex needs to be larger than 0 
def findPreviousIndexOf(startIndex,timeList):
    if startIndex<len(timeList):
        for i in range(startIndex-1,-1,-1):
            if timeList[i]==timeList[startIndex]:
                break
            else:
                i=False
    else:
        i=False
    return(i)

#calculate the relative difference compared to the week average for a chonological list of values and a list of weeksdays with monday=0,tuesday=1,...
def getTimeVariability(timeList,values,intervalSize):
    averageList=[1]*(intervalSize-1)#skip the first
    
    for i in range(intervalSize,len(values)-intervalSize+2):
        beforeStartIndex=findPreviousIndexOf(i,timeList)
        afterEndIndex=findNextIndexOf(i,timeList)
        
        intervalListBefore=values[beforeStartIndex:i-1]
        intervalListAfter=values[i:afterEndIndex-1]
        
        intervalListBefore=values[i-intervalSize:i-1]
        
        avg=(sum(intervalListBefore)+sum(intervalListAfter))/(len(intervalListAfter)+len(intervalListBefore))
        #print('the value at index '+str(i-1)+' is '+str(values[i-1]))
        averageList.append(values[i-1]/avg)
        
    for i in range(len(values)-intervalSize+1,len(values)):#skipt the last
        averageList.append(1)
        
    return averageList

values=[3,4,5,6,7,8,9,10,9,8]
timeList=[3,4,0,1,2,3,4,0,1,2]
intervalSize=5
#print(getTimeVariability(timeList,values,intervalSize))

#calculate the relative difference compared to the week average for a chonological list of values and a list of weeksdays with monday=0,tuesday=1,...
def getAverage(valuesList,sizeNumber):
    averageList=[valuesList[0]] 
    for i in range(1,len(valuesList)):
        sizeList=valuesList[max(0,i-sizeNumber):min(len(valuesList),i)]
        averageList.append(sum(sizeList)/max(len(sizeList),1))
        
    return averageList

#apply a certain buy/sell strategy on historical data
# meDirect transactie tarieven per beurs https://www.medirect.be/nl-be/tarieven-en-kosten/transactietarieven 
def strategy(buySellList,sharePrice,trafficTax=0.0035,tafficCost=0.001):
    """
    Function that returns the % of profit fro a given buy-sell strategy
    
    buySellList = list of tuple indexes with (buyIndex,sellIndex)
    sharePrice = list of sharePrice over time
    """
    profitPercent=100
    buyValue=1

    for buySell in buySellList:
        buyValue=round(sharePrice[buySell[0]],2)*(1+trafficTax)*(1+tafficCost)
        sellValue=round(sharePrice[buySell[1]],2)*(1-trafficTax)*(1-tafficCost)
        profitPercent=profitPercent*sellValue/buyValue
        
    return round(profitPercent-100,2)

def findBuy(i,trix,EMA_on_Trix,EMA,sharePrice):
    if trix[i]>EMA_on_Trix[i]:      
        for index in range(i,len(sharePrice)):
            if trix[index]>0:
                #print("trix>0")
                #print(trix[index],EMA_on_Trix[index],EMA[index],sharePrice[index])
                return None
            elif trix[index]<0 and sharePrice[index]>EMA[index]:
                #print(trix[index],EMA_on_Trix[index],EMA[index],sharePrice[index])
                return index
        return None
    else:
        return None

def findSell(i,trix,EMA_on_Trix):
    for j in range(i,len(trix)):
        if trix[j]<0 and trix[j]<EMA_on_Trix[j]:
            return j
    return None

###############################################################################
#####################BODY OF THE CODE##########################################
###############################################################################

def createView(symbol,start=None,getStrategyYield=False,EMA_days=200,Trix_EMA_days=39,EMA_on_Trix_days=9):
    
    data=pdr.get_data_yahoo(symbol,start=start)
    #print(data.keys())
    
    #get the x-axis values: datetime
    timeList=data.index.values
    data['date']=timeList

    data[f"EMA_{EMA_days}"]=data['Close'].ewm(span=EMA_days, adjust=False).mean()#add an exponential moving average
    
    #calculate the Tripe Exponential Average, Trix see https://www.investopedia.com/terms/t/trix.asp
    data['ema1']=data['Close'].ewm(span=Trix_EMA_days, adjust=False).mean()
    data['ema2']=data['ema1'].ewm(span=Trix_EMA_days, adjust=False).mean()
    data['ema3']=data['ema2'].ewm(span=Trix_EMA_days, adjust=False).mean()
    data['ema3_yesterday']=data['ema3'].shift(1)
    data['trix']=100*(data['ema3']-data['ema3_yesterday'])/data['ema3_yesterday']
    #data['trix']=3*data['ema1']-3*data['ema2']+data['ema3']#calculate the trix, see https://en.wikipedia.org/wiki/Triple_exponential_moving_average
    data['EMA_on_Trix']=data['trix'].ewm(span=EMA_on_Trix_days, adjust=False).mean()
    data['zero']=0
        
    crossIndexes=crossing(data['trix'],data['EMA_on_Trix'])#get the indexes when the trix and the ema(trix) cross
    crossIndexes=[i for i in crossIndexes if i>EMA_days]#remove the indexes with data before the full EMA can be taken
    #posCrossing=[i for i in crossIndexes if data['trix'][i]>data['EMA_on_Trix'][i] and data['trix'][i]<0 and data['Close'][i]>data[f"EMA_{EMA_days}"][i] ]

    buySellList=[]
    for i in crossIndexes[1:]:
        #print("cross at ",timeList[i])
        buy=findBuy(i,data['trix'],data['EMA_on_Trix'],data[f"EMA_{EMA_days}"],data['Close'])
        #print("buy",buy)
        if buy != None:    
            sell=findSell(buy,data['trix'],data['EMA_on_Trix'])
            #print("sell",sell)
            if sell != None:
                buySellList.append((buy,sell))
            else:
                buySellList.append((buy,-1))
    print(buySellList)
    buySellList=list(dict.fromkeys(buySellList))
    
    trixResult=strategy(buySellList,data['Close'],trafficTax=0.0035,tafficCost=0.001)
    buyHoldResult=strategy([(EMA_days,-1)],data['Close'],trafficTax=0.0035,tafficCost=0.001)
    resultDiv=Div(text=f"""
    Excluding dividends, these strategies would have resulted in a yield of:<br>
    Trix: {trixResult}%.<br>
    Buy and hold: {buyHoldResult}%.<br>
    <br>
    For a tax rate of 0.35% per transaction and a broker fee for 0.1% per transaction.
    """)

    ################### PLOT THE STOCK PRICE WITH BUY AND SELL SIGNALS#########
    yRangeMax=1.05*max(data['Close'])
    stock_value=figure(height=350,width=1200,x_axis_type='datetime',title =f"{symbol} value (source=Yahoo finance)",
                       y_range=[0.95*min(data['Close']),yRangeMax])    
    stockSource=ColumnDataSource(data)
    stock_value.line(source=stockSource,x='date',y='Close',color='black')#q line with the stock price
    stock_value.line(source=stockSource,x='date',y=f"EMA_{EMA_days}",color='blue',legend_label=f"{EMA_days} days Exponential Moving Average")#200 days average
    
    for buySell in buySellList:
        stock_value.line(x=[timeList[buySell[0]],timeList[buySell[0]]],y=[0,yRangeMax],color='green')
        if buySell[1] !=-1:
            stock_value.line(x=[timeList[buySell[1]],timeList[buySell[1]]],y=[0,yRangeMax],color='red')
        band = Quad(left=timeList[buySell[0]], right=timeList[buySell[1]], top=yRangeMax, bottom=0, fill_color="green",fill_alpha=0.1,line_width=0)
        stock_value.add_glyph(band)
 
    stock_value.legend.location = "bottom_left"
    stock_value.legend.click_policy="hide"

    ################### A PLOT WITH THE TRIX AND ITS SIGNAL###################
    signalPlot=figure(height=300,width=1200,x_axis_type='datetime',tools=['pan','box_zoom','wheel_zoom','reset'],y_range=(-0.3,0.4),x_range=stock_value.x_range,
                      title =f"{EMA_on_Trix_days} days EMA on Trix with {Trix_EMA_days} days")
    signalPlot.line(timeList,data['trix'],color='blue',legend_label=f"{Trix_EMA_days} days Trix")#signal
    signalPlot.line(timeList,data['EMA_on_Trix'],color='violet',legend_label=f"{EMA_on_Trix_days} days EMA on Trix")#signal
    signalPlot.line(x=[timeList[0],timeList[-1]],y=[0,0],color='black',line_dash='dashed')#signal

    for buySell in buySellList:
        signalPlot.line(x=[timeList[buySell[0]],timeList[buySell[0]]],y=[-1,1],color='green')
        if buySell[1] !=-1:
            signalPlot.line(x=[timeList[buySell[1]],timeList[buySell[1]]],y=[-1,1],color='red')
        band = Quad(left=timeList[buySell[0]], right=timeList[buySell[1]], top=1, bottom=-1, fill_color="green",fill_alpha=0.1,line_width=0)
        stock_value.add_glyph(band)    

    ############################ trading volume versus time###################
    stock_volume=figure(height=300,width=1200,x_axis_type='datetime',x_range=stock_value.x_range,
                        title =f"{symbol} trading volume (source=Yahoo finance)")
    stock_volume.vbar(x=timeList, top=data['Volume'], bottom=0, width=50000000, fill_color="#b3de69")

    #######################DIVIDEND EVENTS#####################################
    dividendDict=get_dividend(symbol,start=start)#scrape the dividend data from the yahoo website
    if dividendDict['date']==[]:
        dividendPlot=Div(text=f'<br>In this period, {symbol} did not pay any dividend.<br><br>',width=1200)
    else:
        dividendPlot=createDivPlot(dividendDict,data,start=start)
    
    ############## fluctuation depending on day of the week####################                 
    dates = pd.DatetimeIndex(timeList) #convert to datetime format
    weekdays = dates.weekday.values[:-365]#get the weekdays (0=monday, 1=tuesday,...)
    values=list(data['Open'])[:-365]#get the values in a list
    relToWeekAvg=getTimeVariability(timeList=list(weekdays),values=values,intervalSize=5)
    weekdaysStrings=[]
    for i in weekdays:
        if i==0:
            weekdaysStrings.append('1_Monday')
        elif i==1:
            weekdaysStrings.append('2_Tuesday')
        elif i==2:
            weekdaysStrings.append('3_Wednesday')
        elif i==3:
            weekdaysStrings.append('4_Thursday')
        elif i==4:
            weekdaysStrings.append('5_Friday')
        elif i==5:
            weekdaysStrings.append('6_Saturday')
        elif i==6:
            weekdaysStrings.append('7_Sunday')     
    sourceDays=ColumnDataSource({'ratio to week average':relToWeekAvg,'day of the week':weekdaysStrings})
    weekdayBoxPlot=createBoxPlot(Filter='day of the week',yAxisFilter='ratio to week average',source=sourceDays,title='Variability depending on the day of the week',width=1200)
    
    ################# fluctuation depending on month of the year ##############                
    months = dates.month.values#get the weekdays (0=monday, 1=tuesday,...)
    values=list(data['Open'])#get the values in a list
    relToYearAvg=getTimeVariability(timeList=list(months),values=values,intervalSize=12)
    monthStrings=[]
    for i in months:
        if i==1:
            monthStrings.append('01_Jan')
        elif i==2:
            monthStrings.append('02_Feb')
        elif i==3:
            monthStrings.append('03_Mar')
        elif i==4:
            monthStrings.append('04_Apr')
        elif i==5:
            monthStrings.append('05_May')
        elif i==6:
            monthStrings.append('06_Jun')
        elif i==7:
            monthStrings.append('07_Jul')
        elif i==8:
            monthStrings.append('08_Aug')
        elif i==9:
            monthStrings.append('09_Sep')
        elif i==10:
            monthStrings.append('10_Oct')
        elif i==11:
            monthStrings.append('11_Nov')
        elif i==12:
            monthStrings.append('12_Dec')
    
    sourceMonth=ColumnDataSource({'ratio to year average':relToYearAvg,'month':monthStrings})
    monthBoxPlot=createBoxPlot(Filter='month',yAxisFilter='ratio to year average',source=sourceMonth,title='Variability depending on the month of the year',width=1200)
    
# =============================================================================
#     ############## fluctuation depending on day of the month #################             
#     days = dates.day.values#get the weekdays (0=monday, 1=tuesday,...)
#     values=list(data['Open'])#get the values in a list
#     relToMonthAvg=getTimeVariability(timeList=list(days),values=values,intervalSize=27)#getWeekAverage(days,values,start=1)
#     daysStrings=[str(i) if i>9 else '0'+str(i) for i in days]
#     
#     sourceMonth=ColumnDataSource({'ratio to month average':relToMonthAvg,'day':daysStrings})
#     dayBoxPlot=createBoxPlot(Filter='day',yAxisFilter='ratio to month average',source=sourceMonth,title='Variability depending on the day of the month',width=1200)
#     
# =============================================================================
    ################## PUT ALL TOGHETER ######################################
    layout=Column(resultDiv,stock_value,signalPlot,stock_volume,dividendPlot,monthBoxPlot,weekdayBoxPlot)
    
    reset_output()
    output_file(r"C:/Users/joris/.spyder-py3/good code/scraping/"+symbol+".html")
    return layout
    
###############################################################################
###################DATABASE####################################################
###############################################################################

#weekly check for opportunity to sell/buy
stocksOfInterest=['ABI.BR']#,'AGS.BR','IMMO.BR','MIKO.BR','SMAR.BR','SOF.BR','UCB.BR','TUB.BR','AM.PA','HBH.DE','TL5.MC','DCC.L','MGGT.L','AV.L','PG','INTC','CSIQ','NPSNY',"WB",'VMW']

for symbol in stocksOfInterest:
    layout=createView(symbol, start='1/1/2018',getStrategyYield=True,EMA_days=55,Trix_EMA_days=39,EMA_on_Trix_days=9)
    show(layout)

# =============================================================================
# with open('profitStrategy.csv','a') as file:
#     file.write('longTermDays,mediumTermDays,ShortTermDays,symbol \n')
# =============================================================================
    
#print(sumGain)
#insider trading belgie: https://www.fsma.be/nl/transaction-search
    

    
