#Author: From https://algotrading101.com/learn/backtrader-for-backtesting/
import backtrader as bt
import datetime

class PrintClose(bt.Strategy):

    def __init__(self):
        #Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
		#Print date and close
        print(f'{dt.isoformat()} {txt}') 
        
    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])

if __name__ == '__main__':

    #Instantiate Cerebro engine
    cerebro = bt.Cerebro()

    #Add data feed to Cerebro
    data = bt.feeds.YahooFinanceData(
        dataname = 'AAPL',
        name = 'AAPL',
        fromdate = datetime.datetime(2019,12,15),
        todate = datetime.datetime(2020,12,15),
        reverse = False
    )

    cerebro.adddata(data)

    #Add strategy to Cerebro
    cerebro.addstrategy(PrintClose)

    #Run Cerebro Engine
    cerebro.run()

#Next:
#How to run a backtest using Backtrader

#First, we will separate our strategy into its own file.

#We also have to separate our data into two parts. 
# This way, we can test our strategy on the first part, run some optimization, 
# and then see how it performs with our optimized parameters on the second set of data.

