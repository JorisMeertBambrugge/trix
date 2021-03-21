# trix
Stock autocorrelation analysis via Trix

This project is a data science and data visualization demo with python (bokeh), by Joris Meert.

This replicates the stock trading strategy as described by Paul Gins in the edition #3 of 2021 of "Beste Belegger" on page 71 (<a href="https://vfb.be/onlinemagazines" target="_blank">VFB<a>). 
The strategy consists on calculating a <a href="https://www.investopedia.com/terms/t/trix.asp" target="_blank">Triple Exponential Average</a> of 39 days, and a 9 days Exponential Average as signal on that Trix. In addition to the Trix and it's signal Mr. Gins applies either a 200 of 55 days Exponential Moving Average on the stock prices itself.'
The result is compared with a buy-and-hold strategy. 
The BUY strategy: Trix < 0 & Trix crosses EMA(Trix) upwards & SP > EMA(SP)
The SELL strategy: Trix < 0 & Trix < EMA(Trix)


To serve the website from your PC: 
1. download the files
2. Open a cmd in the folder where you saved the files
3. run the command: "bokeh serve stocks5_app --port 5010 --show --allow-websocket-origin=*"

Prior to this, you should have installed Python v3+, and the python module called Bokeh. Make sure the bokeh.exe and pyton.exe file directories are added to your PATH variables.

The raw stock price date is pullled from the Yahoo Finance API and the Dividend data is scraped from Yahoo Finance.
