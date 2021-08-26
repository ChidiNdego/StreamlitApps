import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# Simple Stock Price App

Shown are the stock closing price and volume of AMC!

""")

tickerSymbol = 'AMC'

tickerData = yf.Ticker(tickerSymbol)

tickerOf = tickerData.history(period='id', start ='2020-5-31', end='2021-8-24')

st.line_chart(tickerOf.Close)
st.line_chart(tickerOf.Volume)