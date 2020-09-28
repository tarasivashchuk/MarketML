import asyncio
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import aiohttp
from aiostream import stream
from pandas_datareader import DataReader
from yahoo_fin import stock_info as si

host = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
parameters = {
    "formatted": "true",
    "crumb": "swg7qs5y9UP",
    "lang": "en-US",
    "region": "US",
    "modules": "upgradeDowngradeHistory,recommendationTrend,financialData,earningsHistory,earningsTrend,industryTrend",
    "corsDomain": "finance.yahoo.com",
}
progress_bar = tqdm()
recommendations = []
tickers = []


async def download_ticker(session: aiohttp.ClientSession, ticker: str):
    global progress_bar
    global recommendations
    url = lhs_url + ticker + rhs_url
    async with session.get(url) as r:
        r = await r.json()
        try:
            result = r
            print(result)
            # recommendation = result['financialData']['recommendationMean']['fmt']
        except:
            recommendation = 6
        recommendations.append(recommendation)
        progress_bar.update()


def append_recommendations(r, rs):
    global progress_bar
    rs.append(r)
    progress_bar.update()
    return rs


async def main():
    global progress_bar
    global tickers
    tickers = si.tickers_sp500()
    tasks = []
    progress_bar = tqdm(total=len(tickers))
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=20, ssl=False)) as sess:
        for ticker in tickers:
            task = asyncio.create_task(download_ticker(sess, ticker))
            tasks.append(task)
        [await _ for _ in asyncio.as_completed(tasks)]


asyncio.run(main())
dataframe = pd.DataFrame(
    list(zip(tickers, recommendations)), columns=["Company", "Recommendations"]
)
dataframe.set_index("Company", inplace=True)
dataframe.to_csv("recommendations.csv")
print(dataframe.head())
