import asyncio
import sqlite3

from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import pandas as pd
import regex

from aiohttp import ClientSession, ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
from more_itertools import chunked
from tqdm.auto import tqdm

from voxpredict import data


session: ClientSession
request_counter = tqdm(desc="Request Counter")
good_request_counter = tqdm(desc="Good Request Counter")

pagination_params = {
    "s": "gs",
    "source": "search",
    "new_home_page": "control",
    "new_symbol_page": "new_sp",
    "page_type": "regular",
}

reference_regex = regex.compile(
    r"([qQ](?P<quarter>[1234]))-(?P<year>[0-9]{4}).*(?:earnings-call-transcript)"
)


async def fetch_html(url, headers, params=None):
    global session
    async with session.get(url, headers=headers, params=params) as response:
        request_counter.update()
        if response.status == 200:
            try:
                cookie = response.cookies.output().split(": ")
                cookie = {cookie[0]: cookie[1]}
                data.headers.update(cookie)
            except:
                pass
            good_request_counter.update()
            return await response.text()
        return None


def parse_transcript_urls(html):
    soup = BeautifulSoup(html, features="lxml")
    transcripts = [transcript.attrs["href"] for transcript in soup.find_all("a", href=True)]
    transcripts = [transcript.replace('\\"', "") for transcript in transcripts]
    return transcripts


async def get_transcript_urls(transcripts_url, headers):
    base_url = transcripts_url.replace("transcripts", "more_transcripts")
    pagination_transcripts = []
    i = 1
    while i < 7:
        params = {**pagination_params, **{"page": str(i)}}
        html = await fetch_html(base_url, headers, params)
        transcripts = await loop.run_in_executor(None, parse_transcript_urls, html)
        pagination_transcripts.append(transcripts)
        i += 1
        if transcripts is not None and len(transcripts) == 0:
            break
    pagination_transcripts = await loop.run_in_executor(
        None, chain.from_iterable, pagination_transcripts
    )
    return pagination_transcripts


async def scrape_company_transcript_urls(ticker):
    company_page_url = f"http://seekingalpha.com/symbol/{ticker}/earnings/transcripts"
    headers = {**data.headers, **{"referrer": company_page_url}}
    transcripts = await get_transcript_urls(company_page_url, headers)
    return ticker, list(set(transcripts))


def transcripts_2_df(ticker, transcripts):
    df = pd.DataFrame(columns=["ticker", "year", "quarter", "url"])
    for transcript in transcripts:
        reference = reference_regex.search(transcript)
        if reference is not None:
            transcript_data = {
                "ticker": ticker,
                "quarter": int(reference.group("quarter")),
                "year": int(reference.group("year")),
                "url": f"https://seekingalpha.com{transcript}",
            }
            df = df.append(pd.DataFrame(transcript_data, index=[0]), ignore_index=True)
    return df


async def scrape_transcript_urls():
    global session

    tickers = pd.read_csv("/Users/taras/OneDrive/Projects/VoxPredict Data/calls.csv", usecols=[1])
    tickers = [ticker.strip() for ticker in list(set(tickers["ticker"]))]
    db_connection = sqlite3.connect(data.earnings_db_path)
    existing_tickers = data.load_db_data(data.earnings_db_path, "transcript_final", ["ticker"])
    existing_tickers = [ticker[0].strip() for ticker in existing_tickers]
    tickers = [ticker for ticker in tickers if ticker not in existing_tickers]
    ticker_group = chunked(tickers, 100)

    for tickers in ticker_group:
        async with ClientSession(
            trust_env=True,
            connector=TCPConnector(ssl=False, limit_per_host=5),
            headers=data.headers,
            timeout=ClientTimeout(sock_read=900, sock_connect=900, connect=900, total=3600),
        ) as session:
            tasks = [loop.create_task(scrape_company_transcript_urls(ticker)) for ticker in tickers]
            data_tuples = [
                await data_tuple
                for data_tuple in tqdm(
                    asyncio.as_completed(tasks), total=len(tasks), desc=f"Tuples"
                )
            ]
            dfs = [
                await loop.run_in_executor(None, transcripts_2_df, data_tuple[0], data_tuple[1])
                for data_tuple in tqdm(data_tuples, total=len(data_tuples), desc="Transcripts 2 DF")
            ]
            df = pd.concat(dfs)
            df.to_sql(
                name="transcript_final", con=db_connection, index=False, if_exists="append",
            )
    db_connection.close()


if __name__ == "__main__":
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    with ThreadPoolExecutor() as threads:
        loop = asyncio.get_event_loop()
        loop.set_default_executor(threads)
        loop.run_until_complete(scrape_transcript_urls())
