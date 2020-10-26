import asyncio
import pickle
import sqlite3

import pandas as pd
import regex
import voxpredict.data.convert
from aiohttp import ClientSession, TCPConnector
from tqdm import tqdm

stream_url_regex = regex.compile(r"(http:.*.m3u8)(?:.*.<embed)")
"""Regex expression matching the audio stream URL."""

chunklist_url_regex = regex.compile(r".*(chunklist_.*.m3u8)")
"""Regex exoression matching the stream chunklist URL."""

chunks_regex = regex.compile(r"media.*.aac")
"""Regex expression matching the chunklist's filename's pattern"""

chunks_count_regex = regex.compile(r".*_([0-9]{1,3})\.aac")
"""Regex expression matching the ID of a chunk filename."""


def filter_scraped_data(data_row):
    global scraped_data

    ticker, _, _, quarter, year = data_row
    data_row = [ticker, quarter, year]
    if data_row not in scraped_data:
        return True
    else:
        return False


def filter_error_data(data_row):
    _, _, _, _, _, _, url, _ = data_row
    if url is None:
        return True
    else:
        return False


async def fetch_html(session: ClientSession, url: str) -> str:
    async with session.get(url) as response:
        return await response.text()


async def scrape(session: ClientSession, row_data):
    try:
        ticker, dt, url, quarter, year = row_data
    except ValueError:
        if row_data in scraped_data:
            return row_data
        ticker, quarter, year, dt, url, _, _, _ = row_data

    row_data = {
        "ticker": ticker,
        "quarter": quarter,
        "year": year,
        "datetime": dt,
        "url": url,
        "audio_chunklist_url": None,
        "audio_url": None,
        "audio_chunk_count": None,
    }
    start_url = f"https://wowza.earningscast.com{url}"
    loop = asyncio.get_running_loop()
    try:
        stream_html = await fetch_html(session, start_url)
        stream_url = await asyncio.get_running_loop().run_in_executor(
            None, stream_url_regex.findall, stream_html
        )
        if len(stream_url) == 0:
            return row_data
        stream_url = stream_url[0]
        base_url = await loop.run_in_executor(None, stream_url.rsplit, "/", 1)
        base_url = base_url[0]
        voxpredict.data.logger.debug(f"Stream URL: {stream_url} | Base URL: {base_url}")

        chunklist_html = await fetch_html(session, stream_url)
        chunklist_url = await loop.run_in_executor(
            None, chunklist_url_regex.findall, chunklist_html
        )
        if len(chunklist_url) == 0:
            voxpredict.data.logger.error(f"Error for {ticker} - {year} - {quarter}")
            return row_data
        chunklist_url = f"{base_url}/{chunklist_url[0]}"
        row_data.update({"audio_chunklist_url": chunklist_url})
        voxpredict.data.logger.debug(f"Chunklist URL: {chunklist_url}")

        chunks_html = await fetch_html(session, chunklist_url)
        chunks_urls = await loop.run_in_executor(None, chunks_regex.findall, chunks_html)
        if len(chunks_urls) == 0:
            return row_data
        final_chunk_url = chunks_urls[-1]
        voxpredict.data.logger.debug(f"Final Chunk URL: {final_chunk_url}")

        chunks_cnt = await loop.run_in_executor(None, chunks_count_regex.findall, final_chunk_url)
        if len(chunks_cnt) != 0:
            chunks_cnt = chunks_cnt[0]
            base_chunk_url = await loop.run_in_executor(
                None, final_chunk_url.replace, chunks_cnt, "CHUNK"
            )
            row_data.update({"audio_url": base_chunk_url, "audio_chunk_count": chunks_cnt})
            voxpredict.data.logger.debug(f"Base Chunk URL: {base_chunk_url} | Count: {chunks_cnt}")
    except Exception as e:
        voxpredict.data.logger.critical(e)

    return row_data


async def main(calls_data):
    try:
        async with ClientSession(
            connector=TCPConnector(ssl=False, limit=25),
            headers=voxpredict.data.headers,
        ) as sess:
            tasks = [scrape(sess, row) for row in calls_data]
            audio_data = [
                await audio for audio in tqdm(asyncio.as_completed(tasks), total=len(calls_data))
            ]
            with open("audio_data.pickle", "wb") as f:
                pickle.dump(audio_data, f)
        audio_data = [pd.DataFrame(data_dict, index=[0]) for data_dict in audio_data]
        audio_df = pd.concat(audio_data)
        with sqlite3.connect(voxpredict.data.earnings_db_path) as sql_connection:
            audio_df.to_sql(
                "call_references",
                con=sql_connection,
                index=False,
                if_exists="append",
            )
    except:
        asyncio.run(main(calls_data))


if __name__ == "__main__":
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    scraped_data = voxpredict.data.load_db_data(voxpredict.data.earnings_db_path, "calls")
    data = list(filter(filter_error_data, scraped_data))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(data))
