import asyncio
import random
from pathlib import Path

import aiofiles
import aiomultiprocess
import voxpredict
from aiohttp import (
    ClientSession,
    ClientTimeout,
    ServerDisconnectedError,
    TCPConnector,
    client_exceptions,
)
from tqdm.auto import tqdm
from voxpredict import data


class EarningsConferenceCallDownloader:

    HEADERS = data.headers.update({"referer": "https://earningscast.com/calls"})

    def __init__(self, num_calls=None, connection_limit=10, timeout=3000):
        self.connection_limit = connection_limit
        self.timeout = timeout
        # self.timeout = ClientTimeout(total=timeout)
        # self.connector = TCPConnector(limit=self.connection_limit)

        self.calls_list = self._load_earnings_calls()
        if num_calls:
            self.calls_list = self.calls_list[:num_calls]

    @staticmethod
    def _get_chunk_urls(chunklist_url, call_id, chunks_cnt):
        base_url = f"{chunklist_url.rsplit('/', 1)[0]}/{call_id}"
        chunk_urls = [base_url.replace("CHUNK", str(x)) for x in range(int(chunks_cnt))]
        return chunk_urls

    def _load_earnings_calls(self):
        earnings_calls_list = []
        table_rows = data.load_db_data(data.earnings_db_path, "earnings_calls")
        for row in table_rows:
            (ticker, quarter, year, _, _, url, call_id, chunks_cnt) = row
            export_filepath = export_dir.joinpath(f"{ticker}_{year}_Q{quarter}.aac")
            if chunks_cnt is not None and not export_filepath.exists():
                earnings_calls_list.append(
                    [
                        self._get_chunk_urls(url, call_id, chunks_cnt),
                        [None] * int(chunks_cnt),
                        export_filepath,
                    ]
                )
        return earnings_calls_list

    @staticmethod
    async def _export_call(call):
        chunks = asyncio.get_running_loop().run_in_executor(None, b"".join, call[1])
        async with aiofiles.open(call[2], "wb") as audio_file:
            await audio_file.write(await chunks)
        return

    async def _fetch_chunk(self, session, url):
        await asyncio.sleep(0.1)
        # await asyncio.sleep(float(random.randrange(0, 100)) / 10.)
        async with session.get(url) as response:
            downloaded_data = await response.read()
            if not isinstance(downloaded_data, bytes):
                return await self._fetch_chunk(session, url)
            return [url, downloaded_data]

    async def _get_call(self, call_info):
        try:
            async with TCPConnector(limit=self.connection_limit, ssl=False) as conn:
                async with ClientSession(
                    headers=self.HEADERS, connector=conn, conn_timeout=self.timeout
                ) as session:
                    downloads = [
                        asyncio.create_task(self._fetch_chunk(session, url)) for url in call_info[0]
                    ]
                    for request in downloads:  # tqdm(downloads, total=len(downloads)):
                        url, chunk = await request
                        call_info[1][call_info[0].index(url)] = chunk
                    await asyncio.sleep(0)
                save = [True for chunk in call_info[1] if chunk is not None]
                if len(save) == len(call_info[1]):
                    await self._export_call(call_info)
                return
        except ServerDisconnectedError:
            await asyncio.sleep(1.5)
            return await self._get_call(call_info)

    async def run(self):
        """Run the downloader with multiprocessing and multithreading."""
        pool_tasks = []
        async with aiomultiprocess.Pool(
            processes=4, maxtasksperchild=64, childconcurrency=8, queuecount=2
        ) as pool:
            for call in self.calls_list:
                pool_tasks.append(pool.apply(self._get_call, args=[call]))
            for download in tqdm(asyncio.as_completed(pool_tasks), total=len(pool_tasks)):
                await download


if __name__ == "__main__":
    export_dir = Path(voxpredict.project_dir.joinpath("data"))
    downloader = EarningsConferenceCallDownloader(2000)
    asyncio.run(downloader.run())
