"""Data operations which work with databases."""
import asyncio
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import pandas as pd
from aiohttp import ServerDisconnectedError
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from deeptrade import project_dir
from deeptrade.datasets.schema import ConferenceCallChunk


class ConferenceCallChunksProcessor:
    """Insert/update conference call chunks objects to the global database."""

    save_path = Path("data/raw/audio/conference_calls")

    def __init__(
        self,
        ticker: str,
        year: int,
        quarter: int,
        url: str,
        url_chunk_path: str,
        num_chunks: int,
        session_maker: Any,
    ):
        self.call_reference = f"{ticker}_{year}_Q{quarter}"
        self.base_url = f"{url.rsplit('/', 1)[0]}/{url_chunk_path}"
        self.num_chunks = num_chunks
        self.save_path = ConferenceCallChunksProcessor.save_path.joinpath(self.call_reference)
        self.session = session_maker()

    def _get_url(self, index):
        return self.base_url.replace("CHUNK", str(index))

    def _get_filepath(self, index):
        return str(self.save_path.joinpath(f"{index}.aac"))

    def _get_data(self, index):
        return ConferenceCallChunk(
            call_reference=self.call_reference,
            index=index,
            total=self.num_chunks,
            filepath=self._get_filepath(index),
            url=self._get_url(index),
        )

    async def download_chunk(self, session: aiohttp.ClientSession, url: str, filepath: Path):
        download_successful = False
        while not download_successful:
            # try:
            await asyncio.sleep(delay=0.05)
            async with session.get(url) as response:
                content = await response.read()
                if isinstance(content, bytes):
                    download_successful = True
                    return content.decode("")
            # except ServerDisconnectedError:
            #     await asyncio.sleep(1.5)
            #     await self.session.close()
            #     await self.download()
                    async with aiofiles.open(filepath, "wb") as chunk_file:
                        await chunk_file.write(content)

    def to_db(self):
        """Save the conference call chunk objects for the conference call to the database."""
        chunks = [self._get_data(index) for index in range(self.num_chunks)]
        self.session.bulk_save_objects(chunks)
        self.session.commit()
        self.session.close()


def load_db(table_name: str, session_maker: Any) -> pd.DataFrame:
    """Return a dataframe populated with the data from a database's table.

        Args:
            table_name(str): Name of the table to source data from.
            session_maker(Any): Session maker bound to the engine.

        Returns:
            (pd.DataFrame): Database table loaded into a dataframe.
    """
    session = session_maker()
    with session.connection() as db_connection:
        db = pd.read_sql_table(table_name, db_connection)
        table = pd.DataFrame(columns=["call_reference", "index", "total", "filepath", "url"])
        table.to_sql("conference_call_chunks", db_connection, if_exists="replace", index=False)
        db.dropna(inplace=True)
    session.close()
    return db


def main(db_url: str, table_name: str = "earnings_calls"):
    """Main script run function to create the conference call chunks table.

        Args:
            db_url(str): Database URL.
            table_name(str): Name of the table to source data from.
    """
    engine = create_engine(db_url)
    session_maker = sessionmaker(bind=engine)
    db = load_db(table_name, session_maker)
    with tqdm(total=len(db)) as progress_bar:
        for _, ticker, quarter, year, _, _, url, path, count in db.itertuples():
            processor = ConferenceCallChunksProcessor(
                ticker, int(year), int(quarter), url, path, int(count), session_maker
            )
            processor.to_db()
            progress_bar.update()
    engine.dispose()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", default=f"sqlite:///{str(project_dir.joinpath('data/db/earnings.sqlite'))}"
    )
    parser.add_argument("--table-name", default="earnings_calls")
    args = parser.parse_args()

    main(args.db, args.table_name)
