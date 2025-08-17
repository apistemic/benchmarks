import logging

import pandas as pd

from apistemic.benchmarks.datasets.util import get_db_engine

LOGGER = logging.getLogger(__name__)


def fetch_competitor_votes():
    # .env has postgres DB_URL
    sql = (
        "SELECT * FROM organizationsimilarityvotes osv "
        "WHERE osv.created_at > '2024-01-01'::date"
    )
    engine = get_db_engine()
    df = pd.read_sql_query(sql, con=engine)
    LOGGER.info(f"fetched competitor votes ({len(df)})")

    file_path = ".data/external/competitor-votes.parquet"
    df.to_parquet(file_path, index=False)
    df = pd.read_parquet(file_path)
    return df
