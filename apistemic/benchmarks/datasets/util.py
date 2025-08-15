from functools import cache
import os
from sqlalchemy import create_engine


@cache
def get_db_engine():
    # using cache to have one instance only
    dsn = os.environ["DB_URL"]
    engine = create_engine(dsn)
    return engine
