import pandas as pd
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.orm import Session

from apistemic.benchmarks.datasets.util import get_db_engine


def fetch_companies_df(company_ids):
    company_ids = list(map(int, company_ids))  # Ensure IDs are integers

    engine = get_db_engine()

    # create sqlalchemy query
    metadata = MetaData()
    organizations_table = Table("organizations", metadata, autoload_with=engine)

    with Session(engine) as session:
        query = session.query(organizations_table).filter(
            organizations_table.c.id.in_(company_ids)
        )
        df = pd.read_sql_query(query.statement, con=engine)
        return df
