import pandas as pd

from apistemic.benchmarks.datasets.util import get_db_engine


def fetch_recommendations() -> pd.DataFrame:
    sql = (
        "SELECT *"
        " FROM domainlistentries dle"
        " JOIN domains d ON dle.domain_id = d.id"
        # recent ratings
        " WHERE dle.created_at > NOW() - '1 year'::interval"
        # remove non-user entries
        " AND dle.user_id IS NOT NULL"
    )
    engine = get_db_engine()
    df = pd.read_sql_query(sql, con=engine)
    return df
