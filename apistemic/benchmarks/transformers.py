import numpy as np
import pandas as pd
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings.embeddings import Embeddings
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from apistemic.benchmarks.datasets.companies import fetch_companies_df


class LoadOrganizationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        """
        # Set fitted attributes for sklearn's check_is_fitted
        self.n_features_in_ = len(X) if hasattr(X, "__len__") else 1
        return self

    def transform(self, X):
        """
        Transform the input data by extracting organization IDs and names.
        """
        if not isinstance(X, pd.Series):
            raise ValueError("Input must be pd.Series")

        df = fetch_companies_df(X.unique())

        # ensure all IDs are present
        assert set(df["id"].unique()) == set(X.unique())

        # use id as index
        df = df.set_index("id")

        # then view through X
        df = df.loc[X]

        return df


class CompanyEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Transformer to create embeddings for companies."""

    embedder: Embeddings

    def __init__(self, embedder: Embeddings):
        self.embedder = embedder

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        """
        # Set fitted attributes for sklearn's check_is_fitted
        self.n_features_in_ = (
            X.shape[1]
            if hasattr(X, "shape")
            else len(X.columns)
            if hasattr(X, "columns")
            else 1
        )
        return self

    def transform(self, X):
        """
        Transform the input data by engineering features.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be pd.DataFrame")

        # use df for convenience
        df = X

        store = LocalFileStore("./cache/")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.embedder, store, namespace=self.embedder.model, key_encoder="sha256"
        )

        df["embedding_name"] = cached_embedder.embed_documents(
            df["name"].values.tolist()
        )
        return df


class DomainEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedder: Embeddings):
        self.embedder = embedder

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        """
        # Set fitted attributes for sklearn's check_is_fitted
        self.n_features_in_ = (
            X.shape[1]
            if hasattr(X, "shape")
            else len(X.columns)
            if hasattr(X, "columns")
            else 1
        )
        return self

    def transform(self, X):
        """
        Transform the input data by engineering features.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be pd.DataFrame")

        # use df for convenience
        df = X

        store = LocalFileStore("./cache/")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.embedder, store, namespace=self.embedder.model, key_encoder="sha256"
        )

        return cached_embedder.embed_documents(df["domain"].values.tolist())


class DomainEmbeddingExtractorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Extract embedding vectors as numpy array for sklearn"""
        return np.array(X["embedding_domain"].tolist())


class CompanyTupleTransformer(BaseEstimator, TransformerMixin):
    # company pipeline turns ids to features
    company_pipeline: TransformerMixin

    def __init__(self, company_pipeline=None):
        self.company_pipeline = company_pipeline

    def fit(self, X, y=None):
        # Fit the company pipeline
        all_org_ids = pd.concat(
            [X["from_organization_id"], X["to_organization_id"]]
        ).unique()
        self.company_pipeline.fit(pd.Series(all_org_ids))

        # Set fitted attributes for sklearn's check_is_fitted
        self.n_features_in_ = (
            X.shape[1]
            if hasattr(X, "shape")
            else len(X.columns)
            if hasattr(X, "columns")
            else 1
        )

        return self

    def transform(self, X, y=None):
        assert "from_organization_id" in X.columns and "to_organization_id" in X.columns
        df = X

        df_left = (
            self.company_pipeline.transform(df["from_organization_id"])
            .reset_index()
            .rename(lambda x: f"from_{x}", axis=1)
        )
        df_right = (
            self.company_pipeline.transform(df["to_organization_id"])
            .reset_index()
            .rename(lambda x: f"to_{x}", axis=1)
        )

        df = pd.concat([df.reset_index(drop=True), df_left, df_right], axis=1)
        return df


class EmbeddingDiffTransformer(BaseEstimator, TransformerMixin):
    """Create embedding difference features from organization embeddings."""

    def fit(self, X, y=None):
        # Set fitted attributes for sklearn's check_is_fitted
        self.n_features_in_ = (
            X.shape[1]
            if hasattr(X, "shape")
            else len(X.columns)
            if hasattr(X, "columns")
            else 1
        )
        return self

    def transform(self, X):
        df = X.copy()

        # Always compute embedding differences
        embedding_diff = df[["from_embedding_name", "to_embedding_name"]].apply(
            lambda t: pd.Series(t["from_embedding_name"])
            - pd.Series(t["to_embedding_name"]),
            axis=1,
        )
        diff_cols = [f"embedding_diff_{i}" for i in range(len(embedding_diff.columns))]
        embedding_diff.columns = diff_cols
        return embedding_diff


class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    """Create one-hot encoded features from a specific column."""

    def __init__(self, column_name):
        self.column_name = column_name
        from sklearn.preprocessing import OneHotEncoder

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be pd.DataFrame")

        if self.column_name not in X.columns:
            raise ValueError(f"Column '{self.column_name}' not found in DataFrame")

        # Fit the encoder on the specified column
        self.encoder.fit(X[[self.column_name]])

        # Set fitted attributes for sklearn's check_is_fitted
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else 1
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be pd.DataFrame")

        if self.column_name not in X.columns:
            raise ValueError(f"Column '{self.column_name}' not found in DataFrame")

        # Transform the specified column to one-hot encoded features
        encoded = self.encoder.transform(X[[self.column_name]])

        return encoded
