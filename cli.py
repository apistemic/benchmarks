import itertools
import logging
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from urllib.parse import quote_plus

import anthropic
import pandas as pd
from minimalkv.fs import FilesystemStore
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import FunctionTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from apistemic.benchmarks.datasets.companies import fetch_companies_df
from apistemic.benchmarks.datasets.competitors import fetch_competitor_votes
from apistemic.benchmarks.datasets.recommendations import fetch_recommendations
from apistemic.benchmarks.llms import get_chat_llms_by_key
from apistemic.benchmarks.llms import get_embedding_llms
from apistemic.benchmarks.models import CompetitvenessRatingAnswer
from apistemic.benchmarks.plots import create_box_plot
from apistemic.benchmarks.plots import create_r2_plot
from apistemic.benchmarks.plots import create_recommendations_box_plot
from apistemic.benchmarks.plots import create_spearman_plot
from apistemic.benchmarks.transformers import CompanyEmbeddingTransformer
from apistemic.benchmarks.transformers import CompanyTupleTransformer
from apistemic.benchmarks.transformers import DomainEmbeddingTransformer
from apistemic.benchmarks.transformers import EmbeddingDiffTransformer
from apistemic.benchmarks.transformers import LoadOrganizationTransformer
from apistemic.benchmarks.transformers import OneHotEncodingTransformer
from apistemic.benchmarks.util import create_competitiveness_prompt
from apistemic.benchmarks.util import evaluate_similarity_predictions


def run_embedding_classification(run_count: int = 10):
    df = fetch_competitor_votes()
    df_agg = df.groupby(["from_organization_id", "to_organization_id"]).mean(
        "similarity"
    )
    df_agg = df_agg.reset_index()[
        ["from_organization_id", "to_organization_id", "similarity"]
    ]

    # shuffle
    df_agg = df_agg.sample(frac=1.0)

    embedders = get_embedding_llms()

    # create one split per run
    splits = [train_test_split(df_agg, test_size=0.25) for _ in range(run_count)]

    # apply each embedder to each split
    runs = list(itertools.product(embedders, splits))

    results_per_model = {}
    for embedder, (df_train, df_test) in tqdm(runs):
        model_name = embedder.model

        print("\n")
        print("#" * 50)
        print(embedder.model)
        print("#" * 50)

        company_pipeline = Pipeline(
            [
                ("load", LoadOrganizationTransformer()),
                ("emb", CompanyEmbeddingTransformer(embedder=embedder)),
            ]
        )

        # Create feature pipeline (expensive operations done once)
        feature_pipeline = Pipeline(
            [
                (
                    "data_transformer",
                    CompanyTupleTransformer(company_pipeline=company_pipeline),
                ),
                ("diff", EmbeddingDiffTransformer()),
            ]
        )

        # Create gridsearch pipeline (only for hyperparameter tuning)
        gridsearch_pipeline = Pipeline(
            [
                ("sel", SelectPercentile(score_func=f_regression, percentile=0)),
                # SVR improves results, but is far too slow
                ("reg", Ridge()),
            ]
        )

        gridsearch = GridSearchCV(
            gridsearch_pipeline,
            param_grid={
                "sel__percentile": [50, 75, 100],
                "reg__alpha": [0.1, 1.0, 10.0, 100.0],
            },
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        # Create full pipeline with GridSearchCV as final estimator
        full_pipeline = Pipeline(
            [
                ("features", feature_pipeline),
                ("model", gridsearch),
            ]
        )

        # Fit full pipeline
        full_pipeline.fit(df_train, df_train["similarity"])

        print(f"Best parameters: {full_pipeline.named_steps['model'].best_params_}")
        best_score = full_pipeline.named_steps["model"].best_score_
        print(f"Best CV R^2 score: {best_score:.4f}")

        # Get predictions on test set
        y_pred = full_pipeline.predict(df_test)

        # Evaluate the model on test set
        metrics = evaluate_similarity_predictions(df_test["similarity"], y_pred)

        # add the run metrics to the results per model
        try:
            results_per_model[model_name].append(metrics)
        except KeyError:
            # first run of model
            results_per_model[model_name] = [metrics]

        # Create results dataframe with test predictions
        df_test_results = df_test.copy()
        df_test_results["similarity_predicted"] = y_pred

        print("\nSample predictions:")
        print(
            df_test_results[
                [
                    "from_organization_id",
                    "to_organization_id",
                    "similarity_predicted",
                    "similarity",
                ]
            ].head()
        )

    # Create box plot
    create_box_plot(results_per_model)


def run_scoring():
    df = fetch_competitor_votes()
    df_from = fetch_companies_df(df["from_organization_id"])
    df_to = fetch_companies_df(df["to_organization_id"])
    df = df.merge(
        df_from, left_on="from_organization_id", right_on="id", suffixes=("", "_from")
    )
    df = df.merge(
        df_to, left_on="to_organization_id", right_on="id", suffixes=("", "_to")
    )

    print(df)

    store = FilesystemStore(".cache")

    llms_by_key = get_chat_llms_by_key()
    results = {}
    for llm_key, llm in sorted(llms_by_key.items(), key=lambda _: random.random()):
        print("#" * 50)
        print(f"Running {llm}")
        print("#" * 50)

        structured_llm = llm.with_structured_output(CompetitvenessRatingAnswer)

        def prompt_rating_with_cache(company_name, competing_company_name) -> int:
            cache_key = f"{company_name}_{competing_company_name}_{llm_key}"
            cache_key = quote_plus(cache_key, safe="")

            try:
                cached_result = store.get(cache_key)
                return pickle.loads(cached_result)
            except KeyError:
                logging.debug(f"No cache hit for {cache_key}, invoking LLM...")
                rating = prompt_rating(company_name, competing_company_name)
                store.put(cache_key, pickle.dumps(rating))
                return rating

        def prompt_rating(company_name, competing_company_name) -> int:
            for i in range(3):
                try:
                    prompt = create_competitiveness_prompt(
                        company_name, competing_company_name
                    )
                    result = structured_llm.invoke(prompt)
                    return result.value
                except KeyboardInterrupt:
                    logging.info("KeyboardInterrupt received, stopping execution.")
                    raise
                except anthropic.RateLimitError:
                    logging.error("Rate limit exceeded, retrying...")
                    sleep(10)
                except Exception as e:
                    logging.error(f"Error invoking LLM: {e}")
                    if i < 2:
                        logging.warning("Retrying...")
                        sleep(2 ** (i + 1))
                    else:
                        logging.error("Failed after 3 attempts.")
                        raise e

        with ThreadPoolExecutor(max_workers=16) as executor:
            scores = list(
                tqdm(
                    executor.map(
                        lambda row: prompt_rating_with_cache(row[0], row[1]),
                        df[["name", "name_to"]].itertuples(index=False),
                    ),
                    total=len(df),
                    smoothing=0.01,
                    desc=f"requesting {llm_key}",
                )
            )

        df["rating"] = list(scores)
        df["score"] = (df["rating"] - 1.0) / 4.0
        print(df)

        # oai has .model_name, anthropic has .model
        # -> use separate identifier
        results[llm_key] = evaluate_similarity_predictions(
            df["similarity"], df["score"]
        )

    for model, metrics in results.items():
        print("\n")
        print(f"{model}:")
        print(metrics)

    # Create R² and Spearman correlation plots
    create_r2_plot(results)
    create_spearman_plot(results)


def run_recommendations(run_count: int = 10):
    df = fetch_recommendations()
    print(df)

    # make it harder by only having one domain in dataset
    # -> regressor cannot pinpoint domain's embedding if seen twice
    df = df.drop_duplicates(subset=["domain"])

    # shuffle as recommendations are ordered by created_at
    # - same users in batches
    # - similar recommendations in batches
    # df = df.sample(n=10_000, random_state=42)
    df = df.sample(frac=1.0, random_state=42)

    splits = [train_test_split(df, test_size=0.25) for _ in range(run_count)]

    # apply each embedder to each split
    embedders = get_embedding_llms() + [None]
    runs = list(itertools.product(embedders, splits))

    results_per_model = {}
    for embedder, (df_train, df_test) in tqdm(runs):
        model_name = embedder.model if embedder else "baseline"

        print("\n")
        print("#" * 50)
        print(embedder.model if embedder else "No embedder")
        print("#" * 50)

        # Create feature pipeline (expensive operations done once)
        feature_pipeline = Pipeline(
            [
                (
                    "fu",
                    FeatureUnion(
                        [
                            (
                                "domain",
                                (
                                    DomainEmbeddingTransformer(embedder)
                                    if embedder
                                    else FunctionTransformer(
                                        lambda df: pd.DataFrame(index=df.index)
                                    )
                                ),
                            ),
                            ("user_id", OneHotEncodingTransformer("user_id")),
                            ("list_id", OneHotEncodingTransformer("list_id")),
                        ]
                    ),
                ),
            ]
        )

        # Create gridsearch pipeline (only for hyperparameter tuning)
        gridsearch_pipeline = Pipeline(
            [
                ("sel", SelectPercentile(score_func=f_regression, percentile=0)),
                ("reg", Ridge()),
            ]
        )

        gridsearch = GridSearchCV(
            gridsearch_pipeline,
            param_grid={
                "sel__percentile": [10, 50, 75, 100],
                "reg__alpha": [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
            },
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=10,
            refit=True,
        )

        # Create full pipeline with GridSearchCV as final estimator
        full_pipeline = Pipeline(
            [("features", feature_pipeline), ("model", gridsearch)]
        )

        # Prepare data for DomainEmbeddingTransformer
        X_train = df_train[["domain", "user_id", "list_id"]]
        y_train = df_train["rating"]
        X_test = df_test[["domain", "user_id", "list_id"]]
        y_test = df_test["rating"]

        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Fit full pipeline
        full_pipeline.fit(X_train, y_train)

        print(f"Best parameters: {full_pipeline.named_steps['model'].best_params_}")
        best_score = full_pipeline.named_steps["model"].best_score_
        print(f"Best CV R^2 score: {best_score:.4f}")

        # Get predictions on test set
        y_pred = full_pipeline.predict(X_test)

        # Evaluate the model on test set
        metrics = evaluate_similarity_predictions(y_test, y_pred)

        # add the run metrics to the results per model
        try:
            results_per_model[model_name].append(metrics)
        except KeyError:
            # first run of model
            results_per_model[model_name] = [metrics]

        # Create results dataframe with test predictions
        df_test_results = X_test.copy()
        df_test_results["rating_predicted"] = y_pred
        df_test_results["rating_actual"] = y_test

        print("\nSample predictions:")
        print(df_test_results[["domain", "rating_predicted", "rating_actual"]].head(10))

    # Create box plot
    create_recommendations_box_plot(results_per_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_recommendations()
    exit()
    run_scoring()
    run_embedding_classification()
