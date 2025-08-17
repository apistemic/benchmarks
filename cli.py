import itertools
import logging
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from urllib.parse import quote_plus

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from minimalkv.fs import FilesystemStore
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from apistemic.benchmarks.datasets.companies import fetch_companies_df
from apistemic.benchmarks.datasets.competitors import fetch_competitor_votes
from apistemic.benchmarks.models import CompetitvenessRatingAnswer
from apistemic.benchmarks.plots import create_box_plot
from apistemic.benchmarks.plots import create_r2_plot
from apistemic.benchmarks.plots import create_spearman_plot
from apistemic.benchmarks.transformers import CompanyEmbeddingTransformer
from apistemic.benchmarks.transformers import CompanyTupleTransformer
from apistemic.benchmarks.transformers import EmbeddingDiffTransformer
from apistemic.benchmarks.transformers import LoadOrganizationTransformer
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

    embedders = [
        GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", google_api_key=os.environ["GEMINI_API_KEY"]
        ),
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        OpenAIEmbeddings(model="text-embedding-3-small"),
        OpenAIEmbeddings(model="text-embedding-3-large"),
    ]

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

    llms = {
        "google__gemini-2.5-flash-lite": ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", google_api_key=os.environ["GEMINI_API_KEY"]
        ),
        "google__gemini-2.5-flash": ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=os.environ["GEMINI_API_KEY"]
        ),
        "google__gemini-2.5-pro": ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", google_api_key=os.environ["GEMINI_API_KEY"]
        ),
        "anthropic__claude-opus-4-1": ChatAnthropic(
            model="claude-opus-4-1-20250805", timeout=30
        ),
        "anthropic__claude-sonnet-4": ChatAnthropic(
            model="claude-sonnet-4-20250514", timeout=30
        ),
        "anthropic__claude-3-5-haiku": ChatAnthropic(
            model="claude-3-5-haiku-20241022", timeout=30
        ),
        "openai__gpt-5": ChatOpenAI(model="gpt-5"),
        "openai__gpt-5-mini": ChatOpenAI(model="gpt-5-mini"),
        "openai__gpt-5-nano": ChatOpenAI(model="gpt-5-nano"),
    }

    store = FilesystemStore(".cache")

    results = {}
    for llm_identifier, llm in sorted(llms.items(), key=lambda _: random.random()):
        print("#" * 50)
        print(f"Running {llm}")
        print("#" * 50)

        structured_llm = llm.with_structured_output(CompetitvenessRatingAnswer)

        def prompt_rating_with_cache(company_name, competing_company_name) -> int:
            cache_key = f"{company_name}_{competing_company_name}_{llm_identifier}"
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
                    desc=f"requesting {llm_identifier}",
                )
            )

        df["rating"] = list(scores)
        df["score"] = (df["rating"] - 1.0) / 4.0
        print(df)

        # oai has .model_name, anthropic has .model
        # -> use separate identifier
        results[llm_identifier] = evaluate_similarity_predictions(
            df["similarity"], df["score"]
        )

    for model, metrics in results.items():
        print("\n")
        print(f"{model}:")
        print(metrics)

    # Create R² and Spearman correlation plots
    create_r2_plot(results)
    create_spearman_plot(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_scoring()
    run_embedding_classification()
