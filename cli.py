from concurrent.futures import ThreadPoolExecutor
import logging
import os
import pickle
import random
from time import sleep
from urllib.parse import quote_plus

import anthropic
from tqdm import tqdm
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from minimalkv.fs import FilesystemStore
from enum import Enum
from pydantic import BaseModel

from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
from apistemic.benchmarks.datasets.companies import fetch_companies_df
from apistemic.benchmarks.datasets.competitors import fetch_competitor_votes
from apistemic.benchmarks.models import CompetitvenessRatingAnswer
from apistemic.benchmarks.transformers import (
    CompanyEmbeddingTransformer,
    LoadOrganizationTransformer,
    CompanyTupleTransformer,
    EmbeddingDiffTransformer,
)


class ModelType(Enum):
    COSINE = "cosine"
    SVR = "svr"


class EvaluationMetrics(BaseModel):
    r2: float
    rmse: float
    mse: float
    spearman_corr: float
    spearman_p: float


def evaluate_similarity_predictions(y_true, y_pred):
    """Evaluate predicted similarity against ground truth."""
    # R² score
    r2 = r2_score(y_true, y_pred)

    # RMSE and MSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)

    print("\n--- Evaluation Results ---")
    print(f"R² score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")

    return EvaluationMetrics(
        r2=r2,
        rmse=rmse,
        mse=mse,
        spearman_corr=spearman_corr,
        spearman_p=spearman_p,
    )


def create_box_plot(all_results: dict[str, list[EvaluationMetrics]]) -> None:
    """Create box plot of R² scores by model."""
    models = list(all_results.keys())
    r2_scores = []

    for model in models:
        model_r2_scores = [metrics.r2 for metrics in all_results[model]]
        r2_scores.append(model_r2_scores)

    plt.figure(figsize=(10, 6))
    plt.boxplot(r2_scores, tick_labels=models, patch_artist=False)

    plt.title("How Well Embedding Models Understand Companies")
    plt.xlabel("Embedding Model (applied to company name)")
    plt.ylabel("R² Score (based on embedded company name only)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(".data/plots/r2-scores-boxplot.png", dpi=300, bbox_inches="tight")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for model, model_r2_scores in zip(models, r2_scores):
        print(f"\n{model}:")
        print(f"  Mean R²: {np.mean(model_r2_scores):.4f}")
        print(f"  Std R²:  {np.std(model_r2_scores):.4f}")
        print(f"  Min R²:  {np.min(model_r2_scores):.4f}")
        print(f"  Max R²:  {np.max(model_r2_scores):.4f}")


def create_r2_plot(results: dict[str, EvaluationMetrics]) -> None:
    """Create bar plot of R² scores by LLM model."""
    # Sort models by R² score in descending order
    models = sorted(results.keys(), key=lambda x: results[x].r2, reverse=True)
    r2_scores = [results[model].r2 for model in models]

    # Clean up model names for display
    display_names = []
    for model in models:
        if "__" in model:
            provider, model_name = model.split("__", 1)
            display_names.append(f"{provider}\n{model_name}")
        else:
            display_names.append(model)

    plt.style.use("grayscale")
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(models)), r2_scores)

    plt.title("LLM Performance on Competitiveness Rating Task")
    plt.xlabel("LLM Model")
    plt.ylabel("R² Score")
    plt.xticks(range(len(models)), display_names, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim(-1.0, 1.0)

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        y_pos = bar.get_height() + 0.01 if score >= 0 else bar.get_height() - 0.01
        va = "bottom" if score >= 0 else "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{score:.3f}",
            ha="center",
            va=va,
        )

    plt.tight_layout()

    # Save the plot
    plt.savefig(".data/plots/r2-scores-barplot.png", dpi=300, bbox_inches="tight")


def create_spearman_plot(results: dict[str, EvaluationMetrics]) -> None:
    """Create bar plot of Spearman correlations by LLM model."""
    # Sort models by Spearman correlation in descending order
    models = sorted(
        results.keys(), key=lambda x: results[x].spearman_corr, reverse=True
    )
    spearman_corrs = [results[model].spearman_corr for model in models]

    # Clean up model names for display
    display_names = []
    for model in models:
        if "__" in model:
            provider, model_name = model.split("__", 1)
            display_names.append(f"{provider}\n{model_name}")
        else:
            display_names.append(model)

    plt.style.use("grayscale")
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(models)), spearman_corrs)

    plt.title("LLM Performance on Competitiveness Rating Task")
    plt.xlabel("LLM Model")
    plt.ylabel("Spearman Correlation")
    plt.xticks(range(len(models)), display_names, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim(0, 1.0)

    # Add value labels on bars
    for bar, corr in zip(bars, spearman_corrs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{corr:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save the plot
    plt.savefig(
        ".data/plots/spearman-correlations-barplot.png", dpi=300, bbox_inches="tight"
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("LLM SPEARMAN CORRELATION RESULTS")
    print("=" * 60)
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"  Spearman ρ: {metrics.spearman_corr:.4f} (p={metrics.spearman_p:.4f})")
        print(f"  R²: {metrics.r2:.4f}")
        print(f"  RMSE: {metrics.rmse:.4f}")


def run_embedding_classification(run_count: int = 5):
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

    results_per_model = {}
    for embedder in embedders:
        results_per_run = []
        model_name = embedder.model

        for _ in range(1, run_count + 1):
            # split per run
            df_train, df_test = train_test_split(df_agg, test_size=0.25)

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
                    ("reg", SVR()),
                ]
            )

            gridsearch = RandomizedSearchCV(
                gridsearch_pipeline,
                param_distributions={
                    "sel__percentile": [50, 75, 100],
                    "reg__kernel": ["linear", "rbf"],
                    "reg__C": [0.1, 1.0, 10.0, 100.0],
                    "reg__epsilon": [0.01, 0.1, 0.2],
                },
                n_iter=100,
                cv=5,
                scoring="r2",
                n_jobs=8,
                verbose=1,
                refit=True,
                random_state=42,
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
            print(
                f"Best CV R^2 score: {full_pipeline.named_steps['model'].best_score_:.4f}"
            )

            # Get predictions on test set
            y_pred = full_pipeline.predict(df_test)

            # Evaluate the model on test set
            metrics = evaluate_similarity_predictions(df_test["similarity"], y_pred)
            results_per_run.append(metrics)

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

        results_per_model[model_name] = results_per_run
        print(results_per_run)

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
    for llm_identifier, llm in sorted(llms.items(), key=lambda x: random.random()):
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


def create_competitiveness_prompt(company_name, competing_company_name):
    prompt = (
        "You are an expert business analyst. "
        "Given the names of two companies, your task is to determine how competitive they are in terms of business focus, products, and market presence. "
        "We see competitiveness as a transitive relation, but not a symmetric one. "
        "For example, if Company A is a competitor of Company B, it does not imply that Company B is a competitor of Company A. "
        "So while a small taxi company has Uber as a direct competitor (e.g. if the taxi company is from NYC), the small taxi company is not a direct competitor of Uber. "
        "Your task is now to evaluate and rate the competitiveness of two companies on a scale from 1 to 5, where:\n"
        "1: neither similar nor competitors\n"
        "2: somewhat similar\n"
        "3: distant competitor, similar product\n"
        "4: competitor, but with different geo/size/etc.\n"
        "5: direct competitor\n\n"
        f"So looking at {company_name}, is {competing_company_name} a relevant competitor? "
    )
    return prompt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_scoring()
