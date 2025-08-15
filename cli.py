import logging
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
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
from apistemic.benchmarks.datasets.competitors import fetch_competitor_votes
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
    
    plt.title('How Well Embedding Models Understand Companies')
    plt.xlabel('Embedding Model (applied to company name)')
    plt.ylabel('R² Score (based on embedded company name only)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('.data/plots/r2-scores-boxplot.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for model, model_r2_scores in zip(models, r2_scores):
        print(f"\n{model}:")
        print(f"  Mean R²: {np.mean(model_r2_scores):.4f}")
        print(f"  Std R²:  {np.std(model_r2_scores):.4f}")
        print(f"  Min R²:  {np.min(model_r2_scores):.4f}")
        print(f"  Max R²:  {np.max(model_r2_scores):.4f}")


def main(run_count: int = 5):

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
        GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=os.environ["GEMINI_API_KEY"]),
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
            print(f"Best CV R^2 score: {full_pipeline.named_steps['model'].best_score_:.4f}")

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
