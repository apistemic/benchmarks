import numpy as np
import matplotlib.pyplot as plt
from .models import EvaluationMetrics


def create_box_plot(all_results: dict[str, list[EvaluationMetrics]]) -> None:
    """Create box plot of R² scores by model."""
    models = list(all_results.keys())
    r2_scores = []

    for model in models:
        model_r2_scores = [metrics.r2 for metrics in all_results[model]]
        r2_scores.append(model_r2_scores)

    plt.style.use("grayscale")
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
