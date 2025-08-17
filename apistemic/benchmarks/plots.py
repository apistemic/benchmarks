from datetime import date

import matplotlib.pyplot as plt
import numpy as np

from .models import EvaluationMetrics


def create_box_plot(all_results: dict[str, list[EvaluationMetrics]]) -> None:
    """Create box plot of R² scores by model."""
    # Sort models by median R² score in ascending order (lowest bottom, highest top)
    models = sorted(
        all_results.keys(),
        key=lambda x: np.median([metrics.r2 for metrics in all_results[x]]),
    )
    r2_scores = []

    for model in models:
        model_r2_scores = [metrics.r2 for metrics in all_results[model]]
        r2_scores.append(model_r2_scores)

    plt.style.use("grayscale")
    plt.figure(figsize=(8, 8))
    plt.tight_layout()
    plt.boxplot(r2_scores, tick_labels=models, patch_artist=False, vert=False)

    today = get_date_str()
    plt.suptitle(f"LLM Company Knowledge: Predictive Power of Embeddings ({today})")
    plt.xlabel("R² Score")
    plt.ylabel("LLM Embedding")
    plt.grid(True, alpha=0.3, axis="x")
    plt.yticks(rotation=0)

    # Add watermark
    add_watermark()

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
    # Sort models by R² score in ascending order (lowest at bottom, highest at top)
    models = sorted(results.keys(), key=lambda x: results[x].r2)
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
    plt.figure(figsize=(8, 8))
    bars = plt.barh(range(len(models)), r2_scores)

    today = get_date_str()
    plt.suptitle(f"LLM Company Knowledge: Accuracy vs Human Experts ({today})")
    plt.xlabel("R² Score")
    plt.ylabel("LLM")
    plt.yticks(range(len(models)), display_names)
    plt.grid(True, alpha=0.3, axis="x")
    plt.xlim(-1.0, 1.0)

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        x_pos = bar.get_width() + 0.01 if score >= 0 else bar.get_width() - 0.01
        ha = "left" if score >= 0 else "right"
        plt.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            ha=ha,
            va="center",
        )

    # Add watermark
    add_watermark()

    # Save the plot
    plt.tight_layout()
    plt.savefig(".data/plots/r2-scores-barplot.png", dpi=300, bbox_inches="tight")


def create_spearman_plot(results: dict[str, EvaluationMetrics]) -> None:
    """Create bar plot of Spearman correlations by LLM model."""
    # Sort models by Spearman correlation ascending (lowest bottom, highest top)
    models = sorted(results.keys(), key=lambda x: results[x].spearman_corr)
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
    plt.figure(figsize=(8, 8))
    bars = plt.barh(range(len(models)), spearman_corrs)

    today = get_date_str()
    plt.suptitle(f"LLM Company Knowledge: Ranking Correlation with Experts ({today})")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("LLM")
    plt.yticks(range(len(models)), display_names)
    plt.grid(True, alpha=0.3, axis="x")
    plt.xlim(0, 1.0)

    # Add value labels on bars
    for bar, corr in zip(bars, spearman_corrs):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{corr:.3f}",
            ha="left",
            va="center",
        )

    # Add watermark
    add_watermark()

    # Save the plot
    plt.tight_layout()
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


def get_date_str() -> str:
    """Get current date formatted as 'Month Year'."""
    return date.today().strftime("%B %Y")


def add_watermark() -> None:
    """Add Apistemic watermark to current plot."""
    plt.text(
        0.98,
        0.02,
        "© Apistemic GmbH, apistemic.com",
        transform=plt.gca().transAxes,
        fontsize=10,
        alpha=0.6,
        ha="right",
        va="bottom",
        color="gray",
    )
