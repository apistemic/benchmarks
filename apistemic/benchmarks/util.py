import numpy as np
from sklearn.isotonic import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from apistemic.benchmarks.models import EvaluationMetrics

COMPETITIVENESS_PROMPT_TEMPLATE = (
    "You are an expert business analyst. "
    "Given the names of two companies, "
    "your task is to determine how competitive they are in terms of "
    "business focus, products, and market presence. "
    "We see competitiveness as a transitive relation, "
    "but not a symmetric one. "
    "For example, if Company A is a competitor of Company B, "
    "it does not imply that Company B is a competitor of Company A. "
    "So while a small taxi company has Uber as a direct competitor "
    "(e.g. if the taxi company is from NYC), "
    "the small taxi company is not a direct competitor of Uber. "
    "Your task is now to evaluate and rate the competitiveness of two "
    "companies on a scale from 1 to 5, where:\n"
    "1: neither similar nor competitors\n"
    "2: somewhat similar\n"
    "3: distant competitor, similar product\n"
    "4: competitor, but with different geo/size/etc.\n"
    "5: direct competitor\n\n"
    "So looking at {company_name}, is {competing_company_name} a relevant "
    "competitor? "
)


def create_competitiveness_prompt(company_name, competing_company_name):
    return COMPETITIVENESS_PROMPT_TEMPLATE.format(
        company_name=company_name, competing_company_name=competing_company_name
    )


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
