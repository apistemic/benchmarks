from pydantic import BaseModel


class CompetitvenessRatingAnswer(BaseModel):
    value: int


class Company(BaseModel):
    name: str


class EvaluationMetrics(BaseModel):
    r2: float
    rmse: float
    mse: float
    spearman_corr: float
    spearman_p: float
