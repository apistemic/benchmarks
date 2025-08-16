from pydantic import BaseModel


class CompetitvenessRatingAnswer(BaseModel):
    value: int


class Company(BaseModel):
    name: str
