#will be creating pydantic schema to validate users request

from pydantic import BaseModel,Field

class StockRequest(BaseModel):
    stock_name: str = Field(..., example="TCS")
    days: int = Field(..., gt=0, le=30, example=7)
    model: str = Field(..., example="lstm")