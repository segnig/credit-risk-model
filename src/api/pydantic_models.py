from pydantic import BaseModel

class PredictionRequest(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CountryCode: str
    CurrencyCode: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str  # ISO format date string
    PricingStrategy: int
    FraudResult: int
    
class PredictionResponse(BaseModel):
    risk_probability: float
