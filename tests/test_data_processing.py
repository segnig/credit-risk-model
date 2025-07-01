# test_data_processing.py
import requests

url = "http://localhost:8000/predict"

payload = {
    "TransactionId": "txn123",
    "BatchId": "batch001",
    "AccountId": "acc456",
    "SubscriptionId": "sub789",
    "CustomerId": "cust001",
    "CountryCode": "ET",
    "CurrencyCode": "ETB",
    "ProviderId": "ProviderId_1",
    "ProductId": "ProductId_10",
    "ProductCategory": "airtime",
    "ChannelId": "ChannelId_2",
    "Amount": 500.0,
    "Value": 500.0,
    "TransactionStartTime": "2025-06-25T10:30:00",
    "PricingStrategy": 1,
    "FraudResult": 0
}

response = requests.post(url, json=payload)
print("Status:", response.status_code)
print("Response:", response.json())
