import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your transaction data
data = pd.read_csv("data/processed/processed_data_encoded.csv")

# -------------------------------------------
# STEP 1: Calculate RFM Metrics
# -------------------------------------------

# Define snapshot date as 1 day after the latest transaction
snapshot_date = pd.to_datetime(data['TransactionStartTime']).max() + pd.Timedelta(days=1)

# Ensure TransactionStartTime is datetime
data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

# Group by CustomerId to calculate RFM
rfm = data.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
    'CustomerId': 'count',                                             # Frequency
    'Amount': 'sum'                                                    # Monetary
}).rename(columns={
    'TransactionStartTime': 'Recency',
    'CustomerId': 'Frequency',
    'Amount': 'Monetary'
}).reset_index()

# -------------------------------------------
# STEP 2: Scale the RFM features
# -------------------------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# -------------------------------------------
# STEP 3: KMeans Clustering
# -------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# -------------------------------------------
# STEP 4: Identify the High-Risk Cluster
# -------------------------------------------
cluster_profiles = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

# Sort clusters based on low engagement
high_risk_cluster = cluster_profiles.sort_values(by=['Frequency', 'Monetary', 'Recency'], ascending=[True, True, False]).index[0]

# Assign binary label
rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

# Drop the cluster column (optional)
rfm.drop(columns='Cluster', inplace=True)

# -------------------------------------------
# STEP 5: Merge with the main dataset
# -------------------------------------------
final_data = data.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# Save it if needed
final_data.to_csv("data/processed/data_with_target.csv", index=False)

print("✅ 'is_high_risk' proxy target added to dataset.")