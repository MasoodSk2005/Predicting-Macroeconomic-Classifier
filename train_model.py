import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Initializing Advanced Machine Learning Data Pipeline...")

# 1. Load World Bank Categorical Labels
df_metadata = pd.read_csv("Metadata_Country_API_NY.GDP.PCAP.CD_DS2_en_csv_v2_358064.csv")
df_metadata_clean = df_metadata[['Country Code', 'IncomeGroup', 'TableName']].dropna(subset=['IncomeGroup'])
df_metadata_clean['TableName'] = df_metadata_clean['TableName'].str.strip()

# 2. Load World Bank Numerical GDP Metrics
df_gdp_pc = pd.read_csv("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_358064.csv", skiprows=4)
df_gdp_pc_clean = df_gdp_pc[['Country Code', '2023']].rename(columns={'2023': 'GDP_pc_2023'}).dropna()
merged_wb = pd.merge(df_metadata_clean, df_gdp_pc_clean, on='Country Code', how='inner')

# 3. Load UN Human Development Metrics
df_hdi = pd.read_excel("HDR25_Statistical_Annex_HDI_Table.xlsx", skiprows=4, sheet_name='Table 1. HDI')
df_hdi_clean = df_hdi.iloc[3:].copy()
df_hdi_clean.columns = [
    'HDI_Rank', 'Country', 'Value', 'empty1', 'Life_Expectancy', 'empty2',
    'Expected_Schooling', 'empty3', 'Mean_Schooling', 'empty4', 'GNI_per_Capita',
    'empty5', 'GNI_minus_HDI_Rank', 'empty6', 'HDI_Rank_2022'
]
df_hdi_clean = df_hdi_clean.rename(columns={'Value': 'HDI_Value'})
df_hdi_clean = df_hdi_clean[['Country', 'HDI_Value', 'Life_Expectancy', 'Expected_Schooling', 'Mean_Schooling', 'GNI_per_Capita']].dropna(subset=['Country', 'HDI_Value'])
df_hdi_clean['Country'] = df_hdi_clean['Country'].str.strip()

# 4. Consolidate into Master Frame
merged_final = pd.merge(merged_wb, df_hdi_clean, left_on='TableName', right_on='Country', how='inner')

features_list = ['GDP_pc_2023', 'HDI_Value', 'Life_Expectancy', 'Expected_Schooling', 'Mean_Schooling', 'GNI_per_Capita']
for col in features_list:
    merged_final[col] = merged_final[col].astype(str).str.replace(r'[^²\d\.]', '', regex=True)
    merged_final[col] = pd.to_numeric(merged_final[col], errors='coerce')

data_model = merged_final.dropna(subset=features_list).copy()
data_model['Target'] = data_model['IncomeGroup'].apply(lambda x: 1 if 'High income' in str(x) else 0)

# 5. Model Compilation
X = data_model[features_list]
y = data_model['Target']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
print(" Random Forest Model Successfully Trained")

# 6. Exporting Artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Create a robust lookup system mapping country names to pre-extracted values
lookup_dict = {}
for idx, row in data_model.iterrows():
    name_key = str(row['TableName']).lower().strip()
    lookup_dict[name_key] = row[features_list].to_dict()

with open('country_lookup.pkl', 'wb') as f:
    pickle.dump(lookup_dict, f)

print(" All model files and lookup maps updated successfully!")