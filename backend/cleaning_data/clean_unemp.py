import pandas as pd

# 1. Load your CSV
# Using 'r' to avoid the Unicode error we discussed earlier
file_path = r'C:\Users\mailm\Downloads\Disorders_by_country\backend\cleaning_data\Employment_Unemployment_GDP_data.csv'
df = pd.read_csv(file_path, low_memory=False)

# 2. Keep only desired columns
columns_to_keep = ["Country Name", "Year", "Unemployment Rate", "GDP (in USD)"]
df_filtered = df[columns_to_keep].copy() # Using .copy() to avoid warnings

# 3. NEW: Filter for specific countries
# This keeps only the rows where "Country Name" matches one of the items in the list
countries_to_keep = ["Morocco", "Lesotho", "Uganda", "Albania", "Myanmar", "Poland", "United States"]
df_filtered = df_filtered[df_filtered["Country Name"].isin(countries_to_keep)]

# 4. Drop rows with any NaN values
df_filtered = df_filtered.dropna()

# 5. Save the cleaned and filtered data
df_filtered.to_csv("Cleaned_Unemployment_GDP.csv", index=False)

# Verification
print(f"Countries remaining: {df_filtered['Country Name'].unique()}")
print(df_filtered.head())