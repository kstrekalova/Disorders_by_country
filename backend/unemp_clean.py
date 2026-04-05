import pandas as pd

# Load your CSV
file_path = r'C:/Users/janicewang/Desktop/depression_data/Disorders_by_country/backend/Employment_Unemployment_GDP_data.csv'
df = pd.read_csv(file_path, low_memory=False)

# Keep only desired columns
columns_to_keep = ["Country Name", "Year", "Unemployment Rate", "GDP (in USD)"]
df_filtered = df[columns_to_keep]

# Drop rows with any NaN values
df_filtered = df_filtered.dropna()

# Optional: save the cleaned data to a new CSV
df_filtered.to_csv("Cleaned_Employment_Unemployment_GDP.csv", index=False)

print(df_filtered.head())