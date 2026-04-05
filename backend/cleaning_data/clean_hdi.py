import pandas as pd
import re

# 1. LOAD DATA
# Using 'r' to prevent Unicode errors
file_path = r'C:\Users\mailm\Downloads\Disorders_by_country\backend\cleaning_data\HDI.csv'
df_hdi = pd.read_csv(file_path, low_memory=False)

# 2. IDENTIFY THE COLUMNS
id_vars = ['iso3', 'country']
year_cols = [c for c in df_hdi.columns if re.search(r'_\d{4}$', c)]

# 3. TRANSFORM (Melt)
df_melted = df_hdi.melt(id_vars=id_vars, value_vars=year_cols, 
                        var_name='Metric_Year', value_name='Value')

# 4. SPLIT METRIC AND YEAR
df_split = df_melted['Metric_Year'].str.rsplit('_', n=1, expand=True)
df_melted['Metric'] = df_split[0]
df_melted['Year'] = df_split[1]

# Ensure Year and Value are numeric
df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

# 5. PIVOT (Final Transformation)
df_tidy = df_melted.pivot_table(
    index=['iso3', 'country', 'Year'], 
    columns='Metric', 
    values='Value'
).reset_index()

df_tidy.columns.name = None

# 6. SIMPLIFY (Keep only country, Year, and hdi)
df_simple = df_tidy[['country', 'Year', 'hdi']].copy()

# 7. ADD COUNTRY FILTER
# This deletes all rows except for your 7 target countries
countries_to_keep = ["Morocco", "Lesotho", "Uganda", "Albania", "Myanmar", "Poland", "United States"]
df_simple = df_simple[df_simple["country"].isin(countries_to_keep)]

# 8. CLEANUP
# Drop rows where hdi is missing
df_simple = df_simple.dropna(subset=['hdi'])

# Final Save
df_simple.to_csv('clean_hdi', index=False)

print("Filter applied! HDI_Simplified.csv now only contains the selected countries.")
print(f"Countries in file: {df_simple['country'].unique()}")
print(df_simple.head(10))