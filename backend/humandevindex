import pandas as pd
import re

# 1. LOAD DATA
# Replace with your actual file name
file_path = r'C:\Users\mailm\Downloads\Disorders_by_country\backend\HDI.csv'
df_hdi = pd.read_csv(file_path, low_memory=False)

# 2. IDENTIFY THE COLUMNS
# We only use 'iso3' and 'country' as identifiers to avoid dropping rows with NaN in region/hdicode
id_vars = ['iso3', 'country']

# Capture all columns that follow the 'Metric_YYYY' pattern (e.g., hdi_1990)
year_cols = [c for c in df_hdi.columns if re.search(r'_\d{4}$', c)]

# 3. TRANSFORM (Melt)
# This turns each year column (hdi_1990, hdi_1991...) into rows
df_melted = df_hdi.melt(id_vars=id_vars, value_vars=year_cols, 
                        var_name='Metric_Year', value_name='Value')

# 4. SPLIT METRIC AND YEAR
# rsplit('_', n=1) splits at the LAST underscore to handle names like 'gnipc_f_2021'
df_split = df_melted['Metric_Year'].str.rsplit('_', n=1, expand=True)
df_melted['Metric'] = df_split[0]
df_melted['Year'] = df_split[1]

# Ensure Year is numeric for proper sorting and modeling
df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

# 5. PIVOT (Final Transformation)
# This creates a column for each metric (hdi, le, gnipc, etc.) per country and year
df_tidy = df_melted.pivot_table(
    index=['iso3', 'country', 'Year'], 
    columns='Metric', 
    values='Value'
).reset_index()

# Remove the 'Metric' title from the columns for a clean header
df_tidy.columns.name = None

# 6. SIMPLIFY (Keep only country, Year, and hdi)
df_simple = df_tidy[['country', 'Year', 'hdi']]

# Drop rows where hdi is missing to keep the data clean for your predictive model
df_simple = df_simple.dropna(subset=['hdi'])

# Final Save
df_simple.to_csv('HDI_Simplified.csv', index=False)

print("Fix complete! HDI_Simplified.csv has been created.")
print(df_simple.head(10))