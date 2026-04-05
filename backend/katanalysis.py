# Data from https://www.kaggle.com/datasets/thedevastator/uncover-global-trends-in-mental-health-disorder

# Entity: Unique identifier for each country or region included in the data set. (String)
# Code: Unique code associated with an Entity/Country or region included in the data set. (String)
# Year: Year that the data about that particular Entity/Country was collected. (Integer)
# Schizophrenia (%): Percentage of people with schizophrenia in that country/region during that year. (Float)
# Bipolar disorder (%): Percentage of people with bipolar disorder in that country/region during that year. (Float)
# Eating disorders (%): Percentage of people with eating disorders in that country/region during that year. (Float)
# Anxiety disorders (%): Percentage of people with anxiety disorders in that country/region during that year. (Float)
# Drug use disorders (%): Percentage of people with drug use disorders in that country/region during that year. (Float)
# Depression (%): Percentage of people with depression in that country/region during that year. (Float)
# Alcohol use disorders (%): Percentage of people with alcohol use disorders in that country/region during that year. (Float)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import country_converter as coco


### Idea: disorders by country over time

raw = pd.read_csv('Mental_health_data.csv', dtype=str)  # read everything as string first

# Find the row index where the second header appears
# It'll have "Entity" in the Entity column again (a dead giveaway)
second_header_mask = raw['Entity'] == 'Entity'
second_header_idx = raw[second_header_mask].index[0]

print(f"Second header found at row: {second_header_idx}")  # sanity check

# df = pd.read_csv("Mental_health_data.csv")

# ---- DATASET 1: Disorder prevalence ----
df_disorders = raw.iloc[:second_header_idx].copy()
df_disorders = df_disorders.drop(columns=['index'])
df_disorders.columns = ['Entity', 'Code', 'Year', 'Schizophrenia', 
                         'Bipolar', 'Eating', 'Anxiety', 
                         'Drug Use', 'Depression', 'Alcohol Use']

# Convert numeric columns from string
numeric_cols = ['Year', 'Schizophrenia', 'Bipolar', 'Eating', 
                'Anxiety', 'Drug Use', 'Depression', 'Alcohol Use']
df_disorders[numeric_cols] = df_disorders[numeric_cols].apply(pd.to_numeric, errors='coerce')

print("Dataset 1 shape:", df_disorders.shape)
print(df_disorders.head(3))

# Filter to real countries only (3-letter ISO code)
df_disorders = df_disorders[df_disorders['Code'].notna() & (df_disorders['Code'].str.len() == 3)]



# ## Check everything:
# print(df_disorders.shape)           # How many rows & columns?
# print(df_disorders.dtypes)          # Are types correct?
# print(df_disorders.describe())      # Min/max/mean for each disorder
# print(df_disorders.isnull().sum())  # Where are the nulls?
# print(df_disorders['Year'].min(), df_disorders['Year'].max())  # Year range? 1990-2017
# print(df_disorders['Entity'].nunique())  # How many unique countries? 195

# Check for duplicates
# print(df_disorders.duplicated(subset=['Entity', 'Year']).sum()) # None

# #No nulls! Hooray.




## Check for impossible values (negative or > 100)
disorder_cols = ['Schizophrenia', 'Bipolar', 'Eating', 
                 'Anxiety', 'Drug Use', 'Depression', 'Alcohol Use']
# for col in disorder_cols:
#     bad = df_disorders[(df_disorders[col] < 0) | (df_disorders[col] > 100)]
#     if len(bad) > 0:
#         print(f"{col}: {len(bad)} suspicious rows")
#         print(bad[['Entity', 'Year', col]])

# # Check the actual max values per disorder
# print(df_disorders[disorder_cols].max())
# print(df_disorders[disorder_cols].min())
# #cool



### Add region column
df_disorders['Region'] = coco.convert(
    names=df_disorders['Code'].tolist(), 
    to='continent'
)

# print(df_disorders['Region'].value_counts())



### Melt to Long Format:
df_long = df_disorders.melt(
    id_vars=['Entity', 'Code', 'Year', 'Region'],
    value_vars=disorder_cols,
    var_name='Disorder',
    value_name='Prevalence'
)

print(df_long.shape)
print(df_long.head())



#================================================================
## Explanatory analysis
# 1. Global average prevalence per disorder (which is most common?)
df_long.groupby('Disorder')['Prevalence'].mean().sort_values(ascending=False)

# 2. Which countries have the highest average depression rate?
(df_disorders.groupby('Entity')['Depression']
 .mean()
 .sort_values(ascending=False)
 .head(10))

# 3. How has each disorder trended globally over time?
(df_long.groupby(['Year', 'Disorder'])['Prevalence']
 .mean()
 .reset_index()
 .sort_values('Year'))

# 4. Correlation between disorders — do they co-occur?
df_disorders[disorder_cols].corr().round(2)








#===================================================
### Idea: correlation between disorders in country


#===================================================
### Idea: disorder trends thru global shock events
# Shocks to annotate:
    # 2008 Financial Crisis
    # 2015 Refugee Crisis
    # 2020 COVID-19
    # Regional conflicts (Syria 2011, Ukraine 2022)

