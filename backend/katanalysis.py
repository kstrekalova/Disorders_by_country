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
import country_converter as coco
from IPython.display import display
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np


### Idea: disorders by country over time

raw = pd.read_csv('Mental health Depression disorder Data.csv', dtype=str)  # read everything as string first

# Find the row index where the second header appears
# It'll have "Entity" in the Entity column again (a dead giveaway)
second_header_mask = raw['Entity'] == 'Entity'
second_header_idx = raw[second_header_mask].index[0]

print(f"Second header found at row: {second_header_idx}")  # sanity check


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

# print(df_long.shape)
# print(df_long.head())



#================================================================
## Explanatory analysis
# 1. Global average prevalence per disorder (which is most common?)
# df_long.groupby('Disorder')['Prevalence'].mean().sort_values(ascending=False)

# 2. Which countries have the highest average depression rate?
# (df_disorders.groupby('Entity')['Depression']
#  .mean()
#  .sort_values(ascending=False)
#  .head(10))

# # 3. How has each disorder trended globally over time?
# (df_long.groupby(['Year', 'Disorder'])['Prevalence']
#  .mean()
#  .reset_index()
#  .sort_values('Year'))

# # 4. Correlation between disorders — do they co-occur?
# df_disorders[disorder_cols].corr().round(2)


# 1. Global average prevalence per disorder
print("\n--- 1. Average Prevalence by Disorder ---")
display(
    df_long.groupby('Disorder')['Prevalence']
    .mean()
    .sort_values(ascending=False)
    .round(4)
    .reset_index()
)
# Conclusions: Anxiety (3.95%) and Depression (3.47%) are in the lead. 
# Note: Everything else is <1%, keep separate or use log scale.


# 2. Top 10 countries by average depression rate
print("\n--- 2. Top 10 Countries by Depression ---")
display(
    df_disorders.groupby('Entity')['Depression']
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .round(4)
    .reset_index()
)

print("\n--- 2. Bottom 10 Countries by Depression ---")
display(
    df_disorders.groupby('Entity')['Depression']
    .mean()
    .sort_values(ascending=True)
    .head(10)
    .round(4)
    .reset_index()
)
# Conclusions: 
# #1-5: Greenland (Extreme seasonality, isolation, high suicide rates, and indigenous community trauma),
#       Morocco, Lesotho, Uganda, Finland (tho Finland consistently tops "World Happiness" rankings)
#   Lower-income countries with high rates suggest high prevalence, likely low treatment


# 3. Global disorder trends over time
print("\n--- 3. Global Trends Over Time ---")
display(
    df_long.groupby(['Year', 'Disorder'])['Prevalence']
    .mean()
    .round(4)
    .reset_index()
    .sort_values(['Disorder', 'Year'])
)
# Conclusions: 
# No disorder exploded or collapsed globally over 27 years
# Small changes (like Schizophrenia 0.2102 → 0.2110) are essentially flat
# Dataset ends at 2017, so it misses COVID entirely — worth noting as a limitation

# 4. Correlation between disorders
print("\n--- 4. Disorder Correlation Matrix ---")
display(df_disorders[disorder_cols].corr().round(2))
# Conclusions: 
# Strong cluster 1 — "Psychotic/Neurological":
#   Eating ↔ Bipolar (0.70)
#   Eating ↔ Schizophrenia (0.69)
#   Bipolar ↔ Anxiety (0.65)

# Strong cluster 2 — "Stress/Substance":
#   Anxiety ↔ Drug Use (0.60)
#   Anxiety ↔ Eating (0.68)

# Loner: Depression (max correlation = 0.34 with Anxiety)
# Loner: Alcohol Use (near-zero or negative with almost everything)

# Key surprises:
# Depression is nearly uncorrelated with most disorders — this is counterintuitive but suggests Depression has a very different geographic/cultural distribution than the others
# Alcohol Use is negatively correlated with Anxiety (-0.16) and Drug Use (-0.16) — countries high in one tend to be lower in the other, suggesting cultural substitution (e.g. alcohol-heavy cultures vs drug-use-heavy ones)
# Eating disorders cluster with Schizophrenia and Bipolar — likely a shared geographic pattern (higher prevalence in wealthier/Western countries)


#===================================================
### Idea: merge in HDI_simplified, see if there's stuff

# Load & inspect
df_hdi = pd.read_csv('HDI_simplified.csv')

print("Shape:", df_hdi.shape)
print("\nColumns:", df_hdi.columns.tolist())
print("\nDtypes:\n", df_hdi.dtypes)
print("\nFirst few rows:")
display(df_hdi.head())
print("\nNulls:\n", df_hdi.isnull().sum())
print("\nYear range:", df_hdi['Year'].min(), "→", df_hdi['Year'].max())  # adjust col name if needed

# Clean & align
# Rename to match disorders dataframe
df_hdi = df_hdi.rename(columns={
    'country': 'Entity',
    'hdi':     'HDI_Value'
})

# Trim whitespace 
df_hdi['Entity']      = df_hdi['Entity'].str.strip()
df_disorders['Entity'] = df_disorders['Entity'].str.strip()

# Check year overlap (MH is 1990-2017, HDI is 1990-2021 — should be full overlap)
hdi_years = set(df_hdi['Year'].unique())
mh_years  = set(df_disorders['Year'].unique())
print("Overlapping years:", len(hdi_years & mh_years), "years")
print("MH years NOT in HDI:", sorted(mh_years - hdi_years))  # should be empty
# good

# Check country name mismatches BEFORE merging
hdi_countries = set(df_hdi['Entity'].unique())
mh_countries  = set(df_disorders['Entity'].unique())

only_in_hdi = hdi_countries - mh_countries
only_in_mh  = mh_countries - hdi_countries

print(f"\n{len(hdi_countries)} countries in HDI")
print(f"{len(mh_countries)} countries in MH dataset")
print(f"\nIn MH but NOT in HDI ({len(only_in_mh)}):", sorted(only_in_mh))

# Find what HDI_simplified actually calls these countries
problem_countries = [
    'Bolivia', 'Brunei', 'Cape Verde', "Cote d'Ivoire", 'Czech Republic',
    'Democratic Republic of Congo', 'Iran', 'Laos', 'Macedonia',
    'Micronesia (country)', 'Moldova', 'Palestine', 'Russia', 'South Korea',
    'Swaziland', 'Syria', 'Tanzania', 'Timor', 'Venezuela', 'Vietnam'
]

print("=== HDI name lookup ===")
for country in problem_countries:
    keyword = country.split()[0]  # search by first word
    matches = df_hdi[df_hdi['Entity'].str.contains(keyword, case=False, na=False)]['Entity'].unique()
    if len(matches) > 0:
        print(f"  {country:35} → HDI has: {list(matches)}")
    else:
        print(f"  {country:35} → ❌ NOT FOUND")

# Update names of countries
name_fixes_v2 = {
    # Previously confirmed
    'Bolivia (Plurinational State of)':          'Bolivia',
    'Brunei Darussalam':                         'Brunei',
    'Czechia':                                   'Czech Republic',
    'Congo (Democratic Republic of the)':        'Democratic Republic of Congo',
    'Iran (Islamic Republic of)':                'Iran',
    'North Macedonia':                           'Macedonia',
    'Micronesia (Federated States of)':          'Micronesia (country)',
    'Moldova (Republic of)':                     'Moldova',
    'Palestine, State of':                       'Palestine',
    'Russian Federation':                        'Russia',
    'Syrian Arab Republic':                      'Syria',
    'Tanzania (United Republic of)':             'Tanzania',
    'Timor-Leste':                               'Timor',
    'Venezuela (Bolivarian Republic of)':        'Venezuela',

    # NOT FOUND fixes
    'Cabo Verde':                                'Cape Verde',       # renamed
    "Côte d'Ivoire":                             "Cote d'Ivoire",   # accent
    "Lao People's Democratic Republic":          'Laos',            # was hiding in Congo search!
    'Lao PDR':                                   'Laos',            # backup alias
    'Eswatini':                                  'Swaziland',       # renamed 2018
    'Viet Nam':                                  'Vietnam',         # spacing

    # South Korea fix (search returned wrong results)
    'Korea (Republic of)':                       'South Korea',
    'Korea, Republic of':                        'South Korea',
}

# Apply fixes to HDI BEFORE merging
df_hdi['Entity'] = df_hdi['Entity'].replace(name_fixes_v2)

# Re-merge
df_enriched = pd.merge(
    df_disorders,
    df_hdi[['Entity', 'Year', 'HDI_Value']],
    on=['Entity', 'Year'],
    how='left'
)

total   = len(df_enriched)
matched = df_enriched['HDI_Value'].notna().sum()
print(f"Merge quality: {matched:,}/{total:,} rows matched ({matched/total*100:.1f}%)")

# Check who's still unmatched
still_missing = (df_enriched[df_enriched['HDI_Value'].isna()]['Entity']
                 .value_counts())
print("\nStill unmatched:\n", still_missing)


## Result of cleaning: ~14% unmatched rows, but 5 are name issues, let's fix:

for keyword in ['Cabo', 'Ivoire', 'Eswatini', 'Viet', 'Lao']:
    matches = df_hdi[df_hdi['Entity'].str.contains(keyword, case=False, na=False)]['Entity'].unique()
    print(f"'{keyword}' → {matches}")

patch_fixes = {
    'Cabo Verde':                        'Cape Verde',
    'Eswatini (Kingdom of)':                          'Swaziland',
    'Viet Nam':                          'Vietnam',
    "Lao People's Democratic Republic":  'Laos',
    'Lao PDR':                           'Laos',
    "Côte d'Ivoire":                     "Cote d'Ivoire",
}

df_hdi['Entity'] = df_hdi['Entity'].replace(patch_fixes)

# Re-merge
df_enriched = pd.merge(
    df_disorders,
    df_hdi[['Entity', 'Year', 'HDI_Value']],
    on=['Entity', 'Year'],
    how='left'
)

total   = len(df_enriched)
matched = df_enriched['HDI_Value'].notna().sum()
print(f"New merge quality: {matched:,}/{total:,} ({matched/total*100:.1f}%)")


# PROBLEM: Greenland was #1 depression country — but it has zero HDI coverage
# Check HDI coverage for your actual top/bottom countries
depression_avg = (df_enriched.groupby('Entity')['Depression']
                  .mean()
                  .sort_values(ascending=False)
                  .reset_index())

# Add HDI coverage info
hdi_coverage = (df_enriched.groupby('Entity')['HDI_Value']
                .apply(lambda x: x.notna().sum())
                .reset_index()
                .rename(columns={'HDI_Value': 'HDI_Years'}))

depression_avg = pd.merge(depression_avg, hdi_coverage, on='Entity')

print("=== TOP 10 Depression + HDI Coverage ===")
display(depression_avg.head(10))

print("\n=== BOTTOM 10 Depression + HDI Coverage ===")
display(depression_avg.tail(10))

# Filter: only countries with at least 20 years of HDI data
well_covered = depression_avg[depression_avg['HDI_Years'] >= 20]

top3    = well_covered.head(3)['Entity'].tolist()
bottom3 = well_covered.tail(3)['Entity'].tolist()

print("Top 3 (with HDI coverage):   ", top3)
print("Bottom 3 (with HDI coverage):", bottom3)




### Summary table:
focus   = ['Morocco', 'Lesotho', 'Uganda', 'Poland', 'Myanmar', 'Albania']
df_focus = df_enriched[df_enriched['Entity'].isin(focus)].copy()

summary = (df_focus.groupby('Entity')
           .agg(
               Avg_Depression = ('Depression', 'mean'),
               Avg_HDI        = ('HDI_Value',  'mean'),
               Avg_Anxiety    = ('Anxiety',    'mean'),
               Avg_Drug_Use   = ('Drug Use',   'mean'),
               Avg_Alcohol    = ('Alcohol Use','mean'),
           )
           .round(3)
           .sort_values('Avg_Depression', ascending=False)
           .reset_index())

summary['Group'] = summary['Entity'].apply(
    lambda x: 'Top 3' if x in ['Morocco', 'Lesotho', 'Uganda'] else 'Bottom 3'
)

display(summary[['Group', 'Entity', 'Avg_Depression', 'Avg_HDI',
                 'Avg_Anxiety', 'Avg_Drug_Use', 'Avg_Alcohol']])

# Results: 
# The HDI story is real but incomplete: Myanmar (HDI 0.450) which is almost identical 
# to Uganda (0.429) and Lesotho (0.469), yet has depression at 2.25% vs their 5.2–5.4%. 
# HDI alone clearly doesn't explain it.

# Morocco is an Outlier Within the Top 3.
# Morocco has nearly 5x higher drug use than Uganda and anxiety almost 1.5 points higher 
# than the others. Its depression likely has a different driver (stress/substance co-occurrence)
# versus Lesotho and Uganda which may be more poverty/trauma driven.

# Alcohol is doing the opposite of what we'd expect.
# The two highest alcohol countries are Poland (1.952) and Albania (1.747) — both in the low 
# depression group. Meanwhile Morocco, the highest depression country, has the lowest alcohol (0.563).
# This echoes the negative correlation (-0.16 overall) we saw in the correlation matrix earlier. 
# Cultural substitution effect is real here — alcohol-heavy cultures tend to have lower reported 
# drug use and depression.

# Anxiety is surprisingly weak as a differentiator.
# Except for Morocco, anxiety rates are clustered tightly between 3.26 and 3.47 across ALL six countries 
# — both high and low depression groups. That's a narrow range. It means anxiety prevalence alone doesn't predict depression ranking here.

# Myanmar breaks every pattern:
# - Low HDI (like the top 3) → but low depression
# - Low anxiety, low drug use, low alcohol
# - Essentially low everything
# might be seriously underreported


#=====================================================================
### Visualizations:
# Scatter plot
# --- Prep: average across all years per country ---
scatter_df = (df_enriched.groupby('Entity')
              .agg(
                  Avg_Depression = ('Depression', 'mean'),
                  Avg_HDI        = ('HDI_Value',  'mean'),
              )
              .round(3)
              .dropna()
              .reset_index())

# --- Define focus groups ---
top3    = ['Morocco', 'Lesotho', 'Uganda']
bottom3 = ['Poland', 'Myanmar', 'Albania']
focus   = top3 + bottom3

bg      = scatter_df[~scatter_df['Entity'].isin(focus)]
foc_top = scatter_df[scatter_df['Entity'].isin(top3)]
foc_bot = scatter_df[scatter_df['Entity'].isin(bottom3)]

fig = go.Figure()

# --- Layer 1: background countries (grey, subtle) ---
fig.add_trace(go.Scatter(
    x=bg['Avg_HDI'],
    y=bg['Avg_Depression'],
    mode='markers',
    name='All Other Countries',
    text=bg['Entity'],
    hovertemplate='<b>%{text}</b><br>HDI: %{x}<br>Depression: %{y}%<extra></extra>',
    marker=dict(color='lightgrey', size=7, opacity=0.6,
                line=dict(color='grey', width=0.5))
))

# --- Layer 2: Top 3 (red) ---
fig.add_trace(go.Scatter(
    x=foc_top['Avg_HDI'],
    y=foc_top['Avg_Depression'],
    mode='markers+text',
    name='Top 3 (Highest Depression)',
    text=foc_top['Entity'],
    textposition='top right',
    hovertemplate='<b>%{text}</b><br>HDI: %{x}<br>Depression: %{y}%<extra></extra>',
    marker=dict(color='crimson', size=13, symbol='circle',
                line=dict(color='darkred', width=1.5))
))

# --- Layer 3: Bottom 3 (green) ---
fig.add_trace(go.Scatter(
    x=foc_bot['Avg_HDI'],
    y=foc_bot['Avg_Depression'],
    mode='markers+text',
    name='Bottom 3 (Lowest Depression)',
    text=foc_bot['Entity'],
    textposition='top right',
    hovertemplate='<b>%{text}</b><br>HDI: %{x}<br>Depression: %{y}%<extra></extra>',
    marker=dict(color='seagreen', size=13, symbol='circle',
                line=dict(color='darkgreen', width=1.5))
))

# Trend line across all countries
import numpy as np
coeffs = np.polyfit(scatter_df['Avg_HDI'], scatter_df['Avg_Depression'], 1)
trendline_x = np.linspace(scatter_df['Avg_HDI'].min(), scatter_df['Avg_HDI'].max(), 100)
trendline_y = np.polyval(coeffs, trendline_x)

fig.add_trace(go.Scatter(
    x=trendline_x,
    y=trendline_y,
    mode='lines',
    name='Trend',
    line=dict(color='steelblue', width=2, dash='dash'),
    hoverinfo='skip'
))

# --- Layout ---
fig.update_layout(
    legend=dict(
        x=0.98,
        y=0.98,
        xanchor='right',
        yanchor='bottom',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='lightgrey',
        borderwidth=1
    )
)

fig.write_html('scatter_hdi_depression.html')
fig.show()
# Results: The trendline is nearly flat (slightly negative slope) 
# which visually undersells the HDI-depression relationship. 
# That's because the overall data is noisy — worth adding a note
# in analysis that the relationship is stronger at the extremes 
# (6 focus countries) than globally.
# 1. HDI is NOT a Strong Global Predictor of Depression
# 2. But the Extremes Tell a Different Story: the 6 highlighted 
#   countries reveal something the trendline hides:
#   - All 3 red dots (Morocco, Lesotho, Uganda) sit in the upper-left — low HDI AND high depression
#   - Poland and Albania sit bottom-right — high HDI AND low depression
#   - That separation is real and meaningful even if the global trend is flat

# 3. Myanmar Breaks Everything: it sits bottom-left alongside the red dots in terms of HDI (~0.45) 
#    but has depression as low as Poland (~2.2%). It's the most important single point on this chart — 
#    it tells you HDI alone cannot explain the top 3's high depression rates. 

# CONCLUSION: HDI separates the extremes but doesn't explain the middle



### Line chart:
# (defined focus)
df_line = df_enriched[df_enriched['Entity'].isin(focus)].copy()

# Color map
colors = {
    'Morocco': 'crimson',
    'Lesotho': 'tomato',
    'Uganda':  'orangered',
    'Poland':  'seagreen',
    'Myanmar': 'mediumseagreen',
    'Albania': 'steelblue'
}
fig = go.Figure()

for country in focus:
    df_c = df_line[df_line['Entity'] == country].sort_values('Year')
    fig.add_trace(go.Scatter(
        x=df_c['Year'],
        y=df_c['Depression'],
        mode='lines+markers',
        name=country,
        line=dict(color=colors[country], width=2.5),
        marker=dict(size=5),
        hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>Depression: %{{y:.2f}}%<extra></extra>'
    ))

fig.update_layout(
    title=dict(
        text='Depression Rate Over Time (1990–2017)<br><sup>Top 3 Highest vs Bottom 3 Lowest Depression Countries</sup>',
        x=0.5
    ),
    xaxis=dict(title='Year', dtick=5),
    yaxis=dict(title='Depression Rate (%)'),
    width=900, height=550,
    legend=dict(x=1.01, y=1, xanchor='left'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_gridcolor='#eeeeee',
    yaxis_gridcolor='#eeeeee',
    # Add a horizontal reference line at the global average
    shapes=[dict(
        type='line',
        x0=1990, x1=2017,
        y0=3.5,  y1=3.5,
        line=dict(color='lightgrey', width=1.5, dash='dot')
    )],
    annotations=[dict(
        x=2017, y=3.5,
        text='Global avg ~3.5%',
        showarrow=False,
        xanchor='right',
        font=dict(color='grey', size=11)
    )]
)

fig.write_html('linechart_depression_trends.html')
fig.show()
# Results:
# - The two groups have stayed roughly 3 percentage points apart from 1990 to 2017 
# with no sign of convergence. This is a structural finding — these aren't 
# temporary fluctuations, these are entrenched patterns.
# - Morocco peaked around 1999–2000 (~5.7%) then slowly declined to ~5.4% by 2017 — slight improvement over time
# - Uganda rose from 5.0% to ~5.5% peaking around 2003, then dropped back to ~4.9% by 2017 — the most movement of any country
# - Lesotho is the most alarming — it has been steadily RISING since 2003 and by 2017 is overtaking both Morocco and Uganda. It's the only country in either group that is clearly getting worse
# - Poland, Myanmar and Albania barely moved across 27 years — their low depression is locked in just as firmly as the top 3's high depression.
# Takeaway (short): The gap is persistent, not closing — and Lesotho's rising trend is the single most concerning finding in this chart and worth highlighting.



# Grouped bar chart chart:
df_radar = df_enriched[df_enriched['Entity'].isin(focus)].copy()

# --- Build normalized summary ---
metrics = ['Depression', 'Anxiety', 'Drug Use', 'Alcohol Use', 'HDI_Value']

radar_df = (df_radar.groupby('Entity')[metrics]
            .mean()
            .round(3)
            .reset_index())

# Normalize each metric 0-1 so all axes are comparable
for col in metrics:
    min_val = radar_df[col].min()
    max_val = radar_df[col].max()
    radar_df[col + '_norm'] = (radar_df[col] - min_val) / (max_val - min_val)

metrics      = ['Depression', 'Anxiety', 'Drug Use', 'Alcohol Use', 'HDI_Value']
metric_labels = ['Depression', 'Anxiety', 'Drug Use', 'Alcohol Use', 'HDI']

colors = {
    'Morocco': 'crimson',
    'Lesotho': 'tomato',
    'Uganda':  'orangered',
    'Poland':  'seagreen',
    'Myanmar': 'mediumseagreen',
    'Albania': 'steelblue'
}

fig = go.Figure()

for _, row in radar_df.iterrows():
    country = row['Entity']
    fig.add_trace(go.Bar(
        name=country,
        x=metric_labels,
        y=[row[m] for m in metrics],
        marker_color=colors[country],
        hovertemplate=f'<b>{country}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>'
    ))

fig.update_layout(
    barmode='group',
    title=dict(
        text='Mental Health & HDI Profile: Top 3 vs Bottom 3 Countries<br><sup>Average 1990–2017 | Actual values (not normalized)</sup>',
        x=0.5
    ),
    xaxis=dict(title='Metric'),
    yaxis=dict(title='Average Value'),
    width=950, height=550,
    legend=dict(x=1.01, y=1, xanchor='left'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_gridcolor='#eeeeee',
    yaxis_gridcolor='#eeeeee',
)

fig.write_html('grouped_bar_profiles.html')
fig.show()
# Results:
# Depression — Clean separation The three red/orange bars tower above the blue/green bars. This is your headline finding, clearly visible at a glance. No ambiguity.
# Anxiety — Morocco is the only outlier Morocco's anxiety bar (~5.0%) is dramatically taller than every other country including Lesotho and Uganda (~3.5%). Everyone else — top 3 AND bottom 3 — clusters between 3.2 and 3.5%. Anxiety does NOT separate the groups. It separates Morocco from everyone else.
# Drug Use — Same story as anxiety Morocco again dominates at ~1.5%. Everyone else sits below 0.85% with no meaningful difference between top and bottom groups. Drug use is a Morocco-specific problem, not a top 3 pattern.
# Alcohol Use — The counterintuitive finding Poland has the HIGHEST alcohol use (~1.95%) and the LOWEST depression. Morocco has the LOWEST alcohol use (~0.6%) and the HIGHEST depression. The relationship is literally inverted.
# HDI — Mostly separates groups but Myanmar breaks it Poland and Albania clearly taller, Uganda and Lesotho clearly shorter — but Myanmar sits identically to Uganda and Lesotho in HDI while matching Poland and Albania in depression.





### One-line summary: Depression and HDI separate the two groups cleanly — but anxiety, drug use and alcohol use reveal that Morocco has an entirely different profile from Lesotho and Uganda, suggesting the top 3 don't share a single common cause.