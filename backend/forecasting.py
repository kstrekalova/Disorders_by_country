import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# 1. SETTINGS & LOAD
file_path = r'C:\Users\mailm\Downloads\Disorders_by_country\backend\Mental health Depression disorder Data.csv'
df = pd.read_csv(file_path, low_memory=False)

disorder_cols = [
    'Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 
    'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 
    'Alcohol use disorders (%)'
]

# 2. CLEANING
df_clean = df.copy()
df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
for col in disorder_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean = df_clean.dropna(subset=['Depression (%)', 'Anxiety disorders (%)'], how='all')


# --- GRAPH 1: HISTORICAL DATA ONLY ---
def show_historical_data(country_name):
    country_df = df_clean[df_clean['Entity'] == country_name].copy()
    country_df = country_df.sort_values('Year')
    
    if country_df.empty:
        return print(f"No data for {country_name}")

    melted_df = country_df.melt(id_vars=['Year'], value_vars=disorder_cols, 
                                var_name='Disorder', value_name='Percentage')
    melted_df = melted_df.dropna(subset=['Percentage'])

    fig = px.line(
        melted_df, x='Year', y='Percentage', color='Disorder',
        title=f'HISTORICAL: Mental Health Prevalence in {country_name} (1990-2019)',
        markers=True, template='plotly_white'
    )
    fig.show()

# --- GRAPH 2: PREDICTIVE FORECAST ONLY ---
def show_predictive_forecast(country_name):
    country_df = df_clean[df_clean['Entity'] == country_name].copy()
    country_df = country_df.sort_values('Year')
    
    if country_df.empty:
        return

    fig = go.Figure()
    max_year = int(country_df['Year'].max())
    future_years = np.arange(max_year + 1, max_year + 28).reshape(-1, 1)

    for col in disorder_cols:
        valid_df = country_df[['Year', col]].dropna()
        X, y = valid_df[['Year']].values, valid_df[col].values
        
        if len(X) > 2:
            # Model Training
            model = LinearRegression().fit(X, y)
            future_preds = model.predict(future_years)
            
            # Forecast Line (Dashed)
            fig.add_trace(go.Scatter(
                x=future_years.flatten(), y=future_preds,
                mode='lines', name=f"Predicted {col}",
                line=dict(dash='dash', width=3)
            ))

    fig.update_layout(
        title=f'FORECAST: Predicted Prevalence in {country_name} (2020-2047)',
        xaxis_title="Future Year", yaxis_title="Predicted %",
        template='plotly_dark', # Using dark mode for the forecast makes it look "futuristic"
        hovermode="x unified"
    )
    fig.show()

# 3. RUN BOTH
if __name__ == "__main__":
    target_country = input("What country would you like to model? ")
    
    # This will open two separate browser tabs
    show_historical_data(target_country)
    show_predictive_forecast(target_country)