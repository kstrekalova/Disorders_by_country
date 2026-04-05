import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOAD DATA
file_path = r'C:\Users\mailm\Downloads\Disorders_by_country\backend\Master_Merged_Dataset.csv'
df = pd.read_csv(file_path, low_memory = False)

# 2. TREND PLOT: Depression Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Year', y='Depression (%)', hue='Entity', marker='o')
plt.title('Depression Trends (1990-2019)')
plt.ylabel('Depression Prevalence (%)')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('depression_trends_over_time.png')

# 3. CORRELATION HEATMAP: Finding the "Reason"
plt.figure(figsize=(8, 6))
# Selecting core socio-economic factors
correlation_cols = ['Depression (%)', 'hdi', 'Unemployment Rate', 'GDP (in USD)']
corr_matrix = df[correlation_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Drivers of Depression: Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

# 4. REGRESSION PLOTS: Relationship Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Relationship with Human Development
sns.regplot(ax=axes[0], data=df, x='hdi', y='Depression (%)', color='blue')
axes[0].set_title('HDI vs. Depression Prevalence')

# Relationship with Economic Stress (Unemployment)
sns.regplot(ax=axes[1], data=df, x='Unemployment Rate', y='Depression (%)', color='green')
axes[1].set_title('Unemployment Rate vs. Depression Prevalence')

plt.tight_layout()
plt.savefig('factor_relationships.png')
plt.show()