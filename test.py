#!/usr/bin/env python3
"""
Comprehensive Data Analysis Example with Visualizations
=======================================================

This script demonstrates key data analysis concepts and visualization techniques
using synthetic datasets. Perfect for learning data analysis with Python!

Key Libraries Used:
- pandas: Data manipulation and analysis
- numpy: Numerical computing and data generation
- matplotlib: Basic plotting and customization
- seaborn: Statistical visualizations
- plotly: Interactive charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Starting Data Analysis Journey!")
print("=" * 50)

# ============================================================================
# SECTION 1: DATA GENERATION & EXPLORATION
# ============================================================================

print("\nSECTION 1: Creating Synthetic Datasets")

# 1.1 Sales Data Generation
print("1.1 Generating Sales Data...")
np.random.seed(42)  # For reproducible results

# Create 1000 sales records
n_records = 1000
data = {
    'date': pd.date_range('2024-01-01', periods=n_records, freq='h'),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_records, p=[0.25, 0.25, 0.20, 0.20, 0.10]),
    'sales_amount': np.random.lognormal(4, 1.5, n_records),  # Log-normal distribution for realistic sales
    'customer_age': np.random.normal(35, 12, n_records),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_records, p=[0.3, 0.25, 0.25, 0.2]),
    'is_weekend': np.random.choice([True, False], n_records, p=[0.3, 0.7])
}

# Clean up data
data['customer_age'] = np.clip(data['customer_age'], 18, 80)
data['sales_amount'] = np.clip(data['sales_amount'], 5, 500)

sales_df = pd.DataFrame(data)
print(f"Created sales dataset with {len(sales_df)} records")

# 1.2 Customer Demographics Data
print("1.2 Generating Customer Demographics...")
customer_data = {
    'customer_id': range(1, 501),
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], 500, p=[0.15, 0.30, 0.25, 0.20, 0.10]),
    'income_level': np.random.choice(['Low', 'Medium', 'High'], 500, p=[0.4, 0.4, 0.2]),
    'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold'], 500, p=[0.5, 0.35, 0.15]),
    'satisfaction_score': np.random.beta(2, 1, 500) * 10  # Beta distribution for satisfaction scores
}

customer_df = pd.DataFrame(customer_data)
print(f"Created customer demographics with {len(customer_df)} customers")

# 1.3 Time Series Data
print("1.3 Generating Time Series Data...")
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
daily_visitors = 1000 + 500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 100, len(dates))
daily_visitors = np.clip(daily_visitors, 200, 2000)

ts_data = pd.DataFrame({
    'date': dates,
    'daily_visitors': daily_visitors.astype(int),
    'conversion_rate': 0.05 + 0.02 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.005, len(dates))
})

print(f"Created time series data with {len(ts_data)} days")

# ============================================================================
# SECTION 2: DATA EXPLORATION & BASIC VISUALIZATIONS
# ============================================================================

print("\nSECTION 2: Data Exploration & Basic Visualizations")

# 2.1 Basic Data Inspection
print("2.1 Basic Data Inspection...")
print("\nSales Data Sample:")
print(sales_df.head())
print(f"\nSales Data Shape: {sales_df.shape}")
print(f"Sales Data Types:\n{sales_df.dtypes}")

print("\nBasic Statistics:")
print(sales_df[['sales_amount', 'customer_age']].describe())

# ============================================================================
# SECTION 3: DISTRIBUTION ANALYSIS
# ============================================================================

print("\nSECTION 3: Distribution Analysis")

# 3.1 Sales Amount Distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Sales Data Distribution Analysis', fontsize=16, fontweight='bold')

# Histogram
axes[0, 0].hist(sales_df['sales_amount'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Sales Amount Distribution')
axes[0, 0].set_xlabel('Sales Amount ($)')
axes[0, 0].set_ylabel('Frequency')

# Box Plot
axes[0, 1].boxplot(sales_df['sales_amount'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
axes[0, 1].set_title('Sales Amount Box Plot')
axes[0, 1].set_ylabel('Sales Amount ($)')

# Violin Plot by Category
sns.violinplot(data=sales_df, x='product_category', y='sales_amount', ax=axes[1, 0])
axes[1, 0].set_title('Sales Distribution by Product Category')
axes[1, 0].tick_params(axis='x', rotation=45)

# Customer Age Distribution
axes[1, 1].hist(sales_df['customer_age'], bins=20, color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Customer Age Distribution')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
print("Saved distribution analysis plot")

# ============================================================================
# SECTION 4: RELATIONSHIP ANALYSIS
# ============================================================================

print("\nSECTION 4: Relationship Analysis")

# 4.1 Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = sales_df[['sales_amount', 'customer_age', 'is_weekend']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix: Sales Data')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved correlation heatmap")

# 4.2 Scatter Plot with Regression
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(sales_df['customer_age'], sales_df['sales_amount'], alpha=0.6, c=sales_df['sales_amount'], cmap='viridis')
plt.colorbar(label='Sales Amount ($)')
plt.xlabel('Customer Age')
plt.ylabel('Sales Amount ($)')
plt.title('Sales vs Customer Age')

plt.subplot(1, 2, 2)
category_sales = sales_df.groupby('product_category')['sales_amount'].mean().sort_values(ascending=False)
plt.bar(category_sales.index, category_sales.values, color='coral', edgecolor='darkred')
plt.title('Average Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average Sales ($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('relationship_analysis.png', dpi=300, bbox_inches='tight')
print("Saved relationship analysis plots")

# ============================================================================
# SECTION 5: TIME SERIES ANALYSIS
# ============================================================================

print("\nSECTION 5: Time Series Analysis")

# 5.1 Daily Visitors Trend
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(ts_data['date'], ts_data['daily_visitors'], linewidth=1, alpha=0.8)
plt.title('Daily Website Visitors (2024)')
plt.xlabel('Date')
plt.ylabel('Daily Visitors')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.plot(ts_data['date'], ts_data['conversion_rate'], color='red', linewidth=1, alpha=0.8)
plt.title('Daily Conversion Rate')
plt.xlabel('Date')
plt.ylabel('Conversion Rate')
plt.xticks(rotation=45)

# 5.2 Monthly Aggregation
ts_data['month'] = ts_data['date'].dt.month
ts_data['day_of_week'] = ts_data['date'].dt.day_name()

monthly_visitors = ts_data.groupby('month')['daily_visitors'].mean()
weekly_visitors = ts_data.groupby('day_of_week')['daily_visitors'].mean()

plt.subplot(2, 2, 3)
plt.bar(monthly_visitors.index, monthly_visitors.values, color='lightblue', edgecolor='navy')
plt.title('Average Monthly Visitors')
plt.xlabel('Month')
plt.ylabel('Average Visitors')

plt.subplot(2, 2, 4)
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_visitors = weekly_visitors.reindex(days_order)
plt.bar(weekly_visitors.index, weekly_visitors.values, color='lightgreen', edgecolor='darkgreen')
plt.title('Average Daily Visitors by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Visitors')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
print("Saved time series analysis plots")

# ============================================================================
# SECTION 6: ADVANCED VISUALIZATIONS
# ============================================================================

print("\nSECTION 6: Advanced Visualizations")

# 6.1 Pair Plot (using seaborn)
print("6.1 Creating Pair Plot...")
sample_df = sales_df[['sales_amount', 'customer_age', 'region', 'product_category']].sample(200)
pair_plot = sns.pairplot(sample_df, hue='region', diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot: Sales Data Relationships', y=1.02)
plt.savefig('pair_plot.png', dpi=300, bbox_inches='tight')
print("Saved pair plot")

# 6.2 3D Scatter Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Sample data for 3D plot
sample_3d = sales_df.sample(100)
x = sample_3d['customer_age']
y = sample_3d['sales_amount']
z = sample_3d.index

scatter = ax.scatter(x, y, z, c=y, cmap='viridis', alpha=0.6)
ax.set_xlabel('Customer Age')
ax.set_ylabel('Sales Amount ($)')
ax.set_zlabel('Record Index')
ax.set_title('3D Scatter Plot: Age vs Sales vs Index')

plt.colorbar(scatter, label='Sales Amount ($)')
plt.savefig('3d_scatter.png', dpi=300, bbox_inches='tight')
print("Saved 3D scatter plot")

# ============================================================================
# SECTION 7: INTERACTIVE VISUALIZATIONS
# ============================================================================

print("\nSECTION 7: Interactive Visualizations")

# 7.1 Interactive Sales Dashboard
print("7.1 Creating Interactive Sales Dashboard...")

# Aggregate data for dashboard
agg_data = sales_df.groupby(['date', 'product_category']).agg({
    'sales_amount': 'sum',
    'customer_age': 'mean'
}).reset_index()

# Interactive line chart
fig = px.line(agg_data, x='date', y='sales_amount', color='product_category',
              title='Interactive Sales Trends by Product Category',
              labels={'sales_amount': 'Sales Amount ($)', 'date': 'Date'})

fig.update_layout(hovermode='x unified')
fig.write_html('interactive_sales_dashboard.html')
print("Saved interactive sales dashboard")

# 7.2 Interactive Scatter Plot
fig2 = px.scatter(sales_df.sample(200), x='customer_age', y='sales_amount', 
                  color='product_category', size='sales_amount',
                  title='Interactive Sales vs Age Analysis',
                  labels={'customer_age': 'Customer Age', 'sales_amount': 'Sales Amount ($)'})

fig2.write_html('interactive_scatter.html')
print("Saved interactive scatter plot")

# ============================================================================
# SECTION 8: CUSTOMER SEGMENTATION
# ============================================================================

print("\nSECTION 8: Customer Segmentation")

# 8.1 Customer Analysis
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
age_group_sales = customer_df.groupby('age_group')['satisfaction_score'].mean()
plt.bar(age_group_sales.index, age_group_sales.values, color='skyblue', edgecolor='navy')
plt.title('Average Satisfaction by Age Group')
plt.ylabel('Satisfaction Score')
plt.xticks(rotation=45)

plt.subplot(2, 3, 2)
income_sales = customer_df.groupby('income_level')['satisfaction_score'].mean()
plt.pie(income_sales.values, labels=income_sales.index, autopct='%1.1f%%', startangle=90)
plt.title('Satisfaction by Income Level')

plt.subplot(2, 3, 3)
loyalty_sales = customer_df.groupby('loyalty_tier')['satisfaction_score'].mean()
plt.bar(loyalty_sales.index, loyalty_sales.values, color='gold', edgecolor='orange')
plt.title('Satisfaction by Loyalty Tier')
plt.ylabel('Satisfaction Score')

plt.subplot(2, 3, 4)
# Combine datasets for cross-analysis
combined_data = sales_df.merge(customer_df, left_on='customer_age', right_on='customer_id', how='inner')
if len(combined_data) > 0:
    region_sales = sales_df.groupby('region')['sales_amount'].mean()
    plt.bar(region_sales.index, region_sales.values, color='lightcoral', edgecolor='red')
    plt.title('Average Sales by Region')
    plt.ylabel('Average Sales ($)')
else:
    plt.text(0.5, 0.5, 'No matching data found', ha='center', va='center')
    plt.title('Region Sales Analysis')

plt.subplot(2, 3, 5)
# Sales by day of week
sales_df['day_of_week'] = sales_df['date'].dt.day_name()
dow_sales = sales_df.groupby('day_of_week')['sales_amount'].mean()
dow_sales = dow_sales.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.plot(dow_sales.index, dow_sales.values, marker='o', linewidth=2, markersize=8)
plt.title('Average Sales by Day of Week')
plt.xticks(rotation=45)
plt.ylabel('Average Sales ($)')

plt.subplot(2, 3, 6)
# Customer age distribution with sales overlay
plt.hist(sales_df['customer_age'], bins=20, alpha=0.7, color='purple', edgecolor='black')
plt.axvline(sales_df['customer_age'].mean(), color='red', linestyle='--', 
           label=f'Mean: {sales_df["customer_age"].mean():.1f}')
plt.axvline(sales_df['customer_age'].median(), color='orange', linestyle='--', 
           label=f'Median: {sales_df["customer_age"].median():.1f}')
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig('customer_segmentation.png', dpi=300, bbox_inches='tight')
print("Saved customer segmentation analysis")

# ============================================================================
# SECTION 9: SUMMARY & INSIGHTS
# ============================================================================

print("\nSECTION 9: Summary & Key Insights")

# Generate summary statistics
print("\nKEY INSIGHTS:")

print(f"1. Total Sales Records: {len(sales_df):,}")
print(f"2. Average Sales Amount: ${sales_df['sales_amount'].mean():.2f}")
print(f"3. Median Sales Amount: ${sales_df['sales_amount'].median():.2f}")
print(f"4. Average Customer Age: {sales_df['customer_age'].mean():.1f} years")
print(f"5. Most Popular Product Category: {sales_df['product_category'].mode()[0]}")

category_performance = sales_df.groupby('product_category')['sales_amount'].agg(['mean', 'sum', 'count'])
best_category = category_performance['sum'].idxmax()
print(f"6. Highest Revenue Category: {best_category} (${category_performance['sum'].max():.2f})")

region_performance = sales_df.groupby('region')['sales_amount'].sum()
best_region = region_performance.idxmax()
print(f"7. Top Performing Region: {best_region} (${region_performance.max():.2f})")

weekend_impact = sales_df.groupby('is_weekend')['sales_amount'].mean()
print(f"8. Weekend vs Weekday Sales: ${weekend_impact[True]:.2f} vs ${weekend_impact[False]:.2f}")

print(f"\nVISUALIZATION FILES CREATED:")
print("   - distribution_analysis.png")
print("   - correlation_heatmap.png") 
print("   - relationship_analysis.png")
print("   - time_series_analysis.png")
print("   - pair_plot.png")
print("   - 3d_scatter.png")
print("   - customer_segmentation.png")
print("   - interactive_sales_dashboard.html")
print("   - interactive_scatter.html")

print("\nData Analysis Complete!")
print("=" * 50)
print("\nTIPS FOR LEARNING:")
print("1. Try modifying the data generation parameters")
print("2. Experiment with different chart types")
print("3. Add your own datasets")
print("4. Practice customizing colors and styles")
print("5. Explore the interactive HTML files in your browser")

# Save summary to text file
with open('analysis_summary.txt', 'w') as f:
    f.write("Data Analysis Summary\n")
    f.write("=" * 30 + "\n\n")
    f.write(f"Total Records: {len(sales_df):,}\n")
    f.write(f"Average Sales: ${sales_df['sales_amount'].mean():.2f}\n")
    f.write(f"Average Age: {sales_df['customer_age'].mean():.1f}\n")
    f.write(f"Best Category: {best_category}\n")
    f.write(f"Best Region: {best_region}\n")

print("Summary saved to analysis_summary.txt")
