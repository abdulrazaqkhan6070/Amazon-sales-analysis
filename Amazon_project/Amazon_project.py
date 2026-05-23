# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv(r"F:\amazon_products.csv")

print('\nit will print the top 10 rows to collect some information  ')
print(df.head(10))

print("\nto print a single column as it is not visible")
print(df['category_id'])

print(df.info())
print("\nBasic Statistics:\n")

# Avoid scientific notation
pd.options.display.float_format = '{:,.2f}'.format
print(df.describe())


# ==============================
# 3. DATA CLEANING
# ==============================

# Drop unnecessary columns
df = df.drop(['imgUrl', 'productURL'], axis=1)

# Fill missing title (only 1 missing)
df['title'] = df['title'].fillna(
    'Winfield 2 Hardside Expandable Luggage'
) # ln line 31 I have just tried random  title name

print("\nunique value ")
print(df.nunique())

# Handle extreme values (log transform price)
df['price'] = np.log1p(df['price'])

# Remove duplicate products
df = df.drop_duplicates(subset='title') 

# Fix listPrice (replace 0 with NaN, then fill)
df['listPrice'] = df['listPrice'].replace(0, np.nan)
df['listPrice'] = df['listPrice'].fillna(df['price'])

# Convert important columns to numeric
cols = ['price', 'stars', 'reviews']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
#error = coerce used if the datatype not converted to numeric than it will give NaN instead of returnng error


print("\nAfter Cleaning:\n")
print(df.info())


# ==============================
# 4. FEATURE ENGINEERING
# ==============================

df['popularity'] = df['reviews'] * df['stars']
df['demandScore'] = df['boughtInLastMonth'] * df['stars']
df['revenue_potential'] = df['price'] * df['demandScore']

df['Discount_percent'] = (
    (df['listPrice'] - df['price']) / df['listPrice']
) * 100

# Handle infinite values
df['Discount_percent'] = df['Discount_percent'].replace(
    [np.inf, -np.inf], np.nan
)


# ==============================
# 5. BUSINESS QUESTIONS
# ==============================

# Q1: Top categories by revenue potential
top_category = (
    df.groupby('category_id')['revenue_potential']
    .mean()
    .sort_values(ascending=False)
    .head()
    .reset_index()
)

print("\nTop Categories by Revenue Potential:\n", top_category)


# Q2: Discount vs Demand
discount_vs_demand = (
    df[df['Discount_percent'].notna()] 
    .groupby(pd.cut(df['Discount_percent'], bins=5))['demandScore']
    .mean()
    .reset_index()
)
#notna() is used to show that which value is not missing (true-> value exist else not)
print("\nDiscount vs Demand:\n", discount_vs_demand)


# Q3: Stars vs Demand
stars_vs_demand = (
    df.groupby(pd.cut(df['stars'], bins=5))['demandScore']
    .mean()
    .reset_index()
)

print("\nStars vs Demand:\n", stars_vs_demand)


# ==============================
# 6. CORRELATION ANALYSIS
# ==============================

corr_matrix = df[['price','stars','reviews',
                  'demandScore','revenue_potential',
                  'Discount_percent']].corr()


print("from this table we can clearly see that demand score and revenuce potential has nearly perfect correlation and the other column operate independently")
print("\nCorrelation Matrix:\n", corr_matrix)


# ==============================
# 7. VISUALIZATION
# ==============================

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("relationship among varius instance",dpi =300)
plt.show()

# Scatter: Discount vs Demand
plt.scatter(df['Discount_percent'], df['demandScore'])
plt.xlabel("Discount Percent")
plt.ylabel("Demand Score")
plt.title("Discount vs Demand")
plt.savefig("Discount_vs_Demand.png", dpi=300)
plt.show()


# # ==============================
# # 8. CATEGORY ANALYSIS
# # ==============================

category_rev_mean = (
df.groupby('category_id')['revenue_potential'].mean().sort_values(ascending=False))

print("\nCategory Revenue Mean:\n", category_rev_mean)

# # ==============================
# # 9. SALES SEGMENTATION
# # ==============================

# Create sales segments based on monthly purchases
df['sales_segment'] = pd.cut(
    df['boughtInLastMonth'],
    bins=[-1, 10, 100, 1000, df['boughtInLastMonth'].max() + 1],
    labels=['Low', 'Medium', 'High', 'Top']
)

# Segment-wise performance
segment_summary = df.groupby('sales_segment')[[
    'price', 'stars', 'reviews',
    'Discount_percent', 'revenue_potential'
]].mean()

print("\nSales Segment Summary:\n", segment_summary)


# # ==============================
# # 10. SEGMENT VISUALIZATION


# =========================
# GROUP DATA
# =========================
grouped_df = df.groupby('sales_segment').mean(numeric_only=True)

# =========================
# CREATE FIGURE
# =========================
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(16, 6)
)

# =========================
# SUBPLOT 1:
# PRODUCT METRICS
# =========================

x = np.arange(len(grouped_df.index))
bar_width = 0.18

# Main bars
ax1.bar(
    x - 1.5 * bar_width,
    grouped_df['price'],
    width=bar_width,
    label='Price',
    edgecolor='black'
)

ax1.bar(
    x - 0.5 * bar_width,
    grouped_df['stars'],
    width=bar_width,
    label='Stars',
    edgecolor='black'
)

ax1.bar(
    x + 0.5 * bar_width,
    grouped_df['Discount_percent'],
    width=bar_width,
    label='Discount %',
    edgecolor='black'
)

# Twin axis for reviews
ax1_twin = ax1.twinx()

ax1_twin.bar(
    x + 1.5 * bar_width,
    grouped_df['reviews'],
    width=bar_width,
    label='Reviews',
    alpha=0.7,
    edgecolor='black'
)

# Titles and labels
ax1.set_title(
    'Product Metrics by Segment',
    fontsize=15,
    pad=15,
    fontweight='bold'
)

ax1.set_ylabel(
    'Average Values',
    fontsize=12
)

ax1_twin.set_ylabel(
    'Average Reviews',
    fontsize=12,
    labelpad=15
)

ax1.set_xticks(x)
ax1.set_xticklabels(grouped_df.index, fontsize=11)

# Grid
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='upper left'
)

# =========================
# SUBPLOT 2:
# REVENUE POTENTIAL
# =========================

bars = ax2.bar(
    grouped_df.index,
    grouped_df['revenue_potential'],
    width=0.5,
    edgecolor='black'
)

# Add values on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{height:.0f}',
        ha='center',
        va='bottom',
        fontsize=10
    )

# Title and labels
ax2.set_title(
    'Revenue Potential by Segment',
    fontsize=15,
    pad=15,
    fontweight='bold'
)

ax2.set_ylabel(
    'Average Revenue ($)',
    fontsize=12,
    labelpad=15
)

# Grid
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# # =========================
# CLEAN LAYOUT
# =========================

# Remove extra borders
for ax in [ax1, ax2, ax1_twin]:
    ax.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Better spacing
plt.subplots_adjust(wspace=0.35)

#save figure 
plt.savefig("Average_Rvenue_potential",dpi=300)

# Final display
plt.show()





fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot revenue on the first axis
segment_summary['revenue_potential'].plot(kind='bar', ax=ax1, color='purple', edgecolor='black')
ax1.set_title("Revenue Potential by Segment")
ax1.set_ylabel("Currency Value")

# Plot the smaller metrics on the second axis
cols_to_show = ['stars', 'Discount_percent', 'price']
segment_summary[cols_to_show].plot(kind='bar', ax=ax2, edgecolor='black')
ax2.set_title("Ratings & Pricing Metrics")
ax2.set_ylabel("Scale (0-100)")

plt.tight_layout()
plt.savefig("revenue_potential",dpi=300)
plt.show()


# # ==============================
# # 11. REVENUE SHARE BY SEGMENT
# # ==============================

segment_revenue_share = (
    df.groupby('sales_segment')['revenue_potential'].sum()
    / df['revenue_potential'].sum()
) * 100

print("\nRevenue Share (%):\n", segment_revenue_share)

plt.figure(figsize=(6,6))
plt.pie(
    segment_revenue_share,
    labels=segment_revenue_share.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Revenue Share by Sales Segment")
plt.savefig("revenue_share%",dpi=300)

plt.show()


# # ==============================
# # 12. BEST SELLER ANALYSIS
# # ==============================

best_seller_comparison = df.groupby('isBestSeller')[
    'boughtInLastMonth'
].mean()

print("\nBest Seller vs Non-Best Seller:\n", best_seller_comparison)

best_seller_comparison.plot(
    kind='bar',
    edgecolor='black'
)

plt.title("Average Monthly Sales (Best Seller vs Non)")
plt.ylabel("Bought In Last Month")
plt.xticks(rotation=0)
plt.savefig("BestSeller_analysis",dpi=300)
plt.show()


# ==============================
# 13. REVIEWS BY SEGMENT
# ==============================

reviews_segment = df.groupby('sales_segment')['reviews'].mean()

print("\nAverage Reviews by Segment:\n", reviews_segment)

reviews_segment.plot(
    kind='bar',
    edgecolor='black'
)

plt.title("Average Reviews by Sales Segment")
plt.ylabel("Reviews")
plt.xticks(rotation=0)
plt.savefig("Average_ReviewBySegment,dpi=300 ")
plt.show()


# # ==============================
# # 14. TOP PRODUCTS ANALYSIS
# # ==============================

print("\nTop 10 Products by Reviews:\n")
print(
    df[['title', 'reviews']]
    .sort_values(by='reviews', ascending=False)
    .head(10)
)

print("\nTop Products (High Discount + High Sales):\n")
print(
    df[['title', 'Discount_percent', 'boughtInLastMonth']]
    .sort_values(by=['Discount_percent','boughtInLastMonth'],
                 ascending=[False, False])
    .head(10)
)


# ==============================
# 15. FINAL BUSINESS INSIGHTS
#  ==============================

print("\n--- FINAL INSIGHTS ---")

print("""
1. A small number of high-performing products (Top segment)
   generate the majority of revenue.

2. DemandScore is the strongest driver of revenue,
   confirming a strong positive correlation.

3. Discounts alone do not significantly increase demand.

4. Ratings (stars) have minimal direct impact on revenue.

5. Best Seller products consistently outperform others
   in terms of sales volume.

6. Revenue is concentrated in a few categories,
   suggesting focused business strategy.
""")  