from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Task 5
def product_category_analysis(customers_df, transactions_df, merged_df):
    """ Objective: Analyze product category performance """

    # 1. Calculate category-wise sales trends
    category_sales = transactions_df.groupby('category').agg(
        total_sales=('amount', 'sum'),
        total_orders=('amount', 'count'),
        avg_order_value=('amount', 'mean')
    ).reset_index()
    print("Category-Wise Sales Trends: \n", category_sales)

    # 2. Identify cross-selling opportunities
    # Create a synthetic basket for each customer
    basket = transactions_df.groupby(['customer_id', 'category'])['amount'].count().unstack().fillna(0)
    # Convert to binary matrix (1 if purchased, 0 otherwise)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Frequent itemsets with a minimum support of 5%
    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

    # Generate association rules with lift > 1
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets))

    # Print the top cross-selling opportunities
    print("Top Cross-Selling Opportunities: \n")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift',
                                                                                             ascending=False).head(10))

    # 3. Analyze category preferences by customer segment
    customer_segments = customers_df[['customer_id', 'membership_tier']]
    category_preferences = transactions_df.merge(customer_segments, on='customer_id', how='left').groupby(
        ['membership_tier', 'category']
    ).agg(total_sales=('amount', 'sum')).reset_index()
    print("Category Preferences by membership_tier: \n", category_preferences)

    # 4. Study seasonal category performance
    transactions_df['month'] = pd.to_datetime(transactions_df['date']).dt.month
    seasonal_performance = transactions_df.groupby(['month', 'category']).agg(
        total_sales=('amount', 'sum')
    ).reset_index()
    print("Seasonal Performance by Category: \n", seasonal_performance)

    # Visualization
    sns.lineplot(data=seasonal_performance, x='month', y='total_sales', hue='category')
    plt.title('Seasonal Performance by Product Category')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.legend(title='Product Category')
    plt.show()

    # Expected Outcome: Category optimization report with actionable insights
    print("Recommendations:")
    print("- Optimize inventory for high-performing categories during peak months.")
    print("- Develop cross-selling strategies based on identified rules.")
    print("- Target specific customer segments with preferred category promotions.")

