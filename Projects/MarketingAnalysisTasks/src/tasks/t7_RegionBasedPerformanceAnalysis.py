import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Task 7
def region_based_performance_analysis(customers_df, transactions_df, merged_df):
    """ Objective: Analyze regional sales patterns """

    # 1. Compare regional performance metrics
    regional_performance = merged_df.groupby('region').agg(
        total_sales=('amount', 'sum'),
        total_transactions=('amount', 'count'),
        avg_order_value=('amount', 'mean'),
        customer_count=('customer_id', 'nunique')
    ).reset_index()
    print("Regional Performance Metrics:\n", regional_performance)

    # 2. Identify region-specific trends
    merged_df['month'] = pd.to_datetime(merged_df['date']).dt.to_period('M')
    merged_df['month_str'] = merged_df['month'].astype(str)
    regional_trends = merged_df.groupby(['region', 'month_str']).agg(
        monthly_sales=('amount', 'sum')
    ).reset_index()

    # Plot trends
    sns.lineplot(data=regional_trends, x='month_str', y='monthly_sales', hue='region')
    plt.title("Monthly Sales Trends by Region")
    plt.xticks(rotation=45)
    plt.show()

    # 3. Analyze customer preferences by region
    regional_category_sales = merged_df.groupby(['region', 'category']).agg(
        category_sales=('amount', 'sum')
    ).reset_index()
    regional_category_sales['sales_percentage'] = (
        regional_category_sales.groupby('region')['category_sales'].transform(lambda x: x / x.sum())
    )
    print("Regional Category Preferences:\n", regional_category_sales)

    # 4. Create regional targeting strategies
    underperforming_regions = regional_performance[
        regional_performance['total_sales'] < regional_performance['total_sales'].mean()]

    print("Underperforming Regions:\n", underperforming_regions, '\n')
