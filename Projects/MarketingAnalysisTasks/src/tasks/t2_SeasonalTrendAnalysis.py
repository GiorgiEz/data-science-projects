import matplotlib.pyplot as plt



# Task 2
def seasonal_trend_analysis(customers_df, transactions_df, merged_df):
    """ Objective: Identify seasonal patterns in sales """
    #  1. Analyze monthly and weekly sales patterns

    transactions_df['month'] = transactions_df['date'].dt.to_period('M')
    transactions_df['week'] = transactions_df['date'].dt.to_period('W')
    transactions_df['quarter'] = transactions_df['date'].dt.to_period('Q')

    monthly_sales = transactions_df.groupby('month')['amount'].sum()
    weekly_sales = transactions_df.groupby('week')['amount'].sum()

    print(monthly_sales, '\n')
    print(weekly_sales, '\n')
    
    # Plot monthly sales
    monthly_sales.plot(kind='line', figsize=(12, 8))
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.grid()
    plt.show()

    #  2. Calculate seasonality indices
    overall_avg = transactions_df['amount'].mean()
    quarterly_avg = transactions_df.groupby('quarter')['amount'].mean()
    seasonal_indices = quarterly_avg / overall_avg
    print("Seasonal indices: \n", seasonal_indices, '\n')
    
    #  3. Identify peak shopping periods
    peak_quarter = seasonal_indices.idxmax()
    print("Peak Period: ", peak_quarter, '\n')

    #  4. Compare patterns across product categories
    category_sales = transactions_df.groupby('category')['amount'].sum()
    category_sales.plot(kind='line', figsize=(12, 8))
    plt.title('Sales Trends by Product Category')
    plt.xlabel('Category')
    plt.ylabel('Total Sales')
    plt.grid()
    plt.legend(title='Category')
    plt.show()
