import matplotlib.pyplot as plt
import seaborn as sns


# Task 3
def channel_performance_optimization(customers_df, transactions_df, merged_df):
    """ Objective: Analyze performance across different sales channels """
    # 1. Compare conversion rates by channel
    conversion_rates = transactions_df.groupby('channel')[['customer_id']].count() / len(transactions_df)
    print("Conversion rates by channel: \n", conversion_rates, '\n')

    # 2. Analyze average order value by channel
    average_order_values = transactions_df.groupby('channel')['amount'].agg(['sum', 'count'])
    average_order_values['average_order_value'] = average_order_values['sum'] / average_order_values['count']
    print("Average order values by channel: \n", average_order_values[['average_order_value']], '\n')

    # 3. Study customer channel preferences
    channel_preference_counts = customers_df.groupby('channel_preference')['customer_id'].count().reset_index(name='channel_count')
    print("Channel preference counts by customer: \n", channel_preference_counts, '\n')

    sns.barplot(data=channel_preference_counts, x='channel_preference', y='channel_count')
    plt.title("Box plot of customer channel preferences")
    plt.ylabel("Channel count")
    plt.show()

    # 4. Identify channel-specific trends
    numerical_columns = ['satisfaction_score', 'engagement_score', 'income', 'loyalty_years', 'age']

    avg_scores = merged_df.groupby('channel_preference')[numerical_columns].mean()
    print("Average scores by Channel:\n", avg_scores)

