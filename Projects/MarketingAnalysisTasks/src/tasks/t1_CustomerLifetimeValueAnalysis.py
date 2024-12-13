import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data



# Task 1
def customer_lifetime_value_analysis(customers_df, merged_df):
    """ Objective: Calculate and analyze Customer Lifetime Value (CLV) """

    # 1. Calculate average monthly purchase value per customer
    merged_df['month'] = merged_df['date'].dt.to_period('M')  # Extract month-year from transaction date
    monthly_avg_purchase = (
        merged_df.groupby(['customer_id', 'month'])['amount']
        .sum().groupby('customer_id').mean().reset_index(name='avg_monthly_purchase_value')
    )
    print('Average monthly purchase value per customer: \n', monthly_avg_purchase.head(5), '\n')

    # 2. Estimate customer lifespan using loyalty_years
    customer_lifespan = round(customers_df['loyalty_years'].mean(), 1)
    print('Customer Lifespan: ', customer_lifespan, '\n')

    # 3. Implement CLV prediction model
    transaction_summary = summary_data_from_transaction_data(
        merged_df, 'customer_id', 'date', monetary_value_col='amount'
    )
    print('Transaction Summary Data: \n', transaction_summary.head(), '\n')

    bgf = BetaGeoFitter()
    bgf.fit(
        transaction_summary['frequency'],
        transaction_summary['recency'],
        transaction_summary['T']
    )
    print('BG/NBD model fitted.')

    transaction_summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        12,
        transaction_summary['frequency'],
        transaction_summary['recency'],
        transaction_summary['T']
    )
    print('Predicted purchases (12 months): \n', transaction_summary['predicted_purchases'].head(), '\n')

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(
        transaction_summary['frequency'],
        transaction_summary['monetary_value']
    )
    print('Gamma-Gamma model fitted.')

    transaction_summary['CLV'] = ggf.customer_lifetime_value(
        bgf,
        transaction_summary['frequency'],
        transaction_summary['recency'],
        transaction_summary['T'],
        transaction_summary['monetary_value'],
        time=12,  # Time horizon for CLV prediction in months
        discount_rate=0.01
    )
    print('Predicted Customer Lifetime Value: \n', transaction_summary[['CLV']].head(), '\n')

    # Merge CLV back to customer data
    customer_clv = customers_df.merge(transaction_summary[['CLV']], left_on='customer_id', right_index=True)
    print('Customer Data with CLV: \n', customer_clv.head(), '\n')

    # 4. Identify top 10% valuable customers
    top_10_percent_customers = customer_clv.sort_values(by='CLV', ascending=False).head(int(len(customer_clv) * 0.1))
    print('Top 10% Customers by CLV: \n', top_10_percent_customers[['customer_id', 'CLV']], '\n')

    # Plot distribution of CLV
    plt.figure(figsize=(10, 6))
    sns.histplot(customer_clv['CLV'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Customer Lifetime Value (CLV)')
    plt.xlabel('CLV')
    plt.ylabel('Frequency')
    plt.show()