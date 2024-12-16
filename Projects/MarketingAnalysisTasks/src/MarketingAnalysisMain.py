import pandas as pd
from DataPreparation import DataPreparation
from tasks.t1_CustomerLifetimeValueAnalysis import customer_lifetime_value_analysis
from tasks.t2_SeasonalTrendAnalysis import seasonal_trend_analysis
from tasks.t3_ChannelPerformanceOptimization import channel_performance_optimization
from tasks.t4_CustomerChurnRiskAnalysis import customer_churn_risk_analysis
from tasks.t5_ProductCategoryAnalysis import product_category_analysis
from tasks.t6_CustomerSatisfactionAnalysis import customer_satisfaction_analysis
from tasks.t7_RegionBasedPerformanceAnalysis import region_based_performance_analysis
from tasks.t8_MarketingCampaignEffectiveness import marketing_campaign_effectiveness



if __name__ == '__main__':
    customers_df = pd.read_csv('../data/customer_data.csv')
    transactions_df = pd.read_csv('../data/transaction_data.csv')

    preparation = DataPreparation(customers_df, transactions_df) # Initializes the data preparation class
    preparation.transactions_date_conversion() # Convert date
    merged_df = preparation.merge_dataframes() # Merges customers and transactions dataframes

    """ Task 1: Customer Lifetime Value Analysis """
    customer_lifetime_value_analysis(customers_df, merged_df)

    """ Task 2: Seasonal Trend Analysis """
    seasonal_trend_analysis(customers_df, transactions_df, merged_df)

    """ Task 3: Channel Performance Optimization """
    channel_performance_optimization(customers_df, transactions_df, merged_df)

    """ Task 4: Customer Churn Risk Analysis """
    customer_churn_risk_analysis(customers_df, transactions_df, merged_df)

    """ Task 5: Product Category Analysis """
    product_category_analysis(customers_df, transactions_df, merged_df)

    """ Task 6: Customer Satisfaction Analysis """
    customer_satisfaction_analysis(customers_df, transactions_df, merged_df)

    """ Task 7: Region-Based Performance Analysis """
    region_based_performance_analysis(customers_df, transactions_df, merged_df)

    """ Task 8: Marketing Campaign Effectiveness """
    marketing_campaign_effectiveness(customers_df, transactions_df, merged_df)
