# Task 8
def marketing_campaign_effectiveness(customers_df, transactions_df, merged_df):
    """ Objective: Measure campaign ROI and effectiveness """

    # 1. Calculate Campaign Conversion Rates

    total_customers = customers_df['customer_id'].nunique()
    customers_with_purchases = transactions_df['customer_id'].nunique()

    conversion_rate = (customers_with_purchases / total_customers) * 100
    print(f"Campaign Conversion Rate: {conversion_rate:.2f}%", '\n')

    # 2. Analyze Response by Customer Segment
    campaign_segment_response = merged_df.groupby('membership_tier')['amount'].sum()
    print("Campaign Response by Customer Segment: \n", campaign_segment_response, '\n')

    # 3. Measure Campaign ROI
    campaign_cost = 1000000  # Example fixed cost for the campaign
    total_campaign_sales = merged_df['amount'].sum()
    campaign_roi = (total_campaign_sales - campaign_cost) / campaign_cost * 100

    print(f"Campaign ROI: {campaign_roi:.2f}%", '\n')

    # 4. Identify Most Effective Channels
    channel_performance = merged_df.groupby('channel')['amount'].sum().reset_index()
    channel_performance.sort_values(by='amount', ascending=False, inplace=True)

    print("Campaign Performance by Channel: \n", channel_performance, '\n')

    # Expected Outcome: Campaign optimization recommendations based on effectiveness analysis
    print("\nCampaign Optimization Recommendations:")
    print("- Focus more resources on the most effective channels (e.g., mobile, store, etc.).")
    print(
        "- Offer more targeted promotions for platinum and Gold membership tiers.")
    print("- Consider increasing the budget for campaigns with a positive ROI to maximize returns.", '\n')
