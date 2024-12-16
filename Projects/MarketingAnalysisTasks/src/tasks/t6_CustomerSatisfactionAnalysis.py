import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Task 6
def customer_satisfaction_analysis(customers_df, transactions_df, merged_df):
    """ Objective: Analyze satisfaction scores and trends """

    #  1. Calculate satisfaction trends over time
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    satisfaction_trends = transactions_df.groupby(transactions_df['date'].dt.to_period('M'))['satisfaction_score'].mean().reset_index()
    satisfaction_trends['date'] = satisfaction_trends['date'].dt.to_timestamp()

    # Plot satisfaction trends over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=satisfaction_trends, x='date', y='satisfaction_score', marker='o')
    plt.title("Satisfaction Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Satisfaction Score")
    plt.grid(True)
    plt.show()

    print("Satisfaction Trends Over Time:\n", satisfaction_trends)

    #  2. Identify factors affecting satisfaction
    # Merge customer data with transactions to include customer-related features
    factors = ['amount', 'frequency', 'monetary_value']
    transactions_df['frequency'] = transactions_df.groupby('customer_id')['date'].transform('count')
    transactions_df['monetary_value'] = transactions_df.groupby('customer_id')['amount'].transform('sum')

    aggregated_data = transactions_df.groupby('customer_id').agg({
        'amount': 'sum',
        'frequency': 'mean',
        'monetary_value': 'sum',
        'satisfaction_score': 'mean'
    }).reset_index()

    # Train a model to determine factor importance
    X = aggregated_data[factors]
    y = aggregated_data['satisfaction_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    feature_importances = pd.DataFrame({
        'Feature': factors,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("Factors Affecting Satisfaction:\n", feature_importances)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    print("Model Performance: R2 Score =", r2_score(y_test, y_pred))
    print("Model Performance: RMSE =", mean_squared_error(y_test, y_pred))

    #  3. Analyze satisfaction by customer segment
    satisfaction_by_segment = merged_df.groupby('membership_tier')['satisfaction_score'].mean().reset_index()

    # Plot satisfaction by segment
    plt.figure(figsize=(8, 5))
    sns.barplot(data=satisfaction_by_segment, x='membership_tier', y='satisfaction_score', palette='viridis')
    plt.title("Satisfaction by Customer Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Average Satisfaction Score")
    plt.show()

    print("Satisfaction by Customer Segment:\n", satisfaction_by_segment)

    #  4. Create improvement recommendations
    recommendations = """
    Customer Satisfaction Improvement Plan:
    1. Enhance engagement with customers in low-satisfaction segments.
    2. Address concerns related to low-performing product categories or service issues.
    3. Offer personalized incentives to customers with declining satisfaction scores.
    4. Focus on improving the purchasing experience to boost satisfaction.
    """
    print(recommendations)
