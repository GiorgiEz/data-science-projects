from lifetimes.utils import summary_data_from_transaction_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


# Task 4
def customer_churn_risk_analysis(customers_df, transactions_df, merged_df):
    """ Objective: Predict and prevent customer churn """

    #  1. Define churn indicators
    # Customers with low number of purchases might have high risk for churn
    churn_indicators = transactions_df.groupby('customer_id')['amount'].count().reset_index(name='purchases_count')
    # Customers with lower satisfaction scores might have high risk for churn
    churn_indicators['avg_satisfaction_score'] = transactions_df.groupby('customer_id')['satisfaction_score'].mean()
    # Customers with lower engagement scores might have high risk for churn
    churn_indicators['engagement_score'] = customers_df['engagement_score']

    print(churn_indicators['purchases_count'].sort_values(ascending=True).head(5))
    print(churn_indicators['avg_satisfaction_score'].sort_values(ascending=True).head(5))
    print(churn_indicators['engagement_score'].sort_values(ascending=True).head(5), '\n')

    #  2. Calculate recency, frequency, monetary (RFM) scores
    rfm_summary = summary_data_from_transaction_data(
        transactions_df,
        customer_id_col='customer_id',
        datetime_col='date',
        monetary_value_col='amount'
    )
    print("RFM Summary: \n", rfm_summary.head(), "\n")

    # Merge with scores
    churn_indicators = churn_indicators.merge(rfm_summary, on='customer_id', how='left')

    churn_indicators['has_churn_risk'] = (
            (churn_indicators['recency'] > churn_indicators['recency'].quantile(0.75)) &
            (churn_indicators['purchases_count'] < churn_indicators['purchases_count'].quantile(0.25)) &
            (churn_indicators['avg_satisfaction_score'] < churn_indicators['avg_satisfaction_score'].quantile(0.25)) &
            (churn_indicators['engagement_score'] < churn_indicators['engagement_score'].quantile(0.25))
    )

    churn_indicators['has_churn_risk'] = churn_indicators['has_churn_risk'].astype(int)

    # 1 means customer has churn risk
    print(churn_indicators['has_churn_risk'].value_counts())

    #  3. Build churn risk prediction model
    features = ['frequency', 'recency', 'monetary_value', 'engagement_score', 'purchases_count', 'avg_satisfaction_score']
    X = churn_indicators[features]
    y = churn_indicators['has_churn_risk']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("Feature Importances: \n", feature_importances)

    # Step 5: Evaluate model performance
    y_pred = model.predict(X_test)
    print("Classification Report: \n", classification_report(y_test, y_pred))

    # Step 6: Identify at-risk customers
    churn_indicators['predicted_churn'] = model.predict(X_scaled)
    at_risk_customers = churn_indicators[churn_indicators['predicted_churn'] == 1]
    print("At-risk customers by prediction model: \n", at_risk_customers['customer_id'].count(), "\n")

    # Expected Outcome: Churn prevention strategy report
    # Recommendations: Create targeted retention campaigns for at-risk customers
    print("Churn prevention strategy recommendations:")
    print("- Focus on customers with low engagement and high recency.")
    print("- Offer incentives for customers with declining loyalty.")
    print("- Improve communication channels to re-engage at-risk customers.")
