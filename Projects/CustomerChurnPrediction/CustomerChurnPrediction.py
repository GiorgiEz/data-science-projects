import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



# Task1: Initial Data Analysis
def initial_data_analysis():
    """ Objective: Analyze basic patterns in customer churn data. """

    # 1. Display the first 5 rows
    print("First 5 rows of the dataset:")
    print(df.head(5), "\n")

    # 2. Calculate basic statistics for numerical features
    print("Basic statistical summary of numerical features:")
    print(df.describe(), "\n")

    # 3. Analyze churn rate across different contract types
    churn_rate = df.groupby('contract_length')['churned'].mean()
    print("Churn rate by contract length:")
    print(churn_rate, "\n")

    # 4. Key observations about churn patterns
    print("Key observations:")
    print("1. The churn rate is highest for 2-Year contracts (60.96%), which is counterintuitive "
          "since longer contracts typically imply customer loyalty.")
    print("2. There are negative values in both tenure_months (min: -34.0) and monthly_charges "
          "(min: -119.86), which are unrealistic and likely data entry errors. ")
    print("3. The average satisfaction_score is relatively low (mean: 3.07 out of 5), "
          "with 25% of customers scoring just 1.")


# Task 2: Feature Relationship Analysis
def feature_relationship_analysis():
    """ Objective: Visualize how different features relate to churn. """

    #  1. Create boxplots comparing churned vs non-churned customers
    plt.figure(figsize=(12, 8))
    numerical_features = ['tenure_months', 'monthly_charges', 'total_usage_gb', 'satisfaction_score']
    for i, feature in enumerate(numerical_features, start=1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='churned', y=feature, data=df)
        plt.title(f'Boxplot of {feature} by Churn')
        plt.xlabel('Churned')
        plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

    #  2. Plot average churn rates by contract length
    contract_churn_rate = df.groupby('contract_length')['churned'].mean()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=contract_churn_rate.index, y=contract_churn_rate.values, palette="coolwarm")
    plt.title('Average Churn Rate by Contract Length')
    plt.xlabel('Contract Length')
    plt.ylabel('Average Churn Rate')
    plt.show()

    #  3. Generate correlation heatmap for numerical features
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numerical_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

    #  4. Write interpretations for each visualization
    print("Boxplot: \n")
    print("Tenure_months: Customers with shorter tenure are less likely to churn, which is unusual.")
    print("Monthly_charges: Customers with higher monthly charges are less likely to churn. ")
    print("Total_Usage_GB: Churned customers use slightly more data. ")
    print("Satisfaction score: Both churned and not churned customers satisfaction score is same")

    print("Average Churn Rate by Contract Length: \n")
    print("Customers with longer contract length are more likely to churn")

    print("Correlation heatmap: \n")
    print("Monthly charges and tenure months are strongly not related with each other, "
          "while monthly_charges is slightly related with satisfaction_score and tenure_months "
          "and total_usage_gb are also slightly related")


# Task 3: Feature Engineering
def feature_engineering():
    """ Objective: Create new features for churn analysis. """

    #  1. Calculate average monthly usage (total_usage_gb/tenure_months)
    # Handling zero or negative tenure to avoid division errors
    df['avg_monthly_usage'] = np.where(df['tenure_months'] > 0, df['total_usage_gb'] / df['tenure_months'], 0)
    print("Average Monthly Usage:")
    print(df['avg_monthly_usage'].head())

    #  2. Create customer_value_score combining monthly_charges and tenure
    df['customer_value_score'] = df['monthly_charges'] * np.log(1 + np.maximum(df['tenure_months'], 0))
    print("\nCustomer Value Score:")
    print(df['customer_value_score'].head())

    #  3. Develop risk_score based on payment_delay and service_calls
    w_payment_delay = 1.5
    w_service_calls = 1.0
    df['risk_score'] = (
            w_payment_delay * np.maximum(df['payment_delay_months'], 0) +
            w_service_calls * np.maximum(df['customer_service_calls'], 0)
    )
    print("\nRisk Score:")
    print(df['risk_score'].head())
    print()


# Task 4: Basic Model Development
def basic_model_development():
    """ Objective: Create initial logistic regression model. """

    #  1. Prepare features (scaling, encoding)
    numerical_features = ['tenure_months', 'monthly_charges', 'avg_monthly_usage',
                          'customer_value_score', 'risk_score']
    categorical_features = ['contract_length']

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),  # Scale numerical features
            ('cat', OneHotEncoder(), categorical_features)  # One-hot encode categorical features
        ]
    )

    #  2. Split data 80/20 train/test
    X = df[numerical_features + categorical_features]
    y = df['churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #  3. Train logistic regression model
    # Train logistic model
    logistic_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    logistic_model.fit(X_train, y_train)

    # Prediction
    y_pred = logistic_model.predict(X_test)

    #  4. Calculate accuracy, precision, recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Model Performance:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# Task 5: Model Comparison
def model_comparison():
    """ Objective: Compare different classification models. """

    #  1. Train Random Forest and XGBoost models

    # Preprocess categorical features (OneHotEncoder) and scale numerical data
    numerical_features = ['tenure_months', 'monthly_charges', 'avg_monthly_usage',
                          'customer_value_score', 'risk_score']
    categorical_features = ['contract_length']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    X = df[numerical_features + categorical_features]
    y = df['churned']

    # Create models
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    xgb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    #  2. Perform 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc')
    xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')

    print(f"Random Forest AUC (5-fold): {np.mean(rf_scores):.3f} ± {np.std(rf_scores):.3f}")
    print(f"XGBoost AUC (5-fold): {np.mean(xgb_scores):.3f} ± {np.std(xgb_scores):.3f}")

    #  3. Compare model performances using ROC curves
    rf_model.fit(X, y)
    xgb_model.fit(X, y)

    # Generate predictions
    rf_probs = rf_model.predict_proba(X)[:, 1]
    xgb_probs = xgb_model.predict_proba(X)[:, 1]

    # Calculate ROC curves
    rf_fpr, rf_tpr, _ = roc_curve(y, rf_probs)
    xgb_fpr, xgb_tpr, _ = roc_curve(y, xgb_probs)

    rf_auc = auc(rf_fpr, rf_tpr)
    xgb_auc = auc(xgb_fpr, xgb_tpr)

    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})')
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.show()

    #  4. Analyze feature importance
    rf_feature_importance = rf_model.named_steps['classifier'].feature_importances_
    xgb_feature_importance = xgb_model.named_steps['classifier'].feature_importances_

    print("\nFeature Importance (Random Forest):")
    print(sorted(zip(rf_feature_importance,
                     numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out())),
                 reverse=True))

    print("\nFeature Importance (XGBoost):")
    print(sorted(zip(xgb_feature_importance,
                     numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out())),
                 reverse=True))



if __name__ == '__main__':
    df = pd.read_csv('customer_churn_data.csv')

    """ Task1: Initial Data Analysis """
    initial_data_analysis()

    """ Task 2: Feature Relationship Analysis """
    feature_relationship_analysis()

    """ Task 3: Feature Engineering """
    feature_engineering()

    """ Task 4: Basic Model Development """
    basic_model_development()

    """ Task 5: Model Comparison """
    model_comparison()
