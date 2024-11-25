import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from ModelTrainingHelper import ModelTrainingHelper, get_features, compare_feature_importance



""" 1. Initial House Price Data Analysis """

def initial_house_price_data():
    """ Objective: Analyze basic patterns and relationships in housing data. """

    # Load the house_prices.csv and display the first 5 rows.
    print("First 5 rows of the dataset: ")
    print(df.head(5).to_markdown(index=False, numalign="left", stralign="left"))

    # Calculate basic statistics for size_sqft, bedrooms, bathrooms, price.
    print("\nBasic statistics for size_sqft, bedrooms, bathrooms, price:")
    print(df[['size_sqft', 'bedrooms', 'bathrooms', 'price']].
          describe().to_markdown(numalign="left", stralign="left"))

    # Compare prices across different neighborhoods.
    print("\nAverage price by neighborhood:")
    print(df.groupby('neighborhood')['price'].mean().
          sort_values(ascending=False).to_markdown(numalign="left", stralign="left"))
    # Visualize price distribution by neighborhood
    plt.figure(figsize=(10, 6))
    df.boxplot(column='price', by='neighborhood', rot=45)
    plt.title('Price Distribution by Neighborhood')
    plt.xlabel('Neighborhood')
    plt.ylabel('Price')
    plt.show()

    # List 3 key observations about housing patterns.
    print("\n3 Key Observations:")
    print("1. Neighborhood significantly influences price, with 'Beachfront' having the highest average.")
    print("2. There's a wide range of apartment sizes, from studios to multi-bedroom units.")
    print("3. Prices generally increase with size, bedrooms, and bathrooms, but there are exceptions.")


""" 2. Price Factor Visualization """

def price_factor_visualization():
    """ Objective: Visualize how different features impact house prices. """

    # Create scatter plots of size_sqft vs price, colored by neighborhood.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='size_sqft', y='price', hue='neighborhood')
    plt.title('House Price vs. Size, Colored by Neighborhood')
    plt.xlabel('Size (sqft)')
    plt.ylabel('Price')
    plt.show()

    # Plot average prices for different bedroom counts.
    avg_price_by_bedroom_count = df.groupby('bedrooms')['price'].mean()
    plt.figure(figsize=(8, 5))
    avg_price_by_bedroom_count.plot(kind='bar')
    plt.title('Average House Price by Bedroom Count')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Average Price')
    plt.xticks(rotation=0)
    plt.show()

    # Generate correlation heatmap for numerical features.
    numerical_features = ['size_sqft', 'bedrooms', 'bathrooms', 'age_years',
                          'lot_size_sqft', 'parking_spots', 'distance_downtown',
                          'distance_highway', 'school_rating', 'crime_rate', 'price']
    corr_matrix = df[numerical_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

    # Write interpretations of each visualization.
    print("\nInterpretations:")
    print("1. Scatter Plot: Shows a positive relationship between size and price. "
          "Neighborhood also plays a role, with some neighborhoods having higher prices overall.")
    print("2. Bar Plot: Indicates that the average price generally increases with the number of bedrooms.")
    print("3. Heatmap: Highlights strong positive correlations between price, size, bedrooms, and bathrooms. "
          "Distance to downtown shows a negative correlation with price.")


""" 3. Custom Metrics Creation """

def custom_metric_creation():
    """ Objective: Develop metrics for house value analysis. """

    # Calculate price_per_sqft.
    df['price_per_sqft'] = round(df['price'] / df['size_sqft'], 2)

    # Create location_value_score combining school_rating and crime_rate.
    df['location_value_score'] = df['school_rating'] / (df['crime_rate'] + 1)  # Add 1 to avoid division by zero

    # Identify top 10 value-for-money properties.
    top_value_properties = (df.sort_values
                            (by=['price_per_sqft', 'location_value_score'], ascending=[True, False]).head(10))

    # Print the top 10 value properties
    print("\nTop 10 Value-for-Money Properties:")
    print(top_value_properties.to_markdown(index=False, numalign="left", stralign="left"))


""" 4. Price Prediction Baseline """

def price_prediction_baseline():
    """ Objective: Create a basic model to predict house prices. """

    # Split data into 80/20 train/test
    helper = ModelTrainingHelper(df)
    r2, mae, model = helper.train_linear_regression()

    print(f"R-squared: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")


""" 5. Model Comparison """

def model_comparison():
    """ Objective: Compare different regression models. """

    # Train Linear, Ridge, Lasso models.
    # Prepare data using ModelTrainingHelper class
    helper = ModelTrainingHelper(df)
    X_train, X_test, y_train, y_test = helper.split_data()

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Lasso Regression': Lasso(random_state=42)
    }

    # Perform 5-fold cross-validation and evaluate
    results = []
    for name, model in models.items():
        cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_scores_mse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_scores_mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

        results.append([
            name,
            np.mean(cv_scores_r2),
            np.std(cv_scores_r2),
            -np.mean(cv_scores_mse),
            np.std(cv_scores_mse),
            -np.mean(cv_scores_mae),
            np.std(cv_scores_mae)
        ])

    # Display results
    results_df = pd.DataFrame(results, columns=['Model', 'Mean R-squared', 'Std R-squared',
                                                'Mean MSE', 'Std MSE', 'Mean MAE', 'Std MAE'])
    print(results_df.to_markdown(index=False, numalign="left", stralign="left"))

    # By the results Ridge regression seems to be slightly better than others
    # Train the best model on the full training set and evaluate on the test set
    best_model = Ridge()
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    r2_test = r2_score(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)

    print(f"\nBest Model Ridge Test Set Performance:")
    print(f"R-squared: {r2_test:.4f}")
    print(f"Mean Squared Error: {mse_test:.2f}")
    print(f"Mean Absolute Error: {mae_test:.2f}")


""" 6. Neighborhood Analysis """

def neighborhood_analysis():
    """ Objective: Analyze price variations across neighborhoods. """

    # Calculate neighborhood price statistics
    print("\nNeighborhood Price Statistics:")
    print(df.groupby('neighborhood')['price'].agg(['mean', 'median', 'std', 'count'])
          .to_markdown(numalign="left", stralign="left"))

    # Create separate models per neighborhood
    neighborhoods = df['neighborhood'].unique()
    results = []

    for neighborhood in neighborhoods:
        df_neighborhood = df[df['neighborhood'] == neighborhood].copy()

        # Prepare data using ModelTrainingHelper
        helper = ModelTrainingHelper(df_neighborhood)
        r2, mae, model = helper.train_linear_regression()

        # Get feature importance's (for models that support it)
        # For Linear Regression, we can use coefficients as a proxy for importance
        feature_importances = dict(zip(get_features(), model.coef_))

        results.append({
            'neighborhood': neighborhood,
            'r2': r2,
            'mae': mae,
            'feature_importances': feature_importances
        })

    compare_feature_importance(results, 'neighborhood')


""" 7. Age Impact Study """

def age_impact_study():
    """  Objective: Analyze how house age affects price. """
    # Create age brackets and analyze prices.
    bins = [0, 10, 25, float('inf')]
    labels = ['0-10', '11-25', '25+']
    df['age_bracket'] = pd.cut(df['age_years'], bins=bins, labels=labels)

    # Analyze prices within age brackets
    print("\nPrice Statistics by Age Bracket:")
    print(df.groupby('age_bracket')['price'].agg(['mean', 'median', 'std', 'count'])
          .to_markdown(numalign="left", stralign="left"))
    age_brackets = df['age_bracket'].unique()
    results = []

    for age_bracket in age_brackets:
        df_age = df[df['age_bracket'] == age_bracket].copy()

        # Check if there are enough samples for training
        if len(df_age) > 1:  # At least 2 samples needed for train/test split
            # Prepare data using ModelTrainingHelper
            helper = ModelTrainingHelper(df_age)
            r2, mae, model = helper.train_linear_regression()

            # Get feature importances (using coefficients for Linear Regression)
            feature_importances = dict(zip(get_features(), model.coef_))

            results.append({
                'age_bracket': age_bracket,
                'r2': r2,
                'mae': mae,
                'feature_importances': feature_importances
            })

    compare_feature_importance(results, 'age_bracket')

    # Compare new vs old house features
    # Example: Compare the top 3 features for the youngest and oldest age brackets
    new_house_features = results[0]['feature_importances']  # Assuming the first bracket is the youngest
    old_house_features = results[-1]['feature_importances']  # Assuming the last bracket is the oldest

    print("\nComparison of New vs. Old House Features:")
    print("Top 3 Features for New Houses:", list(new_house_features.keys())[:3])
    print("Top 3 Features for Old Houses:", list(old_house_features.keys())[:3])


""" 8. Amenity Value Analysis """

def amenity_value_analysis():
    """  Objective: Measure impact of various amenities. """

    # Compare prices with/without pool, garden, parking
    prices_by_pool = df.groupby('has_pool')['price'].mean()
    print("Prices by Pool:\n", prices_by_pool.to_markdown(numalign="left", stralign="left"), '\n')

    prices_by_garden = df.groupby('has_garden')['price'].mean()
    print("Prices by Garden:\n", prices_by_garden.to_markdown(numalign="left", stralign="left"), '\n')

    prices_by_parking_spots = df.groupby('parking_spots')['price'].mean()
    print("Prices by Parking Spots:\n", prices_by_parking_spots.to_markdown(numalign="left", stralign="left"), '\n')

    # Calculate value added by each feature
    value_added_pool = prices_by_pool[1] - prices_by_pool[0]  # Price difference with and without pool
    value_added_garden = prices_by_garden[1] - prices_by_garden[0]  # Price difference with and without pool
    value_added_parking = prices_by_parking_spots.diff().fillna(0)  # Difference for each additional parking spot

    print("Value Added by Pool: \n", value_added_pool)
    print("Value Added by Garden: \n", value_added_garden)
    print("Value Added by Parking Spot: \n", value_added_parking.to_markdown(numalign="left", stralign="left"), '\n')

    # Analyze ROI of different amenities (assuming the actual costs of amenities are as defined)
    cost_pool = 50000
    cost_garden = 10000
    cost_parking_per_spot = 5000

    roi_pool = value_added_pool / cost_pool
    roi_garden = value_added_garden / cost_garden
    roi_parking = value_added_parking / cost_parking_per_spot

    print("ROI of Pool:", roi_pool)
    print("ROI of Garden:", roi_garden)
    print("ROI of Parking Spot:\n", roi_parking.to_markdown(numalign="left", stralign="left"))


""" 9. Location Quality Analysis """

def location_quality_analysis():
    """  Objective: Analyze impact of location features. """
    # Study effect of distance_downtown and school_rating
    print("\nCorrelation with Price:")
    print(df[['distance_downtown', 'school_rating', 'price']]
          .corr()['price'].to_markdown(numalign="left", stralign="left"))

    # Build location quality score model
    features = ['distance_downtown', 'school_rating']
    helper = ModelTrainingHelper(df, features)
    r2, mae, model = helper.train_linear_regression()

    print("\nLocation Quality Score Model:")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE: {mae:.2f}")

    # Find optimal location factors
    optimal_distance = df['distance_downtown'].min()
    optimal_school_rating = df['school_rating'].max()

    print("\nOptimal Location Factors:")
    print(f"Distance to Downtown: {optimal_distance}")
    print(f"School Rating: {optimal_school_rating}")


""" 10. Market Segment Analysis """

def market_segment_analysis():
    """ Objective: Analyze different market segments. """
    # Define luxury vs standard segments
    price_threshold = 1000000  # Example threshold
    df['segment'] = df['price'].apply(lambda x: 'Luxury' if x > price_threshold else 'Standard')

    # Create segment-specific models
    segments = df['segment'].unique()
    results = []

    for segment in segments:
        df_segment = df[df['segment'] == segment].copy()

        # Prepare data using ModelTrainingHelper
        helper = ModelTrainingHelper(df_segment)
        r2, mae, model = helper.train_linear_regression()

        # Get feature importances (using coefficients for Linear Regression)
        feature_importances = dict(zip(get_features(), model.coef_))

        results.append({
            'segment': segment,
            'r2': r2,
            'mae': mae,
            'feature_importances': feature_importances
        })

    # Compare feature importance by segment
    compare_feature_importance(results, 'segment')



if __name__ == '__main__':
    df = pd.read_csv('house_prices.csv')

    """ 1. Initial House Price Data Analysis """
    initial_house_price_data()

    """ 2. Price Factor Visualization """
    price_factor_visualization()

    """ 3. Custom Metrics Creation """
    custom_metric_creation()

    """ 4. Price Prediction Baseline """
    price_prediction_baseline()

    """ 5. Model Comparison """
    model_comparison()

    """ 6. Neighborhood Analysis """
    neighborhood_analysis()

    """ 7. Age Impact Study """
    age_impact_study()

    """ 8. Amenity Value Analysis """
    amenity_value_analysis()

    """ 9. Location Quality Analysis"""
    location_quality_analysis()

    """ 10. Market Segment Analysis """
    market_segment_analysis()
