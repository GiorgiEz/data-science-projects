from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def get_features():
    """Returns a list of feature names used for model training."""
    return ['size_sqft', 'lot_size_sqft', 'parking_spots', 'has_pool',
            'has_garden', 'distance_downtown', 'condition_encoded', 'energy_rating_encoded',
            'distance_highway', 'school_rating', 'crime_rate']


def compare_feature_importance(results, key):
    # Compare feature importance by segment
    print("\nFeature Importance by Market Segment:")
    for result in results:
        print(f"\n{result[key]}:")
        print(f"  R-squared: {result['r2']:.4f}")
        print(f"  MAE: {result['mae']:.2f}")
        # Sort feature importances by absolute value
        sorted_importances = dict(sorted(result['feature_importances'].items(),
                                         key=lambda item: abs(item[1]), reverse=True))
        for feature, importance in sorted_importances.items():
            print(f"    {feature}: {importance:.4f}")


class ModelTrainingHelper:
    """
    A helper class to prepare data for model training.

    Handles encoding of categorical features and splitting data into training and testing sets.
    """

    def __init__(self, df, features=None):
        """
        Initializes the ModelTrainingHelper with a pandas DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing house price data.
        """
        self.df = df
        self.features = get_features() if features is None else features

    def encode_condition_and_energy_rating(self, le):
        """
        Encodes the 'condition' and 'energy_rating' columns using LabelEncoder.

        Args:
            le (LabelEncoder): A LabelEncoder object to fit and transform the features.
        """
        self.df['condition_encoded'] = le.fit_transform(self.df['condition'])
        self.df['energy_rating_encoded'] = le.fit_transform(self.df['energy_rating'])

    def get_X(self):
        """
        Prepares the feature matrix (X) for model training.

        Encodes categorical features and selects the relevant features.

        Returns:
            pd.DataFrame: The feature matrix X.
        """
        le = LabelEncoder()
        self.encode_condition_and_energy_rating(le)  # Encode categorical features

        X = self.df[self.features]
        return X

    def get_y(self):
        """
        Prepares the target variable (y) for model training.

        Returns:
            pd.Series: The target variable y.
        """
        target = 'price'
        y = self.df[target]
        return y

    def split_data(self):
        """
        Splits the data into training and testing sets.

        Returns:
            tuple: A tuple containing X_train, X_test, y_train, y_test.
        """
        X = self.get_X()
        y = self.get_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_linear_regression(self):
        """
        Trains a Linear Regression model on the prepared data.

        Splits the data, trains the model, and returns the R-squared, MAE, and the trained model.

        Returns:
            tuple: A tuple containing (R-squared, MAE, trained LinearRegression model).
        """
        X_train, X_test, y_train, y_test = self.split_data()

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        return r2, mae, model
