import pandas as pd


class DataPreparation:

    def __init__(self, customers_df, transactions_df):
        self.customers_df = customers_df
        self.transactions_df = transactions_df

    def transactions_date_conversion(self):
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])

    def merge_dataframes(self):
        return self.transactions_df.merge(self.customers_df, on='customer_id')

    def validate_data(self):
        checks = {
            'missing_values': {
                'customers': self.customers_df.isnull().sum(),
                'transactions': self.transactions_df.isnull().sum()
            },
            'value_ranges': {
                'satisfaction': self.transactions_df['satisfaction_score'].between(1, 10).all(),
                'engagement': self.customers_df['engagement_score'].between(1, 100).all(),
                'amount': (self.transactions_df['amount'] > 0).all()
            },
            'customer_linkage': set(self.transactions_df['customer_id']).issubset(
                set(self.customers_df['customer_id']))
        }
        return checks
