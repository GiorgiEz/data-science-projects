import pandas as pd
import matplotlib.pyplot as plt


messy_dataset_path = "messy_dataset.csv"


def handling_missing_data():
    """
    This function counts the number of missing values in each column, replaces the missing values in Age column
    with median age, fills missing salary values with the mean salary of the respective department and fills
    the missing values in the Department column with the most frequent department for that persons age group.
    """

    """ 1. Count the number of missing values in each column """
    missing_values_count = df.isnull().sum()

    """ 2. Replace missing values in the Age column with the median age. """
    # Convert non-numeric values to NaN, then count the median
    age_median = int(pd.to_numeric(df['Age'], errors='coerce').median())
    df["Age"] = df["Age"].fillna(age_median)

    """ 3. Fill missing Salary values with the mean salary of the respective department. """
    mean_by_department = df.groupby(by='Department')["Salary"].mean()

    # Create a mask for rows where 'Department' is not NaN
    mask = df['Department'].notna()
    # Fill missing salary values with the department-specific mean only where 'Department' is not NaN
    df.loc[mask, 'Salary'] = df.loc[mask, 'Salary'].fillna(round(df.loc[mask, 'Department'].map(mean_by_department)))
    # Handle rows where 'Department' is NaN (e.g., fill with the global salary mean)
    df['Salary'].fillna(round(df['Salary'].mean()), inplace=True)

    """4. Fill missing values in the Department column with the most frequent department for that person’s age group"""
    # Find the most frequent department by each age group
    most_frequent_department_by_age = df.groupby('Age')['Department'].agg(lambda x: x.mode()[0])
    # Fill missing values in the Department column based on the most frequent department for the corresponding age
    df['Department'] = df['Department'].fillna(df['Age'].map(most_frequent_department_by_age))

    # print('Number of missing values:', missing_values_count, sep='\n')
    # print(f'Median age: {age_median}', '\n')
    # print(f'Checking if theres any None value left in Age Columns: {df[df["Age"].isnull()]}', '\n')
    # print(f'Checking if theres any None value left in Salary columns: {df[df["Salary"].isnull()]}', '\n')
    # print(f'Checking if theres any None value left in Department columns: {df[df["Department"].isnull()]}', '\n')

def identifying_and_removing_duplicates():
    """
    This function counts the number of duplicate rows displays it and removes all duplicate rows.
    """
    duplicated_count = df.duplicated().sum()
    duplicated_rows = df[df.duplicated(keep=False)].value_counts()

    df.drop_duplicates(inplace=True)  # Drop the duplicates while keeping the first occurrence

    updated_rows = df[df.duplicated(keep=False)].value_counts()

    # print(f'Number of duplicated rows', duplicated_count, '\n')
    # print('Before removing duplicates: ', duplicated_rows, sep='\n')
    # print(f'After removing duplicates: ', updated_rows, sep='\n')

def correcting_data_types_handling_outliers():
    """
    Converts age and salary columns to numeric, replaces invalid ages with NaN and replaces extreme salary
    outliers with the department median.
    """
    # Convert the Age and Salary columns to numeric, non-numeric values will be turned into NaN
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

    # Calculate the 95th percentile of the Salary column and replace salary outliers with department median
    salary_95th_percentile = df['Salary'].quantile(0.95)
    median_salary_by_department = df.groupby(by='Department')["Salary"].median()
    df['Salary'] = df.apply(
        lambda row: median_salary_by_department[row['Department']]
        if row['Salary'] >= salary_95th_percentile else row['Salary'],
        axis=1
    )

    # print('Display rows with non-numeric Age values, if theres any:', df[df['Age'].isna()], sep='\n')
    # print('Display rows with non-numeric Salary values, if theres any:', df[df['Salary'].isna()], sep='\n')
    # print(f'95th percentile of salary: {salary_95th_percentile}', '\n')
    # print('Median Salary by each Department: ', median_salary_by_department, sep='\n')
    # print(f'Check if theres any Salary left over the 95th percentile: {df[df["Salary"] >= salary_95th_percentile]}', '\n')

def correcting_date_formats():
    """
    Any invalid entry is replaced by the median join date, all dates are ensured to be in 'YYYY-MM-DD' format.
    """
    # covert join_date column to a valid datetime format
    df['Join_Date'] = pd.to_datetime(df['Join_Date'], errors='coerce')
    # find median join date
    median_join_date = df["Join_Date"].median()
    # fill NaT values with median date
    df['Join_Date'] = df['Join_Date'].fillna(median_join_date)
    # Ensure all dates are displayed in 'YYYY-MM-DD' format
    df['Join_Date'] = df['Join_Date'].dt.strftime('%Y-%m-%d')

    # print(df['Join_Date'])

def analyzing_cleaned_data():
    """
    Calculates the average salary by department, finds the most common join date, determines the distribution
    of ages and creates and displays the histogram, identifies the top 3 departments with the highest
    median salary.
    """
    # calculate the average salary by department
    avg_salary_by_department = df.groupby(by='Department')["Salary"].mean()
    # find the most common join date
    most_frequent_join_date = df['Join_Date'].mode()[0]

    # determine the distribution of ages and create a histogram
    # Using the plot function in pandas
    hist = df['Age'].plot.hist(bins=range(20, 51, 5), edgecolor='black', alpha=0.7)

    # Add title and labels
    hist.set_title('Age Distribution of Employees')
    hist.set_xlabel('Age')
    hist.set_ylabel('Frequency')
    plt.show()

    # Identify the top 3 departments with the highest median salary
    median_salary_by_department = df.groupby(by='Department')["Salary"].median().sort_values(ascending=False)[:3]

    # print('Average Salary by Department: ', avg_salary_by_department, sep='\n')
    # print(f'Most common join date: {most_frequent_join_date}', '\n')
    # print('Top 3 departments with the highest median salary', median_salary_by_department, sep='\n')

def advanced_data_manipulation():
    # Convert Join_Date to datetime
    df['Join_Date'] = pd.to_datetime(df['Join_Date'])
    # Calculate tenure in years
    current_date = pd.to_datetime("today")  # Get the current date
    df['Tenure_Years'] = (current_date - df['Join_Date']).dt.days // 365  # Calculate tenure in years

    df['Tenure_Category'] = df['Tenure_Years'].apply(lambda x: 'Senior' if x > 5 else 'Junior')

    # print(df[['Name', 'Join_Date', 'Tenure_Years', 'Tenure_Category']])



if __name__ == '__main__':
    df = pd.read_csv(messy_dataset_path)

    """ Part 1: Handling Missing Data """
    handling_missing_data()

    """ Part 2: Identifying and Removing Duplicates """
    identifying_and_removing_duplicates()

    """ Part 3: Correcting Data Types and Handling Outliers """
    correcting_data_types_handling_outliers()

    """ Part 4: Correcting Date Formats """
    correcting_date_formats()

    """ Part 5: Analyzing the Cleaned Data """
    analyzing_cleaned_data()
    """ Bonus Challenge: Advanced Date Manipulation """
    advanced_data_manipulation()

    # Save modified dataframe to a new file
    df.to_csv('cleaned_dataset.csv', index=False, sep=',', encoding='utf-8')
