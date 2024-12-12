# Data Cleaning and Analysis

## Overview
This project involves cleaning and analyzing a messy dataset. The tasks include handling missing data, removing duplicates, correcting data types, handling outliers, correcting date formats, and performing advanced data manipulation. Finally, the cleaned dataset is saved as `cleaned_dataset.csv`.

## Prerequisites
- Python 3.x
- Pandas library
- Matplotlib library

Install dependencies using:
```bash
pip install pandas matplotlib
```

## Dataset
The dataset is provided as `messy_dataset.csv` and contains the following columns:
- **Name**: Employee name.
- **Age**: Age of the employee.
- **Salary**: Employee salary.
- **Department**: Department of the employee.
- **Join_Date**: Date the employee joined the company.

## Features and Functions
### 1. Handling Missing Data
- Counts missing values in each column.
- Replaces missing values in the `Age` column with the median age.
- Fills missing values in the `Salary` column with the mean salary of the respective department.
- Fills missing values in the `Department` column with the most frequent department for the respective age group.

### 2. Identifying and Removing Duplicates
- Counts and displays duplicate rows.
- Removes duplicate rows while retaining the first occurrence.

### 3. Correcting Data Types and Handling Outliers
- Converts `Age` and `Salary` columns to numeric values, replacing invalid entries with `NaN`.
- Replaces salary outliers (above the 95th percentile) with the median salary of the respective department.

### 4. Correcting Date Formats
- Converts `Join_Date` to a valid datetime format.
- Replaces invalid dates with the median join date.
- Ensures all dates are formatted as `YYYY-MM-DD`.

### 5. Analyzing the Cleaned Data
- Calculates the average salary by department.
- Finds the most common join date.
- Creates a histogram to display the age distribution of employees.
- Identifies the top 3 departments with the highest median salary.

### 6. Advanced Data Manipulation
- Calculates employee tenure in years based on the current date.
- Categorizes employees as "Senior" (tenure > 5 years) or "Junior".

## Execution
Run the script using:
```bash
python DataCleaning.py
```

### Output
1. Cleaned dataset saved as `cleaned_dataset.csv`.
2. Histogram displaying the age distribution of employees.

## File Structure
- **DataCleaning.py**: Main script with all cleaning and analysis functions.
- **messy_dataset.csv**: Input dataset.
- **cleaned_dataset.csv**: Output cleaned dataset.
- **GenerateData.py**: Script to generate the dataset.


