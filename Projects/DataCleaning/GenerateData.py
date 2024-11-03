import pandas as pd
import numpy as np
import random
from faker import Faker

random.seed(42)
np.random.seed(42)
fake = Faker()
data = {
    "Name": [fake.name() for _ in range(100)],
    "Age": [random.choice([25, 30, 35, np.nan, "forty", 50, 1000]) for _ in range(100)],
    "Salary": [random.choice([50000, 60000, 70000, np.nan, 9999999, 100000]) for _ in range(100)],
    "Join_Date": [fake.date_this_decade() if i % 10 != 0 else fake.date_this_century() for i in range(100)],
    "Department": [random.choice(["IT", "HR", "Finance", None, "Admin", ""]) for _ in range(100)]
 }
df = pd.DataFrame(data)
df = pd.concat([df, df.iloc[:10]], ignore_index=True)
invalid_dates = ["32/01/2020", "2020-14-03", "01-01-20", "Invalid Date"]
indices = np.arange(0, len(df), 10)
num_invalid_dates_needed = len(indices)
invalid_dates_extended = (invalid_dates * (num_invalid_dates_needed // len(invalid_dates) + 1))[:num_invalid_dates_needed]
df.loc[indices, "Join_Date"] = invalid_dates_extended
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('messy_dataset.csv', index=False)
