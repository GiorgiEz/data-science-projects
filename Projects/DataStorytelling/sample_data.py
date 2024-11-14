import pandas as pd
import numpy as np

def generate_data_storytelling_data(n_points=200):
    np.random.seed(42)

    data = {
        "event_id": range(1, n_points + 1),
        "ad_spend": np.random.normal(5000, 1500, n_points),
        "click_through_rate": np.random.uniform(0.05, 0.20, n_points),
        "conversion_rate": np.random.uniform(0.01, 0.10, n_points),
        "region": np.random.choice(["North", "South", "East", "West"], n_points),
        "time_on_site": np.random.normal(200, 50, n_points),
        "bounce_rate": np.random.uniform(0.20, 0.70, n_points),
        "customer_satisfaction": np.random.randint(50, 100, n_points),
    }

    df = pd.DataFrame(data)

    df["click_through_rate"] = df["click_through_rate"].clip(0, 1)
    df["conversion_rate"] = df["conversion_rate"].clip(0, 1)
    df["time_on_site"] = df["time_on_site"].clip(0, 400)
    df["bounce_rate"] = df["bounce_rate"].clip(0, 1)

    return df

df = generate_data_storytelling_data()

output_filename = "sample_data.csv"
df.to_csv(output_filename, index=False)
print(f"Data saved to {output_filename}")
