import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress


# Task 1: Analyze Conversion Rate Drivers
def analyze_conversion_rate_drivers():
    """
    Description:
    Investigate factors influencing conversion rates in different regions.
    """

    # Step 1: Create box plots of conversion rate by region using seaborn
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='region', y='conversion_rate')
    plt.show()

    # Step 2: Analyze mean and variability of conversion rates across regions
    region_stats = df.groupby('region')['conversion_rate'].agg(['mean', 'std'])
    print(region_stats)

    # Step 3: Identify potential regional outliers
    high_variability_regions = region_stats[region_stats['std'] > region_stats['std'].mean()]
    print("\nPotential Regional Outliers (High Variability):\n", high_variability_regions)

    # Step 4: Consider additional factors such as ad spend
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='ad_spend', y='conversion_rate', hue='region')
    plt.show()


# Task 2: Click-Through Rate Analysis
def analyze_ctr_drivers():
    """
    Description:
    Explore relationships between ad spend and click-through rate (CTR).
    """
    x = df['ad_spend']
    y = df['click_through_rate']

    # Step 1: Use scatter plot with regression line to analyze correlation
    # Step 4: Consider using color to show conversion rate on scatter plot
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x=x, y=y, scatter_kws={'s': 30}, line_kws={'color': 'red'})
    plt.show()

    # Step 2: Calculate Pearson correlation between ad spend and CTR
    corr_matrix = df[['ad_spend', 'click_through_rate']].corr()
    pearson_corr = corr_matrix.loc['ad_spend', 'click_through_rate']
    print(f'Pearson correlation between ad spend and CTR: {pearson_corr}')

    # Step 3: Identify thresholds where CTR increases or plateaus
    average_ctr = df['click_through_rate'].mean()

    # Bin ad spend data into specified intervals
    bins = pd.interval_range(start=0, end=df['ad_spend'].max() + 1000, freq=1000)
    df['ad_spend_bins'] = pd.cut(df['ad_spend'], bins)

    # Calculate the average CTR for each bin
    bin_avg_ctr = df.groupby('ad_spend_bins')['click_through_rate'].mean()

    # Identify bins with CTR above the overall average
    high_ctr_bins = bin_avg_ctr[bin_avg_ctr > average_ctr]

    # Display the bins with higher-than-average CTR
    print(f"Overall Average CTR: {average_ctr:.4f}")
    print("Bins with higher-than-average CTR: \n", high_ctr_bins)


# Task 3: Customer Satisfaction Analysis
def analyze_customer_satisfaction():
    """
    Description:
    Analyze customer satisfaction and its correlation with bounce rate.
    """
    # Step 1: Use heatmap to display correlation matrix
    corr_matrix = df[['bounce_rate', 'customer_satisfaction']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True)
    plt.show()

    # Step 2: Plot scatter plot with bounce rate on x-axis and satisfaction on y-axis
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['bounce_rate'], y=df['customer_satisfaction'])
    plt.show()

    # Step 3: Look for thresholds where bounce rate impacts satisfaction
    # Bin the data by bounce rate
    df['bounce_rate_bins'] = pd.cut(df['bounce_rate'], bins=5)

    # Calculate average satisfaction in each bin
    thresholds = df.groupby('bounce_rate_bins')['customer_satisfaction'].mean()
    print("Average Customer Satisfaction for Bounce Rate Bins: \n", thresholds)

    # Identify significant drops
    diff = thresholds.diff().dropna()  # Differences between consecutive bins
    significant_drops = diff[diff < -5]  # Threshold for notable decreases
    print("\nThresholds where Bounce Rate significantly impacts Customer Satisfaction: \n", significant_drops)

    # Step 4: Highlight any inverse relationships
    # Highlighting thresholds with steep drops
    plt.figure(figsize=(8, 6))
    plt.scatter(df['bounce_rate'], df['customer_satisfaction'], alpha=0.6)
    plt.title("Bounce Rate vs Customer Satisfaction with Highlighted Drops")
    plt.xlabel("Bounce Rate")
    plt.ylabel("Customer Satisfaction")

    # Highlight bins with significant drops
    for bin_range, drop in significant_drops.items():
        plt.axvspan(bin_range.left, bin_range.right, color='red', alpha=0.2, label=f"Drop: {drop:.2f}")

    plt.legend()
    plt.show()


# Task 4: Time-on-Site Impact
def analyze_time_on_site_impact():
    """
    Description:
    Investigate how time spent on site influences conversion rate.
    """

    # Step 1: Bin time-on-site values into short, medium, and long visits
    bins = [0, 150, 300, df['time_on_site'].max()]
    labels = ['Short', 'Medium', 'Long']
    df['time_category'] = pd.cut(df['time_on_site'], bins=bins, labels=labels)

    # Step 2: Create violin plot of conversion rates for different time-on-site categories
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=df['time_category'], y=df['conversion_rate'])
    plt.title("Conversion Rates by Time-on-Site Categories")
    plt.xlabel("Time on Site Category")
    plt.ylabel("Conversion Rate")
    plt.show()

    # Step 3: Analyze mean conversion rates within each category
    mean_conversion = df.groupby('time_category')['conversion_rate'].mean()
    print("Mean Conversion Rates by Time on Site Category:\n", mean_conversion)

    # Step 4: Perform ANOVA to check significance
    # Group conversion rates by time categories
    short_rates = df[df['time_category'] == 'Short']['conversion_rate']
    medium_rates = df[df['time_category'] == 'Medium']['conversion_rate']
    long_rates = df[df['time_category'] == 'Long']['conversion_rate']

    # Perform ANOVA
    f_stat, p_value = f_oneway(short_rates, medium_rates, long_rates)
    print(f"ANOVA Results:\nF-statistic = {f_stat:.2f}, p-value = {p_value:.5f}")

    # Interpretation
    if p_value < 0.05:
        print("There is a statistically significant difference in conversion rates among the time-on-site categories.")
    else:
        print("No statistically significant difference in conversion rates among the time-on-site categories.")


# Task 5: Conversion Patterns in High Ad Spend
def analyze_high_ad_spend_conversion_patterns():
    """
    Description:
    Analyze conversion rate for high ad spend campaigns.
    """

    # Filter data for high ad spend campaigns
    high_spend_df = df[df['ad_spend'] > 7000]

    # Step 1: Create histogram of conversion rates for ad spend > $7000
    plt.figure(figsize=(8, 6))
    sns.histplot(data=high_spend_df, x='conversion_rate', bins='auto')
    plt.show()

    # Step 2: Calculate conversion rate statistics for high spend
    high_spend_stats = high_spend_df['conversion_rate'].describe()
    print("High Ad Spend Conversion Rate Statistics:\n", high_spend_stats)

    # Step 3: Compare with low ad spend conversions
    low_spend_df = df[df['ad_spend'] <= 7000]
    low_spend_stats = low_spend_df['conversion_rate'].describe()
    print("Low Ad Spend Conversion Rate Statistics:\n", low_spend_stats)

    # Step 4: Use violin plot for deeper visualization
    df['ad_spend_category'] = df['ad_spend'] > 7000
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='ad_spend_category', y='conversion_rate')
    plt.xlabel("High Ad Spend (True = >$7000)")
    plt.ylabel("Conversion Rate")
    plt.title("Conversion Rate Distribution by Ad Spend Category")
    plt.show()


# Task 6: Bounce Rate Clustering
def analyze_bounce_rate_clustering():
    """
    Description:
    Use clustering to categorize bounce rate patterns.
    """

    # Step 1: Standardize variables related to site interaction
    site_interaction_cols = ['bounce_rate', 'time_on_site', 'conversion_rate']  # Relevant columns
    standardized_data = StandardScaler().fit_transform(df[site_interaction_cols])

    # Step 2: Apply K-means clustering to group bounce rates
    kmeans = KMeans(n_clusters=3, random_state=42)  # We can adjust number of clusters if needed
    df['cluster'] = kmeans.fit_predict(standardized_data)

    # Step 3: Visualize clusters in scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df['time_on_site'],
        y=df['bounce_rate'],
        hue=df['cluster'],
        palette='viridis',
        s=50
    )
    plt.title('Bounce Rate Clusters')
    plt.xlabel('Time on Site')
    plt.ylabel('Bounce Rate')
    plt.legend(title='Cluster')
    plt.show()

    # Step 4: Analyze cluster centers and interpret behavior
    cluster_centers = kmeans.cluster_centers_
    cluster_df = pd.DataFrame(cluster_centers, columns=site_interaction_cols)
    print("Cluster Centers:")
    print(cluster_df)

    print("\nCluster Descriptions:")
    for i, row in cluster_df.iterrows():
        print(f"Cluster {i}: {row.to_dict()}")


# Task 7: Conversion Rate by Time on Site
def analyze_conversion_by_time_on_site():
    """
    Description:
    Analyze conversion efficiency by time spent on site.
    """

    # Step 1: Create scatter plot with time on site vs. conversion rate
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='time_on_site', y='conversion_rate', color='blue', alpha=0.6)
    plt.title('Conversion Rate vs. Time on Site')
    plt.xlabel('Time on Site')
    plt.ylabel('Conversion Rate')
    plt.show()

    # Step 2: Fit regression line to show relationship
    slope, intercept, r_value, p_value, std_err = linregress(df['time_on_site'], df['conversion_rate'])
    print(f"Regression Line: Conversion Rate = {slope:.4f} * Time on Site + {intercept:.4f}")
    print(f"R-squared: {r_value ** 2:.4f}")

    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='time_on_site', y='conversion_rate', scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
    plt.title('Regression Line: Conversion Rate vs. Time on Site')
    plt.xlabel('Time on Site')
    plt.ylabel('Conversion Rate')
    plt.show()

    # Step 3: Use heatmap to show high and low conversion regions
    # Bins for heatmap
    heatmap_data = df[['time_on_site', 'conversion_rate']].copy()
    heatmap_data['time_bins'] = pd.cut(df['time_on_site'], bins=10)
    heatmap_data['conversion_bins'] = pd.cut(df['conversion_rate'], bins=10)

    # Pivot table for heatmap
    heatmap_pivot = heatmap_data.pivot_table(index='conversion_bins', columns='time_bins', aggfunc='size', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, cmap='YlGnBu', annot=False, cbar=True)
    plt.title('Conversion Rate by Time on Site (Heatmap)')
    plt.xlabel('Time on Site Bins')
    plt.ylabel('Conversion Rate Bins')
    plt.show()

    # Step 4: Discuss how each metric influences overall performance
    print("Key Insights:")
    print("- The regression line provides an estimate of how conversion rate changes with time on site.")
    print(f"- R-squared value: {r_value ** 2:.4f} suggests the strength of the relationship.")
    print("- Heatmap reveals regions with high and low conversion rates based on time spent on site.")


# Task 8: Bounce Rate vs. Satisfaction
def analyze_bounce_rate_vs_satisfaction():
    """
    Description:
    Explore the impact of high bounce rates on customer satisfaction.
    """
    # Step 1: Use violin plot to show satisfaction for high vs. low bounce rates
    df['satisfaction_category'] = pd.cut(
        df['customer_satisfaction'],
        bins=[50, 70, 85, 100],
        labels=['Low', 'Moderate', 'High']
    )

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='satisfaction_category', y='customer_satisfaction', palette='Set2')
    plt.title('Customer Satisfaction for High vs Moderate vs Low Bounce Rates')
    plt.xlabel('Bounce Rate Category')
    plt.ylabel('Customer Satisfaction')
    plt.show()

    # Step 2: Calculate mean satisfaction for each bounce rate category
    mean_satisfaction = df.groupby('satisfaction_category')['customer_satisfaction'].mean()
    print("Mean Customer Satisfaction by Bounce Rate Category:")
    print(mean_satisfaction)

    # Step 3: Identify trends and discuss potential improvements
    low_bounce_satisfaction = mean_satisfaction['Low']
    moderate_bounce_satisfaction = mean_satisfaction['Moderate']
    high_bounce_satisfaction = mean_satisfaction['High']

    print("\nInsights:")
    print(f"- Average satisfaction for Low Bounce Rate: {low_bounce_satisfaction:.2f}")
    print(f"- Average satisfaction for Moderate Bounce Rate: {moderate_bounce_satisfaction:.2f}")
    print(f"- Average satisfaction for High Bounce Rate: {high_bounce_satisfaction:.2f}")

    # Step 4: Provide summary statistics
    print("\nSummary Statistics:")
    summary_stats = df.groupby('satisfaction_category')['customer_satisfaction'].describe()
    print(summary_stats)


# Task 9: Click-Through Efficiency
def analyze_ctr_efficiency():
    """
    Description:
    Analyze ad spend efficiency based on CTR.
    """
    # Step 1: Calculate CTR/ad_spend ratio
    df['ctr_efficiency'] = df['click_through_rate'] / df['ad_spend']

    # Step 2: Create distribution plot of efficiency
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='ctr_efficiency', kde=True, bins=30, color='blue')
    plt.xlabel('CTR Efficiency (CTR/Ad Spend)')
    plt.title('Distribution of CTR Efficiency')
    plt.show()

    # Step 3: Identify high-efficiency ad spends
    high_efficiency_threshold = df['ctr_efficiency'].quantile(0.90)  # Top 10% threshold
    high_efficiency_ads = df[df['ctr_efficiency'] > high_efficiency_threshold]
    print(f"High-efficiency threshold: {high_efficiency_threshold}")
    print(f"Number of high-efficiency ads: {len(high_efficiency_ads)}")

    # Step 4: Generate summary statistics
    efficiency_stats = df['ctr_efficiency'].describe()
    print("\nSummary Statistics for CTR Efficiency:")
    print(efficiency_stats)

    # Step 5: Profile high-efficiency outliers
    print("\nHigh-Efficiency Ad Spend Profiles:")
    print(high_efficiency_ads[['ad_spend', 'click_through_rate', 'ctr_efficiency']])


# Task 10: Comprehensive Performance Dashboard
def build_performance_dashboard():
    """
    Description:
    Compile top visualizations to build a data story dashboard.
    """

    # Step 1: Identify most impactful visualizations for each metric
    # Step 2: Create combined plots to summarize findings
    # Step 3: Build a narrative with visuals to guide through insights
    # Step 4: Discuss how each metric influences overall performance

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comprehensive Performance Dashboard", fontsize=16)

    # Ad Spend Efficiency
    df['ctr_efficiency'] = df['click_through_rate'] / df['ad_spend']
    sns.histplot(data=df, x='ctr_efficiency', kde=True, ax=axes[0, 0], bins=30, color='blue')
    axes[0, 0].set_title('CTR Efficiency Distribution')
    axes[0, 0].set_xlabel('CTR Efficiency (CTR/Ad Spend)')
    axes[0, 0].set_ylabel('Frequency')

    # Time on Site vs Conversion Rate
    sns.scatterplot(data=df, x='time_on_site', y='conversion_rate', ax=axes[0, 1])
    sns.regplot(data=df, x='time_on_site', y='conversion_rate', scatter=False, ax=axes[0, 1], color='red')
    axes[0, 1].set_title('Conversion Rate vs Time on Site')
    axes[0, 1].set_xlabel('Time on Site')
    axes[0, 1].set_ylabel('Conversion Rate')

    # Bounce Rate & Customer Satisfaction
    df['satisfaction_category'] = pd.cut(
        df['customer_satisfaction'],
        bins=[50, 70, 85, 100],
        labels=['Low', 'Moderate', 'High']
    )

    sns.violinplot(data=df, x='satisfaction_category', y='customer_satisfaction', ax=axes[1, 0], palette='muted')
    axes[1, 0].set_title('Customer Satisfaction for Bounce Rate Categories')
    axes[1, 0].set_xlabel('Bounce Rate Category')
    axes[1, 0].set_ylabel('Customer Satisfaction')

    # Conversion Rate by Ad Spend
    df['ad_spend_category'] = df['ad_spend'] > 7000
    sns.violinplot(data=df, x='ad_spend_category', y='conversion_rate', ax=axes[1, 1], palette='pastel')
    axes[1, 1].set_title('Conversion Rate by Ad Spend')
    axes[1, 1].set_xlabel('Ad Spend Category')
    axes[1, 1].set_ylabel('Conversion Rate')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Step 4: Add a summary of insights
    print("Insights:")
    print("""
        1. CTR Efficiency: High efficiency occurs at certain ad spend levels, indicating diminishing returns beyond a threshold.
        2. Time on Site: Longer time on site correlates with higher conversion rates. Focus on engaging users.
        3. Bounce Rate: Although this dataset shows an unexpected positive correlation between bounce rate and customer satisfaction, this pattern likely stems from the random nature of the data..
        4. Conversion Rate: High ad spend shows slightly better conversion rates but requires strategic budgeting to optimize efficiency.
        """)


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv("sample_data.csv")

    """ Task 1: Analyze Conversion Rate Drivers """
    # analyze_conversion_rate_drivers()

    """ Task 2: Click-Through Rate Analysis """
    # analyze_ctr_drivers()

    """ Task 3: Customer Satisfaction Analysis """
    # analyze_customer_satisfaction()

    """ Task 4: Time-on-Site Impact """
    # analyze_time_on_site_impact()

    """ Task 5: Conversion Patterns in High Ad Spend """
    # analyze_high_ad_spend_conversion_patterns()

    """ Task 6: Bounce Rate Clustering """
    # analyze_bounce_rate_clustering()

    """ Task 7: Conversion Rate by Time on Site """
    # analyze_conversion_by_time_on_site()

    """ Task 8: Bounce Rate vs. Satisfaction """
    # analyze_bounce_rate_vs_satisfaction()

    """ Task 9: Click-Through Efficiency """
    # analyze_ctr_efficiency()

    """ Task 10: Comprehensive Performance Dashboard """
    build_performance_dashboard()
