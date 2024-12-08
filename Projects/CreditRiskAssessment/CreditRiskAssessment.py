import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.tree import plot_tree


#  Task 2: Data Preparation and Initial Analysis
def data_preparation():
    """ Objective: Prepare and analyze credit risk dataset for decision tree modeling. """

    #  1. Load the credit risk dataset and perform: display first 10 rows,
    #  calculate basic statistics, identify missing values and outliers
    print("First 10 rows of the dataset:")
    print(df.head(10))
    print("\nBasic Statistics:")
    print(df.describe(include='all'))
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Detect outliers using IQR for numerical features
    for column in ['annual_income', 'credit_score', 'loan_amount', 'existing_debt']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
        print(f"\nOutliers in {column}: {len(outliers)} instances")

    #  2. Create visualizations showing: distribution of risk categories,
    # relationship between income and default risk, correlation matrix

    # Distribution of risk categories
    plt.figure(figsize=(8, 6))
    df['default_status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribution of Risk Categories (Default Status)')
    plt.xlabel('Default Status')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['0', '1'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Relationship between income and default risk
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='default_status', y='annual_income', data=df, palette='Set2')
    plt.title('Income vs Default Risk')
    plt.xlabel('Default Status')
    plt.ylabel('Annual Income')
    plt.xticks(ticks=[0, 1], labels=['No Default', 'Default'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr = df[numerical_features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix')
    plt.show()

    #  3. Write a summary of key findings
    print("There is 44 missing values in annual_income column, 58 in employment_duration column and"
          "42 in credit_score column, other columns have no missing values.")
    print("In total 157 outliers where spotted")
    print("Around 20% of the records have default status")
    print("Annual income doesn't tell much about default risk")
    print("debt_to_income and existing_debt have very strong correlation (0.83)")


#  Task 3: Implementation of Gini Index and Entropy
def gini_index_and_entropy():
    """ Objective: Understand and implement different splitting criteria. """

    #  1. Implement two decision trees using Gini index and Entropy as splitting criteria
    X = df[numerical_features]
    y = df['default_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Implementation of decision trees using Gini and Entropy
    clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_split=5)
    clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_split=5)

    # Training of Gini and entropy indices
    clf_gini.fit(X_train, y_train)
    clf_entropy.fit(X_train, y_train)

    # Making predictions
    gini_pred = clf_gini.predict(X_test)
    entropy_pred = clf_entropy.predict(X_test)

    # Evaluation
    print("Gini Index Tree:")
    print(classification_report(y_test, gini_pred))
    print("\nEntropy Tree:")
    print(classification_report(y_test, entropy_pred))

    # Accuracy Comparison
    gini_accuracy = accuracy_score(y_test, gini_pred)
    entropy_accuracy = accuracy_score(y_test, entropy_pred)
    print(f"\nAccuracy - Gini Index: {gini_accuracy:.4f}")
    print(f"Accuracy - Entropy: {entropy_accuracy:.4f}")

    #  2. Calculate and compare: node impurity at each split, information gain for each feature
    # Function to extract node impurity and information gain
    def get_impurity_and_info_gain(tree_model):
        # Get the tree's attributes
        tree = tree_model.tree_

        # Extracting impurities (Gini or Entropy at each node)
        impurities = tree.impurity

        # Calculate information gain for each split (parent impurity - weighted avg of child impurities)
        node_count = len(impurities)
        information_gain = []

        for i in range(node_count):
            # For leaf nodes, no split occurred, so no information gain
            if tree.children_left[i] == tree.children_right[i]:
                information_gain.append(0)
            else:
                # Calculate the weighted average of child impurities
                left_impurity = impurities[tree.children_left[i]]
                right_impurity = impurities[tree.children_right[i]]

                left_size = tree.n_node_samples[tree.children_left[i]]
                right_size = tree.n_node_samples[tree.children_right[i]]

                total_size = left_size + right_size

                # Information gain formula: parent impurity - weighted avg of child impurities
                gain = impurities[i] - (left_size / total_size) * left_impurity - (
                            right_size / total_size) * right_impurity
                information_gain.append(gain)

        return impurities, information_gain

    # For Gini
    gini_impurities, gini_information_gain = get_impurity_and_info_gain(clf_gini)
    print()
    print("Gini Impurities at Each Node: \n", gini_impurities)
    print("Gini Information Gain at Each Node: \n", gini_information_gain)

    # For Entropy
    entropy_impurities, entropy_information_gain = get_impurity_and_info_gain(clf_entropy)
    print()
    print("Entropy Impurities at Each Node: \n", entropy_impurities)
    print("Entropy Information Gain at Each Node: \n", entropy_information_gain)

    #  3. Document which criterion performs better for credit risk assessment and why
    print("The Gini impurity tends to be lower overall compared to entropy at most nodes.")
    print("This suggests that Gini tends to result in purer splits in the tree, indicating "
          "it might produce better classification performance on average.")
    print("Information Gain: Entropy shows higher information gain values at certain nodes, "
          "but Gini is more consistent in its gains, especially in the deeper nodes.")
    print("Gini Index is likely to perform better overall for this credit risk dataset, as "
          "it produces more consistent and purer splits across the decision tree, leading to "
          "more reliable classifications.")


#  Task 4: Model Development and Tuning
def model_development_and_tuning():
    """ Objective: Build and optimize a decision tree model for credit risk prediction. """

    #  1. Split data into training (70%) and testing (30%) sets
    X = df[numerical_features]
    y = df['default_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #  2. Create initial decision tree model with default parameters
    decision_tree = DecisionTreeClassifier(random_state=42)

    # Train initial model
    decision_tree.fit(X_train, y_train)

    # Predict with the initial model
    y_pred_initial = decision_tree.predict(X_test)

    # Initial model evaluation
    print("Initial Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred_initial))
    print("Classification Report:\n", classification_report(y_test, y_pred_initial))

    #  3. Perform hyperparameter tuning: test different max_depth values,
    #  adjust min_samples_split, modify min_samples_leaf
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],  # Try different tree depths
        'min_samples_split': [2, 5, 10],  # Control how many samples are needed to split a node
        'min_samples_leaf': [1, 5, 10],  # Minimum samples per leaf node
    }

    grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best parameters from GridSearchCV
    best_params = grid_search.best_params_
    print("\nBest Hyperparameters from Grid Search:")
    print(best_params)

    # Re-train the model with the best parameters found from GridSearchCV
    best_model = grid_search.best_estimator_

    # Predict with the optimized model
    y_pred_optimized = best_model.predict(X_test)

    # Optimized model evaluation
    print("\nOptimized Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred_optimized))
    print("Classification Report:\n", classification_report(y_test, y_pred_optimized))

    #  4. Document the impact of each parameter on model performance
    print("Accuracy on the best model is 0.76 compared to the initial model which is 0.65")


#  Task 5: Model Evaluation and Performance Analysis
def model_evaluation_performance_analysis():
    """ Objective: Evaluate model performance and handle class imbalance. """

    #  1. Calculate and analyze: confusion matrix, precision/recall/F1-score, ROC curve and AUC score
    X = df[numerical_features]
    y = df['default_status']

    # Impute missing values in the dataset
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)  # Apply imputer to all features

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

    #  2. Decision tree model with the best parameters
    decision_tree = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=10, min_samples_leaf=1)

    # Train model
    decision_tree.fit(X_train, y_train)

    # Predict with the model
    y_pred = decision_tree.predict(X_test)

    print("Confusion Matrix (Before Handling Class Imbalance):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report (Before Handling Class Imbalance):")
    print(classification_report(y_test, y_pred))

    # ROC Curve and AUC Score
    fpr, tpr, thresholds = roc_curve(y_test, decision_tree.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC Score (Before Handling Class Imbalance): {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    #  2. Implement techniques to handle class imbalance: use class weights, apply SMOTE, adjust decision threshold
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Train the model again on the balanced dataset
    decision_tree.fit(X_res, y_res)
    y_pred_res = decision_tree.predict(X_test)

    #  3. Compare performance metrics before and after balancing
    print("\nConfusion Matrix (After Handling Class Imbalance):")
    print(confusion_matrix(y_test, y_pred_res))

    print("\nClassification Report (After Handling Class Imbalance):")
    print(classification_report(y_test, y_pred_res))

    # ROC Curve and AUC Score after SMOTE
    fpr_res, tpr_res, thresholds_res = roc_curve(y_test, decision_tree.predict_proba(X_test)[:, 1])
    roc_auc_res = auc(fpr_res, tpr_res)
    print(f"ROC AUC Score (After Handling Class Imbalance): {roc_auc_res:.4f}")

    # Plot ROC Curve after SMOTE
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_res, tpr_res, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_res:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - After SMOTE')
    plt.legend(loc="lower right")
    plt.show()

    print("\nComparison of Performance Metrics:")
    print(f"Accuracy Before: {decision_tree.score(X_test, y_test):.4f}")
    print(f"Accuracy After: {decision_tree.score(X_test, y_test):.4f}")
    print(f"ROC AUC Before: {roc_auc:.4f}")
    print(f"ROC AUC After: {roc_auc_res:.4f}")


#  Task 6: Pruning and Model Improvement
def pruning_and_model_improvement():
    """ Objective: Implement pruning techniques to prevent overfitting."""
    # 1. Apply pre-pruning techniques: set maximum depth, define minimum samples per leaf, establish
    # minimum impurity decrease
    X = df[numerical_features]
    y = df['default_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pre_pruned_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=20, random_state=42)
    pre_pruned_tree.fit(X_train, y_train)

    y_pred_pre_pruned = pre_pruned_tree.predict(X_test)
    print("Pre-pruned Model Evaluation:")
    print(classification_report(y_test, y_pred_pre_pruned))
    print(confusion_matrix(y_test, y_pred_pre_pruned))

    # 2. Implement post-pruning: use cost complexity pruning, create pruning path visualization
    # Generate the cost-complexity pruning path
    path = pre_pruned_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Find the best alpha value by testing each value
    models = []
    for ccp_alpha in ccp_alphas:
        # Create a decision tree with specific alpha
        tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        tree.fit(X_train, y_train)
        models.append((ccp_alpha, tree))

    # Plot the impurity vs. alpha values
    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas, impurities, marker='o', drawstyle="steps-post")
    plt.title("Impurity vs. Alpha for Cost Complexity Pruning")
    plt.xlabel("Effective Alpha")
    plt.ylabel("Total Impurity of Leaves")
    plt.show()

    # Evaluate performance at each alpha value
    for ccp_alpha, tree in models:
        y_pred = tree.predict(X_test)
        print(f"Model with ccp_alpha={ccp_alpha:.4f} - Accuracy: {tree.score(X_test, y_test):.4f}")
        print(classification_report(y_test, y_pred))

    # Choose the best model after pruning based on performance metrics
    best_model = models[np.argmax([tree.score(X_test, y_test) for _, tree in models])]
    best_ccp_alpha, best_tree = best_model
    print(f"Best model selected with ccp_alpha={best_ccp_alpha:.4f}")

    # Final evaluation on the best pruned tree
    y_pred_best = best_tree.predict(X_test)
    print("Best Pruned Model Evaluation:")
    print(classification_report(y_test, y_pred_best))
    print(confusion_matrix(y_test, y_pred_best))

    # Visualize the pruned tree (if required)
    plt.figure(figsize=(12, 8))
    plot_tree(best_tree, filled=True, feature_names=numerical_features, class_names=["No Default", "Default"])
    plt.show()

    # 3. Compare performance before and after pruning
    print("\nComparison of Performance Metrics Before and After Pruning:")
    print("Pre-pruned Model Accuracy:", pre_pruned_tree.score(X_test, y_test))
    print("Pruned Model Accuracy:", best_tree.score(X_test, y_test))



if __name__ == '__main__':
    df = pd.read_csv('credit_risk_sample_data.csv')

    numerical_features = ['age', 'annual_income', 'employment_duration', 'credit_score',
                          'loan_amount', 'loan_term', 'existing_debt', 'debt_to_income',
                          'late_payments_30_days', 'late_payments_90_days', 'num_credit_cards',
                          'num_bank_accounts']

    """ Task 2: Data Preparation and Initial Analysis """
    data_preparation()

    """ Task 3: Implementation of Gini Index and Entropy """
    gini_index_and_entropy()

    """ Task 4: Model Development and Tuning """
    model_development_and_tuning()

    """ Task 5: Model Evaluation and Performance Analysis """
    model_evaluation_performance_analysis()

    """ Task 6: Pruning and Model Improvement """
    pruning_and_model_improvement()
