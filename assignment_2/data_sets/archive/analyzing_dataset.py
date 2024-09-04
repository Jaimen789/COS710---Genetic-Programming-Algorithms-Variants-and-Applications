import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('cleaned_diabetes_dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Count how many diabetic and non-diabetic people there are
diabetic_count = data[data['diabetes'] == 1]['diabetes'].count()
non_diabetic_count = data[data['diabetes'] == 0]['diabetes'].count()

print("Diabetic Count:", diabetic_count)
print("Non-Diabetic Count:", non_diabetic_count)


# Feature Extraction/Selection
X = data[data.columns[:-1]]  # Features
y = data['diabetes']         # Target variable

# Convert categorical variables into numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Selecting top 5 features using chi-squared test
best_features = SelectKBest(score_func=chi2, k=8)
fit = best_features.fit(X, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ['Feature', 'Score']
print(feature_scores.nlargest(8, 'Score'))

# Statistical Analysis
# Exclude 'smoking_history' and 'gender' from the statistical analysis
attributes_to_include = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'heart_disease']
attribute_means = data[attributes_to_include].mean()
print("Average of each attribute:")
print(attribute_means)

non_diabetic_subset = data[data['diabetes'] == 0]
diabetic_subset = data[data['diabetes'] == 1]

non_diabetic_avg = non_diabetic_subset[attributes_to_include].mean()
diabetic_avg = diabetic_subset[attributes_to_include].mean()

print("Average values for non-diabetic individuals:")
print(non_diabetic_avg)
print("\nAverage values for diabetic individuals:")
print(diabetic_avg)

# Group the data by the 'diabetes'.
# Calculate the mean and standard deviation for each attribute for each outcome group.
# Define a range around the mean within, say, 1 standard deviation.
# Print or visualize these ranges for each attribute and outcome.
grouped = data.groupby('diabetes')

mean_std = grouped.agg({'age': ['mean', 'std'],
                        'bmi': ['mean', 'std'],
                        'blood_glucose_level': ['mean', 'std'],
                        'HbA1c_level': ['mean', 'std']})

def get_range(mean, std):
    lower_bound = mean - std
    upper_bound = mean + std
    return lower_bound, upper_bound

for attribute in mean_std.columns.levels[0]:
    if attribute in attributes_to_include:
        print(f"Attribute: {attribute}")
        for outcome in [0, 1]:
            mean = mean_std[attribute]['mean'][outcome]
            std = mean_std[attribute]['std'][outcome]
            lower_bound, upper_bound = get_range(mean, std)
            print(f"Outcome {outcome}: Mean = {mean}, Std = {std}, Range = ({lower_bound}, {upper_bound})")
        print()

# Calculate the 5th and 90th percentiles for each attribute
print("Percentile where 90% lie")
percentiles = grouped[attributes_to_include].aggregate(lambda x: (x.quantile(0.05), x.quantile(0.90)))

for attribute in percentiles.columns:
    print(f"Attribute: {attribute}")
    for outcome in [0, 1]:
        lower_bound, upper_bound = percentiles[attribute].iloc[outcome]
        print(f"Outcome {outcome}: Range = ({lower_bound}, {upper_bound})")
    print()

top_features = feature_scores.nlargest(8, 'Score')

# Visualizations
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Score'], color='skyblue')
plt.xlabel('Score')
plt.ylabel('Feature')
plt.title('Top Features Selected by Chi-Squared Test')
plt.gca().invert_yaxis()
plt.show()

# Calculate correlation matrix
correlation_matrix = data[attributes_to_include + ['diabetes']].corr()
print(correlation_matrix)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.xticks(rotation=45, horizontalalignment='right' , fontsize=5)
plt.title('Correlation Matrix')
plt.show()

# Create pairplot
pairplot = sns.pairplot(data[attributes_to_include + ['diabetes']], markers='.', diag_kind='kde', plot_kws={'alpha': 0.5})
for ax in pairplot.axes.flatten():
    ax.set_ylabel(ax.get_ylabel(), rotation=90, fontsize=5)
plt.show()

# Histograms
for column in X.columns:
    plt.figure()
    plt.hist(data[column])
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title('Histogram of ' + column)
    plt.show()

# Blood Glucose ranges
low_range_diabetic = diabetic_subset['blood_glucose_level'].quantile(0.25)
high_range_diabetic = diabetic_subset['blood_glucose_level'].quantile(0.75)
print("Low Range for Blood Glucose (Type 1 Diabetes):", low_range_diabetic)
print("High Range for Blood Glucose (Type 2 Diabetes):", high_range_diabetic)

records_in_low_range = diabetic_subset[(diabetic_subset['blood_glucose_level'] >= low_range_diabetic) & (diabetic_subset['blood_glucose_level'] < high_range_diabetic)].shape[0]
records_in_high_range = diabetic_subset[diabetic_subset['blood_glucose_level'] >= high_range_diabetic].shape[0]
print("Number of Records in Low Range:", records_in_low_range)
print("Number of Records in High Range:", records_in_high_range)
