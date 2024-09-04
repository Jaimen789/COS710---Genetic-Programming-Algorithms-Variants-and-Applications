import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2
# Load data
data = pd.read_csv('diabetes_processed.csv')

# Display the first few rows of the dataset
print(data.head())


# Feature Extraction/Selection
X = data[data.columns[:-1]]  # Features
y = data['Outcome']           # Target variable

# Selecting top 5 features

#chi-squared statistic (how score is calculated)
#Calculate Observed Frequenciesc-> the observed frequencies of occurrence are calculated. 
#Calculate Expected frequency -> To compute expected frequencies, you would typically calculate the marginal probabilities of each feature and each class (outcome). For example, if you have a binary classification problem with two classes (0 and 1), you'd calculate the probability of each feature occurring with class 0 and class 1 separately.
#Compute Chi-Squared Statistic -> the expected frequencies under the assumption of independence between the feature and the target variable are computed
#Aggregate Chi-Squared Values -> This ensures that only the most informative features, as determined by the chi-squared test, are retained for modeling


# The chi-squared statistic measures the dependence between categorical variables. 
# In feature selection, it quantifies the discrepancy between the observed frequency 
# and the expected frequency of an event occurring under independence. Essentially, it 
# helps determine whether the presence of a particular feature significantly influences the 
# target variable or not.

# In this output, the score associated with each feature indicates how much that feature contributes 
# to the prediction of the target variable. Higher scores imply stronger predictive power or greater 
# relevance of the feature for predicting the outcome.



best_features = SelectKBest(score_func=chi2, k=8)
fit = best_features.fit(X, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ['Feature', 'Score']
print(feature_scores.nlargest(8, 'Score'))

attribute_means = data.mean()

print("Average of each attribute:")
print(attribute_means)


non_diabetic_subset = data[data['Outcome'] == 0]
diabetic_subset = data[data['Outcome'] == 1]


non_diabetic_avg = non_diabetic_subset.mean()
diabetic_avg = diabetic_subset.mean()

print("Average values for non-diabetic individuals:")
print(non_diabetic_avg)
print("\nAverage values for diabetic individuals:")
print(diabetic_avg)

# Group the data by the 'Outcome'.
# Calculate the mean and standard deviation for each attribute for each outcome group.
# Define a range around the mean within, say, 1 standard deviation.
# Print or visualize these ranges for each attribute and outcome.

grouped = data.groupby('Outcome')

# Calculate mean and standard deviation for each attribute for each outcome
mean_std = grouped.agg({'Pregnancies': ['mean', 'std'],
                        'Glucose': ['mean', 'std'],
                        'BloodPressure': ['mean', 'std'],
                        'SkinThickness': ['mean', 'std'],
                        'Insulin': ['mean', 'std'],
                        'BMI': ['mean', 'std'],
                        'DiabetesPedigreeFunction': ['mean', 'std'],
                        'Age': ['mean', 'std']})

# Define a function to get the range
def get_range(mean, std):
    lower_bound = mean - std
    upper_bound = mean + std
    return lower_bound, upper_bound

# Print the ranges for each attribute and outcome
for attribute in mean_std.columns.levels[0]:
    print(f"Attribute: {attribute}")
    for outcome in [0, 1]:
        mean = mean_std[attribute]['mean'][outcome]
        std = mean_std[attribute]['std'][outcome]
        lower_bound, upper_bound = get_range(mean, std)
        print(f"Outcome {outcome}: Mean = {mean}, Std = {std}, Range = ({lower_bound}, {upper_bound})")
    print()


# Calculate the 5th and 90th percentiles for each attribute
print("Percentile where 90% lie")
percentiles = grouped.aggregate(lambda x: (x.quantile(0.05), x.quantile(0.90)))

# Print the range for each attribute and outcome
for attribute in percentiles.columns:
    print(f"Attribute: {attribute}")
    for outcome in [0, 1]:
        lower_bound, upper_bound = percentiles[attribute].iloc[outcome]
        print(f"Outcome {outcome}: Range = ({lower_bound}, {upper_bound})")
    print()


top_features = feature_scores.nlargest(8, 'Score')



# Filter the dataset to include only diabetic individuals
diabetic_subset = data[data['Outcome'] == 1]

# Calculate the 25th percentile for insulin among diabetic individuals (Type 1 Diabetes - Low Range)
low_range_diabetic = diabetic_subset['Insulin'].quantile(0.25)

# Calculate the 75th percentile for insulin among diabetic individuals (Type 2 Diabetes - High Range)
high_range_diabetic = diabetic_subset['Insulin'].quantile(0.75)

print("Low Range for Insulin (Type 1 Diabetes):", low_range_diabetic)
print("High Range for Insulin (Type 2 Diabetes):", high_range_diabetic)


# Count the number of records falling into the low and high ranges
records_in_low_range = diabetic_subset[(diabetic_subset['Insulin'] >= low_range_diabetic) & (diabetic_subset['Insulin'] < high_range_diabetic)].shape[0]
records_in_high_range = diabetic_subset[diabetic_subset['Insulin'] >= high_range_diabetic].shape[0]

print("Low Range for Insulin (Diabetic):", low_range_diabetic)
print("High Range for Insulin (Diabetic):", high_range_diabetic)
print("Number of Records in Low Range:", records_in_low_range)
print("Number of Records in High Range:", records_in_high_range)



# Create a bar plot
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Score'], color='skyblue')
plt.xlabel('Score')
plt.ylabel('Feature')
plt.title('Top Features Selected by Chi-Squared Test')
plt.gca().invert_yaxis()  # Invert y-axis to display features with highest scores at the top
plt.show()



#outputs

# This matrix shows the correlation coefficient between each pair of features.
# Positive values indicate a positive correlation, negative values indicate 
# a negative  correlation, and values close to zero indicate no correlation.
# higher glucose levels might correlate with a higher likelihood of diabetes.

# Calculate correlation matrix
correlation_matrix = data.corr()
print(correlation_matrix)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Rotate x-axis labels
plt.xticks(rotation=45, horizontalalignment='right' , fontsize = 5)

# Add title
plt.title('Correlation Matrix')

# Show plot
plt.show()


# # Create pairplot
pairplot = sns.pairplot(data, markers='.', diag_kind='kde', plot_kws={'alpha': 0.5})

# Rotate y-axis labels
for ax in pairplot.axes.flatten():
    ax.set_ylabel(ax.get_ylabel(), rotation = 90 , fontsize = 5)

plt.show()


# #histograms
for column in X.columns:
    plt.figure()
    plt.hist(data[column])
    plt.xlabel(column)               # Adding x-axis label
    plt.ylabel('Frequency')          # Adding y-axis label
    plt.title('Histogram of ' + column)  # Adding title
    plt.show()



# # The chi-squared test identifies features that are most likely to be related to the outcome variable (diabetes).
# # Features with higher chi-squared scores are considered more informative for predicting diabetes.
# # Glucose, Insulin, Age, BMI, and Pregnancies emerge as key features in predicting diabetes based on their scores and 
# # correlations with the outcome.These findings suggest that high glucose levels, high insulin levels, older age, higher
# #  BMI, and number of pregnancies might be significant factors contributing to the likelihood of diabetes.