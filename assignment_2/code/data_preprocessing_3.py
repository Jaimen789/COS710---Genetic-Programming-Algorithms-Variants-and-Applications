import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# Load the dataset
data = pd.read_csv("Dataset_of_Diabetes.csv")

# Drop unnecessary columns
data = data.drop(columns=['ID', 'No_Pation', 'Gender'])

# Encode the CLASS column to numerical values
data['CLASS'] = data['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})

# Separate features and target variable
X = data.drop(columns=['CLASS'])
y = data['CLASS']

# Group data by CLASS and count instances of each class label
class_counts = data.groupby('CLASS').size()

# Print class counts
for class_label, count in class_counts.items():
    if class_label == 0:
        print(f"Class N: {count} instances")
    elif class_label == 1:
        print(f"Class P: {count} instances")
    elif class_label == 2:
        print(f"Class Y: {count} instances")

# Check for missing values in target variable
if y.isnull().values.any():
    # Handle missing values in target variable
    print("Target variable contains missing values. Imputing...")
    imputer = SimpleImputer(strategy='most_frequent')
    y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Apply SMOTE to balance the classes
# Apply SMOTE to balance the classes with specified sampling strategy
sampling_strategy = {0: 1500}  # Specify desired number of samples for each class
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Calculate the number of samples added
num_samples_added = len(y_resampled) - len(y)
print(f"Number of samples added: {num_samples_added}")

# Count occurrences of each class label in the resampled data
resampled_class_counts = pd.Series(y_resampled).value_counts()

# Print the counts
print("Number of non-diabetic (N) instances added:", resampled_class_counts[0])
print("Number of diabetic (Y) instances added:", resampled_class_counts[1])

# Append resampled data to the original dataset
resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame({'CLASS': y_resampled})], axis=1)

# Save the updated dataset to CSV
resampled_data.to_csv("Dataset_of_Diabetes.csv", index=False)

# Initialize Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf.fit(X_resampled, y_resampled)

# Get feature importances
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

print("Feature importances:")
print(feature_importances)

# Determine which attribute affects each class the most
for class_label in range(3):
    print(f"\nMost important attributes for class {class_label}:")
    class_features = feature_importances.index[rf.feature_importances_.argsort()[::-1]]
    for feature in class_features:
        print(f"{feature}: {feature_importances.loc[feature]['importance']}")

# Group data by CLASS
grouped_data = data.groupby('CLASS')

# Calculate average and range for each attribute
attribute_stats = {}
for attribute in X.columns:
    attribute_stats[attribute] = {}
    for class_label, group in grouped_data:
        avg = group[attribute].mean()
        data_range = group[attribute].max() - group[attribute].min()
        attribute_stats[attribute][class_label] = {'average': avg, 'range': (group[attribute].min(), group[attribute].max())}

# Print attribute statistics
for attribute, stats in attribute_stats.items():
    print(f"\nAttribute: {attribute}")
    for class_label, values in stats.items():
        print(f"Class: {class_label}")
        print(f"  Average: {values['average']}")
        print(f"  Range: {values['range']}")

# Calculate range for 90% of values for each attribute
attribute_90_range = {}
for attribute in X.columns:
    attribute_90_range[attribute] = {}
    for class_label, group in grouped_data:
        lower_quantile = group[attribute].quantile(0.05)
        upper_quantile = group[attribute].quantile(0.95)
        attribute_90_range[attribute][class_label] = (lower_quantile, upper_quantile)

# Print attribute statistics for 90% range
for attribute, stats in attribute_90_range.items():
    print(f"\nAttribute: {attribute}")
    for class_label, values in stats.items():
        print(f"Class: {class_label}")
        print(f"  90% Range: {values}")




# Load the resampled dataset
resampled_data = pd.read_csv("Dataset_of_Diabetes.csv")

# Filter data to remove rows with class label 'P'
filtered_data = resampled_data[resampled_data['CLASS'] != 1]

# Convert class label 'Y' to '1'
filtered_data['CLASS'] = filtered_data['CLASS'].replace(2, 1)


class_0_data = filtered_data[filtered_data['CLASS'] == 0]
class_1_data = filtered_data[filtered_data['CLASS'] == 1]
sampled_class_0_data = class_0_data.sample(frac=0.7, random_state=42)
sampled_class_1_data = class_1_data.sample(frac=0.3, random_state=42)
final_data = pd.concat([sampled_class_0_data, sampled_class_1_data])

X = final_data.drop(columns=['CLASS'])
y = final_data['CLASS']

# Save the filtered dataset to a new CSV file
# filtered_data.to_csv("Dataset_of_Diabetes_cleaned.csv", index=False)

# Filter data to remove rows with class label 'P'
# filtered_data = data[data['CLASS'] != 1]

# Convert class label 'Y' to '1'
# filtered_data['CLASS'] = filtered_data['CLASS'].replace(2, 1)

# Count the total records after filtering out 'P'
total_records_filtered = len(filtered_data)
print(f"Total records after removing 'P': {total_records_filtered}")

# Save the filtered dataset to a new CSV file
filtered_data.to_csv("Dataset_of_Diabetes_cleaned.csv", index=False)

# Read the filtered dataset without class 'P'
filtered_data_no_P = pd.read_csv("Dataset_of_Diabetes_cleaned.csv")

# Count instances of each class label in the filtered data
class_counts_filtered_no_P = filtered_data_no_P['CLASS'].value_counts()

# Print the counts
print("Number of non-diabetic (N) instances:", class_counts_filtered_no_P[0])
print("Number of diabetic (Y) instances:", class_counts_filtered_no_P[1])

# Load the filtered dataset
filtered_data = pd.read_csv("Dataset_of_Diabetes_cleaned.csv")

# Drop rows with missing values
filtered_data.dropna(inplace=True)

# Separate the data into features (X) and target variable (y)
X = filtered_data.drop(columns=['CLASS'])
y = filtered_data['CLASS']

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Check if the folders exist, if not, create them
if not os.path.exists('train_set_3'):
    os.makedirs('train_set_3')
if not os.path.exists('test_set_3'):
    os.makedirs('test_set_3')

# Save training and testing sets to CSV files
X_train.to_csv("train_set_3/x_train.csv", index=False)
y_train.to_csv("train_set_3/y_train.csv", index=False)
X_test.to_csv("test_set_3/x_test.csv", index=False)
y_test.to_csv("test_set_3/y_test.csv", index=False)

# Load y_train and y_test from CSV files
y_train = pd.read_csv("train_set_3/y_train.csv")
y_test = pd.read_csv("test_set_3/y_test.csv")

# Count occurrences of diabetic and non-diabetic people in y_train and y_test
train_class_counts = y_train['CLASS'].value_counts()
test_class_counts = y_test['CLASS'].value_counts()

print("Training set class distribution:")
print(train_class_counts)
print("\nTesting set class distribution:")
print(test_class_counts)






















# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
# import os

# # Load the dataset
# data = pd.read_csv("Dataset_of_Diabetes.csv")

# # Drop unnecessary columns
# data = data.drop(columns=['ID', 'No_Pation', 'Gender'])

# # Encode the CLASS column to numerical values
# data['CLASS'] = data['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})

# # Separate features and target variable
# X = data.drop(columns=['CLASS'])
# y = data['CLASS']

# # Check for missing values in target variable
# if y.isnull().values.any():
#     # Handle missing values in target variable
#     print("Target variable contains missing values. Imputing...")
#     imputer = SimpleImputer(strategy='most_frequent')
#     y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# # Initialize Random Forest Classifier
# rf = RandomForestClassifier(n_estimators=100, random_state=42)

# # Fit the model
# rf.fit(X, y)

# # Get feature importances
# feature_importances = pd.DataFrame(rf.feature_importances_,
#                                    index = X.columns,
#                                    columns=['importance']).sort_values('importance', ascending=False)

# print("Feature importances:")
# print(feature_importances)

# # Determine which attribute affects each class the most
# for class_label in range(3):
#     print(f"\nMost important attributes for class {class_label}:")
#     class_features = feature_importances.index[rf.feature_importances_.argsort()[::-1]]
#     for feature in class_features:
#         print(f"{feature}: {feature_importances.loc[feature]['importance']}")



# # Group data by CLASS
# grouped_data = data.groupby('CLASS')

# # Calculate average and range for each attribute
# attribute_stats = {}
# for attribute in X.columns:
#     attribute_stats[attribute] = {}
#     for class_label, group in grouped_data:
#         avg = group[attribute].mean()
#         data_range = group[attribute].max() - group[attribute].min()
#         attribute_stats[attribute][class_label] = {'average': avg, 'range': (group[attribute].min(), group[attribute].max())}

# # Print attribute statistics
# for attribute, stats in attribute_stats.items():
#     print(f"\nAttribute: {attribute}")
#     for class_label, values in stats.items():
#         print(f"Class: {class_label}")
#         print(f"  Average: {values['average']}")
#         print(f"  Range: {values['range']}")



# # Calculate range for 90% of values for each attribute
# attribute_90_range = {}
# for attribute in X.columns:
#     attribute_90_range[attribute] = {}
#     for class_label, group in grouped_data:
#         lower_quantile = group[attribute].quantile(0.05)
#         upper_quantile = group[attribute].quantile(0.95)
#         attribute_90_range[attribute][class_label] = (lower_quantile, upper_quantile)

# # Print attribute statistics for 90% range
# for attribute, stats in attribute_90_range.items():
#     print(f"\nAttribute: {attribute}")
#     for class_label, values in stats.items():
#         print(f"Class: {class_label}")
#         print(f"  90% Range: {values}")


# # Group data by CLASS and count instances of each class label
# class_counts = data.groupby('CLASS').size()

# # Print class counts
# for class_label, count in class_counts.items():
#     if class_label == 0:
#         print(f"Class N: {count} instances")
#     elif class_label == 1:
#         print(f"Class P: {count} instances")
#     elif class_label == 2:
#         print(f"Class Y: {count} instances")


# # Filter data to remove rows with class label 'P'
# filtered_data = data[data['CLASS'] != 1]

# # Convert class label 'Y' to '1'
# filtered_data['CLASS'] = filtered_data['CLASS'].replace(2, 1)

# # Count the total records after filtering out 'P'
# total_records_filtered = len(filtered_data)
# print(f"Total records after removing 'P': {total_records_filtered}")

# # Save the filtered dataset to a new CSV file
# filtered_data.to_csv("Dataset_of_Diabetes_cleaned.csv", index=False)

# # Read the filtered dataset without class 'P'
# filtered_data_no_P = pd.read_csv("Dataset_of_Diabetes_cleaned.csv")

# # Count instances of each class label in the filtered data
# class_counts_filtered_no_P = filtered_data_no_P['CLASS'].value_counts()

# # Print the counts
# print("Number of non-diabetic (N) instances:", class_counts_filtered_no_P[0])
# print("Number of diabetic (Y) instances:", class_counts_filtered_no_P[1])

# # Load the filtered dataset
# filtered_data = pd.read_csv("Dataset_of_Diabetes_cleaned.csv")

# # Drop rows with missing values
# filtered_data.dropna(inplace=True)

# # Separate the data into features (X) and target variable (y)
# X = filtered_data.drop(columns=['CLASS'])
# y = filtered_data['CLASS']

# # Split the data into training and testing sets with stratification
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# # Check if the folders exist, if not, create them
# if not os.path.exists('train_set_3'):
#     os.makedirs('train_set_3')
# if not os.path.exists('test_set_3'):
#     os.makedirs('test_set_3')

# # Save training and testing sets to CSV files
# X_train.to_csv("train_set_3/x_train.csv", index=False)
# y_train.to_csv("train_set_3/y_train.csv", index=False)
# X_test.to_csv("test_set_3/x_test.csv", index=False)
# y_test.to_csv("test_set_3/y_test.csv", index=False)

# # Load y_train and y_test from CSV files
# y_train = pd.read_csv("train_set_3/y_train.csv")
# y_test = pd.read_csv("test_set_3/y_test.csv")

# # Count occurrences of diabetic and non-diabetic people in y_train and y_test
# train_class_counts = y_train['CLASS'].value_counts()
# test_class_counts = y_test['CLASS'].value_counts()

# print("Training set class distribution:")
# print(train_class_counts)
# print("\nTesting set class distribution:")
# print(test_class_counts)

