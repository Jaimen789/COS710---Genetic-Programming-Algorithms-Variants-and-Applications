import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Set the option to suppress the FutureWarning
pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv('diabetes.csv')

columns_to_fill = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df[columns_to_fill] = df[columns_to_fill].replace(0, pd.NA)

# Fill NaN values with the mean of their respective columns
df.fillna(df.mean(), inplace=True)
df.infer_objects(copy=False) 

print("Mean of each attribute:")
print(df.mean())

df.to_csv('diabetes_processed.csv', index=False)

#Split features (X) and target variable (y)
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']                # Target variable


# Split the data into training and test sets (70% train, 30% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create directories if they don't exist
if not os.path.exists('train_set'):
    os.makedirs('train_set')

if not os.path.exists('test_set'):
    os.makedirs('test_set')

# Save training set
X_train.to_csv('train_set/X_train.csv', index=False)
y_train.to_csv('train_set/y_train.csv', index=False)

# Save testing set
X_test.to_csv('test_set/X_test.csv', index=False)
y_test.to_csv('test_set/y_test.csv', index=False)

original_counts = y.value_counts()
print("Original Dataset:")
print("Non-Diabetic (Outcome 0):", original_counts[0])
print("Diabetic (Outcome 1):", original_counts[1])

# Count the occurrences of each class in the training set
train_counts = y_train.value_counts()
print("\nTraining Set:")
print("Non-Diabetic (Outcome 0):", train_counts[0])
print("Diabetic (Outcome 1):", train_counts[1])

# Count the occurrences of each class in the test set
test_counts = y_test.value_counts()
print("\nTest Set:")
print("Non-Diabetic (Outcome 0):", test_counts[0])
print("Diabetic (Outcome 1):", test_counts[1])



# Optionally, you can save the split datasets to new CSV files
# X_train.to_csv('training_set/X_train.csv', index=False)
# X_test.to_csv('test_set/X_test.csv', index=False)
# y_train.to_csv('training_set/y_train.csv', index=False)
# y_test.to_csv('test_set/y_test.csv', index=False)
