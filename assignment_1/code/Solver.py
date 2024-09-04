import pandas as pd
from GP import GeneticProgramming
import time


# Read the CSV files 
X_train = pd.read_csv('../training_set/X_train.csv')
y_train = pd.read_csv('../training_set/y_train.csv')


num_generations = 100

gp = GeneticProgramming(population_size=100, seed=69)

# Start the timer
start_time = time.time()

# Generate initial population
gp.generate_initial_population()

initial_train_accuracies = []

# Iterate through each individual in the initial population
for individual in gp.population:
    # Make predictions on the training dataset
    train_predictions = [gp.make_prediction(individual.root, sample) for _, sample in X_train.iterrows()]
    
    # Evaluate fitness (accuracy) on the training dataset
    train_accuracy = gp.evaluate_fitness(train_predictions, y_train)
    initial_train_accuracies.append(train_accuracy)

# After evaluating performance on the training dataset, you can analyze the results
average_initial_train_accuracy = sum(initial_train_accuracies) / len(initial_train_accuracies)


gp.check_unique_population()
gp.print_population()

# Iterate through each generation
for generation in range(num_generations):
    print(f"Generation {generation + 1}")
    
    # Evolve the population (train and create next generation)
    gp.evolve_population(X_train, y_train)

# Measure the elapsed time
elapsed_time = time.time() - start_time


# After the loop finishes, you can print the final population, get top and bottom individuals.
print("Top 10 individuals:")
top_individuals = gp.get_top_individuals()

for i, (individual, fitness) in enumerate(top_individuals[:10]): 
    print(f"Individual {i+1}: Fitness = {fitness}")

print("Bottom 10 individuals:")
bottom_individuals = gp.get_bottom_individuals()

for i, (individual, fitness) in enumerate(bottom_individuals):
    print(f"Individual {i+1}: Fitness = {fitness}")

gp.check_unique_population()

# Read the CSV files into DataFrame objects for test data
X_test = pd.read_csv('../test_set/X_test.csv')
y_test = pd.read_csv('../test_set/y_test.csv')

# Find the fittest individual in the evolved population
fittest_individual = max(gp.population, key=lambda x: x.fitness)

print("fittest individual in population")
print(fittest_individual.fitness)
fittest_individual.print_tree()
fittest_individual.visualize_tree(fittest_individual.root, "fittest_individual")

# Make predictions on the test dataset using the fittest individual
test_predictions = [gp.make_prediction(fittest_individual.root, sample) for _, sample in X_test.iterrows()]

# Evaluate test accuracy
test_accuracy = gp.evaluate_fitness(test_predictions, y_test)



gp.print_population()


print(f"Elapsed time: {elapsed_time} seconds")


print(f"Metric calculations")


# Extract ground truth labels from the DataFrame
y_test_values = y_test['Outcome'].values

# Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
TP = sum((pred == 1) and (true == 1) for pred, true in zip(test_predictions, y_test_values))
TN = sum((pred == 0) and (true == 0) for pred, true in zip(test_predictions, y_test_values))
FP = sum((pred == 1) and (true == 0) for pred, true in zip(test_predictions, y_test_values))
FN = sum((pred == 0) and (true == 1) for pred, true in zip(test_predictions, y_test_values))

print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)

# Calculate Precision
precision = TP / (TP + FP) if (TP + FP) != 0 else 0

# Calculate Recall
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

# Calculate F1-score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Calculate Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(f"Test accuracy of the fittest individual: {test_accuracy * 100} %")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)


















# Initialize a list to store test accuracies of evolved individuals
# evolved_test_accuracies = []

# # Iterate through each individual in the evolved population
# for individual in gp.population:
#     # Make predictions on the test dataset
#     test_predictions = [gp.make_prediction(individual.root, sample) for _, sample in X_test.iterrows()]
    
#     # Evaluate test accuracy
#     test_accuracy = gp.evaluate_fitness(test_predictions, y_test)
#     evolved_test_accuracies.append(test_accuracy)

# # After evaluating performance on the test dataset, calculate the average test accuracy
# average_evolved_test_accuracy = sum(evolved_test_accuracies) / len(evolved_test_accuracies)


# print(f"Average training accuracy of initial population: {average_initial_train_accuracy}")
# print(f"Average test accuracy of evolved individuals: {average_evolved_test_accuracy}")

# # You can also perform further analysis such as visualizing the distribution of test accuracies,
# # comparing with the training set performance, etc.

# gp.print_population()
