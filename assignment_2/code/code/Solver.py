import pandas as pd
from GP import GeneticProgramming
from GPS2 import GeneticProgrammingS2
from GPT import GeneticProgrammingT
from FeatureMapper import FeatureMapper
from FeatureMapper2 import FeatureMapper2
import time
import random


num_generations = 100
seed_value = 117


# ===============================================> Train source 1 GP ========================================#
print("====================================== Source 1 Start =========================================")
# Read the CSV files 
X_train = pd.read_csv('../train_set/X_train.csv')
y_train = pd.read_csv('../train_set/y_train.csv')

# Create the source GP
gp_source = GeneticProgramming(population_size=100, seed=seed_value)

start_time_total = time.time()

# Generate initial population for the source GP
gp_source.generate_initial_population()

initial_train_accuracies = []

# Iterate through each individual in the initial population of the source GP
for individual in gp_source.population:
    # Make predictions on the training dataset
    train_predictions = [gp_source.make_prediction(individual.root, sample) for _, sample in X_train.iterrows()]
    
    # Evaluate fitness (accuracy) on the training dataset
    train_accuracy = gp_source.evaluate_fitness(train_predictions, y_train)
    initial_train_accuracies.append(train_accuracy)

# After evaluating performance on the training dataset, you can analyze the results
average_initial_train_accuracy = sum(initial_train_accuracies) / len(initial_train_accuracies)

# Print the initial population of the source GP
# gp_source.print_population()

# Evolve the population of the source GP
for generation in range(num_generations):
    print(f"Generation {generation + 1}")
    # Evolve the population (train and create next generation)
    gp_source.evolve_population(X_train, y_train)


# Find the fittest individual in the evolved population of the source GP
fittest_individual_source = max(gp_source.population, key=lambda x: x.fitness)

fittest_individual_source.visualize_tree(fittest_individual_source.root, "fittest_individual_s1")

# Read the CSV files into DataFrame objects for test data
X_test = pd.read_csv('../test_set/X_test.csv')
y_test = pd.read_csv('../test_set/y_test.csv')

# Make predictions on the test dataset using the fittest individual of the source GP
test_predictions_source = [gp_source.make_prediction(fittest_individual_source.root, sample) for _, sample in X_test.iterrows()]

# Evaluate test accuracy of the fittest individual of the source GP
test_accuracy_source = gp_source.evaluate_fitness(test_predictions_source, y_test)


print(f"Test accuracy of the fittest individual of the source GP: {test_accuracy_source * 100} %")

print("====================================== Source 1 Complete =======================================")


# ===============================================> Train source 2 GP ========================================#
print("====================================== Source 2 Start =========================================")

# Read the CSV files 
X_train_1 = pd.read_csv('../train_set_2/X_train.csv')
y_train_1 = pd.read_csv('../train_set_2/y_train.csv')

gp_source_2 = GeneticProgrammingS2(population_size=100, seed=seed_value)

# Generate initial population for the source GP
gp_source_2.generate_initial_population()

# gp_source_2.print_population()


initial_train_accuracies_2 = []

#Iterate through each individual in the initial population of the source 2 GP
for individual in gp_source_2.population:
    # Make predictions on the training dataset
    train_predictions = [gp_source_2.make_prediction(individual.root, sample) for _, sample in X_train_1.iterrows()]
    
    # Evaluate fitness (accuracy) on the training dataset
    train_accuracy = gp_source_2.evaluate_fitness(train_predictions, y_train_1)
    initial_train_accuracies_2.append(train_accuracy)

# Print the initial trained accuracies
# print("Initial trained accuracies:")
# for i, accuracy in enumerate(initial_train_accuracies_2):
#     print(f"Individual {i+1}: {accuracy}")


# After evaluating performance on the training dataset, you can analyze the results
average_initial_train_accuracy_2 = sum(initial_train_accuracies_2) / len(initial_train_accuracies_2)

#Print the initial population of the source 2 GP
# gp_source_2.print_population()

# Evolve the population of the source 2 GP
for generation in range(num_generations):
    print(f"Generation {generation + 1}")
    # Evolve the population (train and create next generation)
    gp_source_2.evolve_population(X_train_1, y_train_1)


X_test_1 = pd.read_csv('../test_set_2/X_test.csv')
y_test_1 = pd.read_csv('../test_set_2/y_test.csv')

# Find the fittest individual in the evolved population of the source 2 GP
fittest_individual_source_2 = max(gp_source_2.population, key=lambda x: x.fitness)

# Make predictions on the test dataset using the fittest individual of the source 2 GP
test_predictions_source_2 = [gp_source_2.make_prediction(fittest_individual_source_2.root, sample) for _, sample in X_test_1.iterrows()]

# Evaluate test accuracy of the fittest individual of the source 2 GP
test_accuracy_source_2 = gp_source_2.evaluate_fitness(test_predictions_source_2, y_test_1)

# Print the fittest individual of the source 2 GP
print("Fittest individual of the source 2 GP:")
fittest_individual_source_2.visualize_tree(fittest_individual_source_2.root, "fittest_individual_s2")

# print(f"Elapsed time for the source 2 GP: {elapsed_time_source_2} seconds")
print(f"Test accuracy of the fittest individual of the source 2 GP: {test_accuracy_source_2 * 100} %")

print("====================================== Source 2 Complete =========================================")

#================================================= Target GP ========================================#
print("====================================== Target Start ============================================")
# Start the timer
start_time = time.time()

# Get the top 50 fittest individuals from source 1 GP
top_fittest_source_1 = sorted(gp_source.population, key=lambda x: x.fitness, reverse=True)[:50]

# feature mapper 1
mapper = FeatureMapper(["BMI", "Glucose", "Insulin", "SkinThickness" , "DiabetesPedigreeFunction"], [ "AGE", "HbA1c" ," Chol" , "TG" , "BMI"])


# Map and replace features for each individual in the evolved population
for idx, individual in enumerate(top_fittest_source_1):
    individual.root = mapper.map_and_replace_features_s1_t(individual.root)


#Get the top 50 fittest individuals from source 2 GP
top_fittest_source_2 = sorted(gp_source_2.population, key=lambda x: x.fitness, reverse=True)[:50]

# feature mapper 2
mapper2 = FeatureMapper2(["bmi", "blood_glucose_level", "HbA1c_level", "age"], [ "AGE","HbA1c" ,"Chol" , "TG" , "BMI"])

for idx, individual in enumerate(top_fittest_source_2):
    individual.root = mapper2.map_and_replace_features_s2_t(individual.root)


# # Merge populations from source 1 and source 2 GPs
target_initial_population = top_fittest_source_1 + top_fittest_source_2

# # Shuffle the merged population
random.seed(42)
random.shuffle(target_initial_population)


# Read the training data for the target population
X_train_target = pd.read_csv('../train_set_3/x_train.csv')
y_train_target = pd.read_csv('../train_set_3/y_train.csv')

# Instantiate the GeneticProgrammingT class for the target GP
gp_target = GeneticProgrammingT(population_size=100, seed=seed_value)

# # Generate the initial population for the target GP
gp_target.population = target_initial_population

# print("Updated population after feature mapping:")
# for idx, individual in enumerate(gp_target.population):
#     individual.print_tree()

initial_train_accuracies_target = []

#Iterate through each individual in the initial population of the source 2 GP
for individual in gp_target.population:
    # Make predictions on the training dataset
    train_predictions = [gp_target.make_prediction(individual.root, sample) for _, sample in X_train_target.iterrows()]
    
    # Evaluate fitness (accuracy) on the training dataset
    train_accuracy = gp_target.evaluate_fitness(train_predictions, y_train_target)
    initial_train_accuracies_target.append(train_accuracy)

print("Initial train accuracies for the target GP:")
for i, accuracy in enumerate(initial_train_accuracies_target):
    print(f"Individual {i+1}: {accuracy}")



for generation in range(num_generations):
    print(f"Generation {generation + 1}")
    # Evolve the population (train and create next generation)
    gp_target.evolve_population(X_train_target, y_train_target)



X_test_target = pd.read_csv('../test_set_3/x_test.csv')
y_test_target = pd.read_csv('../test_set_3/y_test.csv')

# Find the fittest individual in the evolved population of the target GP
fittest_individual_target = max(gp_target.population, key=lambda x: x.fitness)

fittest_individual_target.visualize_tree(fittest_individual_target.root, "final-tree")

# Make predictions on the test dataset using the fittest individual of the target GP
test_predictions_target = [gp_target.make_prediction(fittest_individual_target.root, sample) for _, sample in X_test_target.iterrows()]

elapsed_time_source = time.time() - start_time
elapsed_time_total = time.time() - start_time_total



print("final predictions made by fittest individual in Target GP:")
print(test_predictions_target)

# # Evaluate test accuracy of the fittest individual of the target GP
test_accuracy_target = gp_target.evaluate_fitness(test_predictions_target, y_test_target)

print("====================================== Target Complete =========================================")

print(f"Test accuracy of the fittest individual of the source 1 GP: {test_accuracy_source * 100} %")
print(f"Test accuracy of the fittest individual of the source 2 GP: {test_accuracy_source_2 * 100} %")
print(f"Test accuracy of the fittest individual of the target GP: {test_accuracy_target * 100} %")

print(f"Elapsed time for the target GP: {elapsed_time_source} seconds")
print(f"Elapsed time total all the source GP and Target: {elapsed_time_total} seconds")



# ====================================> Target GP without training ===================================#

print("======================= Target GP without Transfer Learning: Start ============================")
# Read the CSV files 
X_train_target = pd.read_csv('../train_set_3/x_train.csv')
y_train_target = pd.read_csv('../train_set_3/y_train.csv')

# Create the source GP
gp_target_no_tl = GeneticProgrammingT(population_size=100, seed=seed_value)

# Start the timer
start_time2 = time.time()

# Generate initial population for the source GP
gp_target_no_tl.generate_initial_population()

initial_train_accuracies = []

# Iterate through each individual in the initial population of the source GP
for individual in gp_target_no_tl.population:
    # Make predictions on the training dataset
    train_predictions = [gp_target_no_tl.make_prediction(individual.root, sample) for _, sample in X_train_target.iterrows()]
    
    # Evaluate fitness (accuracy) on the training dataset
    train_accuracy = gp_target_no_tl.evaluate_fitness(train_predictions, y_train_target)
    initial_train_accuracies.append(train_accuracy)

# After evaluating performance on the training dataset, you can analyze the results
average_initial_train_accuracy = sum(initial_train_accuracies) / len(initial_train_accuracies)

# Print the initial population of the source GP
# gp_source.print_population()

# Evolve the population of the source GP
for generation in range(num_generations):
    print(f"Generation {generation + 1}")
    # Evolve the population (train and create next generation)
    gp_target_no_tl.evolve_population(X_train_target, y_train_target)


# Find the fittest individual in the evolved population of the source GP
fittest_individual_source = max(gp_target_no_tl.population, key=lambda x: x.fitness)

fittest_individual_source.visualize_tree(fittest_individual_source.root, "fittest_individual_no_tl")

# Read the CSV files into DataFrame objects for test data
X_test_target = pd.read_csv('../test_set_3/x_test.csv')
y_test_target = pd.read_csv('../test_set_3/y_test.csv')

# Make predictions on the test dataset using the fittest individual of the source GP
test_predictions_source = [gp_target_no_tl.make_prediction(fittest_individual_source.root, sample) for _, sample in X_test_target.iterrows()]

elapsed_time_source2 = time.time() - start_time2

print(f"Elapsed time for the source GP: {elapsed_time_source2} seconds")

# Evaluate test accuracy of the fittest individual of the source GP
test_accuracy_source_no_tl = gp_target_no_tl.evaluate_fitness(test_predictions_source, y_test_target)


print(f"Test accuracy of the fittest individual of the source GP: {test_accuracy_source_no_tl * 100} %")


print("======================= Target GP without Transfer Learning: Complete ============================")