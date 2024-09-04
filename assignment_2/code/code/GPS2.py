import os
import random
from TreeS2 import TreeS2


# random.seed(42)

class GeneticProgrammingS2:
    
    feature_names =  ["bmi", "blood_glucose_level", "HbA1c_level",  "age"]
    
    def __init__(self, population_size=100, seed=55):
        self.population = [TreeS2(self.feature_names) for _ in range(population_size)]
        self.seed = seed
        random.seed(seed)
    
    def generate_initial_population(self):

        # grow population generation only
        # for tree in self.population:
        #     tree.generate_random_individual(method = 'grow', population = 100)'

        # full population generation only
        # for tree in self.population:
        #     tree.generate_random_individual(method = 'full', population = 100)
        
        # ramped half and half
        for i in range(len(self.population)):
            # Alternate between "grow" and "full" methods based on the index
            if i % 2 == 0:
                self.population[i].generate_random_individual(method='grow')
            else:
                self.population[i].generate_random_individual(method='full')


    def print_population(self):
        for i, tree in enumerate(self.population):
            tree_height = self.calculate_tree_height(tree.root)

            print(f"Individual {i + 1}: tree height: {tree_height}")

            #self.print_tree(tree.root, indent=0)

            #Visualize the tree using graphviz
            # dot = tree.to_dot()
            # dot.render(f'tree_{i + 1}', format='png', cleanup=True)  # Render the tree to a PNG file

            # # Display the DOT representation in text format
            # print(dot.source)

    def train_population(self, X_train, y_train):
        print("Training Source 2 GP")
        for i, tree in enumerate(self.population):
            predictions = []

            for index, sample in X_train.iterrows():
                prediction = self.make_prediction(tree.root, sample)
                predictions.append(prediction)

            fitness_score = self.evaluate_fitness(predictions, y_train)

            # Print debugging information
            #print(f"Individual {i + 1}:")
            #print(f"Predictions: {predictions}")
            #print(f"Fitness Score: {fitness_score}")

            tree.fitness = fitness_score

    def evolve_population(self, X_train, y_train):
        # Train the initial population
        self.train_population(X_train, y_train)

        # Perform tournament selection and create a new population
        new_population = []

        # Check if the population is empty or smaller than the tournament size
        print(f"Current population size: {len(self.population)}")

        if len(self.population) < 2:
            print("Error: Population size is too small.")
            return

        # Create the next generation by repeating selection, crossover, and mutation
        # Perform tournament selection

        parent1_individual, parent1_fitness = self.perform_tournament_selection()
        parent2_individual, parent2_fitness = self.perform_tournament_selection()

        # Perform crossover to create offspring    
        offspring1, offspring2 = parent1_individual.perform_crossover(parent1_individual, parent2_individual)
 

        # Perform mutation
        mutation_rate = 0.85

        offspring1.perform_mutation(mutation_rate)
        offspring2.perform_mutation(mutation_rate)

        # offspring1.visualize_tree(offspring1.root, "mutated offspring_1")
        # offspring2.visualize_tree(offspring2.root, "mutated offspring_2")


        # Evaluate fitness of offspring
        offspring1_predictions = []
        offspring2_predictions = []

        for index, sample in X_train.iterrows():
            offspring1_prediction = self.make_prediction(offspring1.root, sample)
            offspring1_predictions.append(offspring1_prediction)

            offspring2_prediction = self.make_prediction(offspring2.root, sample)
            offspring2_predictions.append(offspring2_prediction)

        offspring1_fitness = self.evaluate_fitness(offspring1_predictions, y_train)
        offspring2_fitness = self.evaluate_fitness(offspring2_predictions, y_train)

        offspring1.fitness = offspring1_fitness
        offspring2.fitness = offspring2_fitness

        # print("parent 1 fitness: " + str(parent1_fitness))
        # print("parent 2 fitness: " + str(parent2_fitness))

        # print("offspring 1 fitness: " + str(offspring1_fitness))
        # print("offspring 2 fitness: " + str(offspring2_fitness))

        weakest_index1 = None
        weakest_index2 = None
        weakest_fitness1 = float('inf')  # Set to infinity initially
        weakest_fitness2 = float('inf')  # Set to infinity initially

        # Find the weakest individuals in the population
        for i, individual in enumerate(self.population):
            if individual.fitness < weakest_fitness1:
                weakest_fitness2 = weakest_fitness1
                weakest_index2 = weakest_index1
                weakest_fitness1 = individual.fitness
                weakest_index1 = i
            elif individual.fitness < weakest_fitness2:
                weakest_fitness2 = individual.fitness
                weakest_index2 = i

        # Check for duplicates before adding offspring to the population
        if offspring1 not in self.population:
            if offspring1_fitness > weakest_fitness1:
                self.population[weakest_index1] = offspring1
        else:
            print("Offspring 1 is a duplicate and discarded.")

        if offspring2 not in self.population:
            if offspring2_fitness > weakest_fitness2:
                self.population[weakest_index2] = offspring2
        else:
            print("Offspring 2 is a duplicate and discarded.")

    
    def perform_tournament_selection(self):
        # Tournament selection parameters
        tournament_size = 8
          # Adjust the tournament size as needed

        # Randomly select individuals for the tournament
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_candidates = [self.population[i] for i in tournament_indices]

        # Evaluate fitness for each candidate
        fitness_scores = [candidate.fitness for candidate in tournament_candidates]

        # Select the two individuals with the highest fitness scores
        winner_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:1]
        parent1_individual = tournament_candidates[winner_indices[0]]

        #Print information about selected parents (for debugging purposes)
        # print("Selected Parents for Reproduction:")
        # print(f"Parent 1 (Individual {tournament_indices[winner_indices[0]] + 1}): Fitness = {fitness_scores[winner_indices[0]]}")
        # parent1_individual.print_tree()
        # print(f"Parent 2 (Individual {tournament_indices[winner_indices[1]] + 1}): Fitness = {fitness_scores[winner_indices[1]]}")
        # parent2_individual.print_tree()

        return (parent1_individual, fitness_scores[winner_indices[0]])

    def make_prediction(self, node, sample):
        if node.value is not None:
            return node.value
        
        # Recursive case: Traverse the tree based on the feature and threshold
        if sample[node.feature] <= node.threshold:
            return self.make_prediction(node.left, sample)
        else:
            return self.make_prediction(node.right, sample)
        
    def evaluate_fitness(self, predictions, actual_labels):
        # Evaluate fitness based on predictions and actual labels
        
        # Fitness Case : the fitness case is the predicted outcome of the individual 
        # in comparison with the actual outcome in the the dataset based on various 
        # diabetic attributes for each entry in the dataset.

        # the fitness function is be accuracy

        correct_predictions = 0
        for pred, actual in zip(predictions, actual_labels['diabetes']):
            if pred == actual:
                correct_predictions += 1

        #correct_predictions = sum(predictions == actual_labels)

        total_samples = len(predictions)
        accuracy = correct_predictions / total_samples

        return accuracy
    
    def get_top_individuals(self, n=10):
        """Return the top 'n' individuals based on fitness scores."""
        sorted_population = sorted(enumerate(self.population, start=1), key=lambda x: x[1].fitness, reverse=True)
        return [(index, individual.fitness) for index, individual in sorted_population[:n]]

    def get_bottom_individuals(self, n=10):
        """Return the bottom 'n' individuals based on fitness scores."""
        sorted_population = sorted(enumerate(self.population, start=1), key=lambda x: x[1].fitness)
        return [(index, individual.fitness) for index, individual in sorted_population[:n]]
    
    def get_fittest_individuals(self, n=1):
        """Return the top 'n' individuals based on fitness scores."""
        sorted_population = sorted(enumerate(self.population, start=1), key=lambda x: x[1].fitness, reverse=True)
        return [(index, individual.fitness) for index, individual in sorted_population[:n]]
    
    def print_population_fitness(self):
        for i, tree in enumerate(self.population):
            print(f"Individual {i + 1} Fitness: {tree.fitness}")


    def print_tree(self, node, indent):
        if node is None:
            return
        print("  " * indent + f"Feature: {node.feature}, Threshold: {node.threshold}, Value: {node.value}")
        self.print_tree(node.left, indent + 1)
        self.print_tree(node.right, indent + 1)

    def calculate_tree_height(self, node):
        if node is None:
            return 0
        else:
            # Recursively calculate the height of the left and right subtrees
            left_height = self.calculate_tree_height(node.left)
            right_height = self.calculate_tree_height(node.right)

            # Return the maximum of the left and right subtree heights, plus 1 for the current node
            return max(left_height, right_height) + 1
        
    def check_unique_population(self):
        """Check if all individuals in the population are unique."""
        unique_individuals = set()

        def compare_trees(tree1, tree2):
            if tree1 is None and tree2 is None:
                return True
            if tree1 is None or tree2 is None:
                return False
            if tree1.feature != tree2.feature or tree1.threshold != tree2.threshold or tree1.value != tree2.value:
                return False
            return compare_trees(tree1.left, tree2.left) and compare_trees(tree1.right, tree2.right)

        for i, individual in enumerate(self.population):
            is_unique = True
            for unique_individual in unique_individuals:
                if compare_trees(individual.root, unique_individual.root):
                    is_unique = False
                    print(f"\033[91mDuplicate found for Individual {i+1}\033[0m")  # Print in red color
                    break
            if is_unique:
                unique_individuals.add(individual)
        print("\033[92mNo Duplicates in the population found.\033[0m")  # Print in green color

