import random
from DecisionNode import DecisionNode
import graphviz
from graphviz import Digraph


class Tree:
    #feature_names =  ["Glucose" , "BloodPressure" , "SkinThickness" , "Insulin" , "BMI" , "DiabetesPedigreeFunction"]

    #feature_names =  ["Glucose" , "Insulin" , "BMI"]

    feature_names =  ["BMI" , "Glucose" , "Insulin", "SkinThickness" , "DiabetesPedigreeFunction" ]


    
    
    def __init__(self, feature_names, max_depth=4):
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.root = None
        self.threshold_ranges = None

    def to_dot(self):
        dot = Digraph()
        self.root.add_to_dot(dot)
        return dot

    
    def visualize_tree(self, node, filename):
        dot = graphviz.Digraph(comment='Decision Tree')
        
        def add_nodes_edges(dot, node):
            if node is None:
                return
            label = f"{node.feature} <= {node.threshold}\nValue: {node.value}" if node.value is not None else f"{node.feature} <= {node.threshold}"
            dot.node(str(id(node)), label=label)
            if node.left:
                dot.edge(str(id(node)), str(id(node.left)), label="True")
                add_nodes_edges(dot, node.left)
            if node.right:
                dot.edge(str(id(node)), str(id(node.right)), label="False")
                add_nodes_edges(dot, node.right)

        add_nodes_edges(dot, node)
        dot.render(filename, format='png', cleanup=True)

    def evaluate(self, sample):
        return self.root.evaluate_tree(sample)

    def generate_random_individual(self, max_depth=4, method='grow', population=20):
        if method not in ['grow', 'full']:
            raise ValueError("Invalid method. Choose from 'grow', 'full'.")

        if self.feature_names is None:
            raise ValueError("Feature names must be provided.")

        num_features = len(self.feature_names)

        def create_grow_individual(depth):
            if depth >= max_depth or (depth >= 3 and random.random() < 0.4):
                if  random.random() > 0.5:
                    value = 1
                else:
                    value = 0
                
                return DecisionNode(value=value)

            else:
                feature = random.choice(self.feature_names)

                min_val, max_val = 0.0, 0.0

                if feature == 'Pregnancies':
                    min_val, max_val = 1, 10
                elif feature == 'Glucose':
                    min_val, max_val = 65, 183
                elif feature == 'Insulin':
                    min_val, max_val = 45, 271
                elif feature == 'BMI':
                    min_val, max_val = 20, 44
                elif feature == 'Age':
                    min_val, max_val = 22, 52
                elif feature ==  'SkinThickness':
                    min_val, max_val = 19, 42
                elif feature == 'DiabetesPedigreeFunction':
                    min_val, max_val = 0.12, 1.05

                threshold = random.uniform(min_val, max_val)

                next_depth = depth + random.randint(1, 2)

                left = create_grow_individual(next_depth)
                right = create_grow_individual(next_depth)
                return DecisionNode(feature=feature, threshold=threshold, left=left, right=right)

        def create_full_individual(depth):
            if depth >= max_depth:
                if  random.random() > 0.5:
                    value = 1
                else:
                    value = 0
                
                return DecisionNode(value=value)
            
                # return DecisionNode(value=random.choice([0, 1]))
            else:
                feature = random.choice(self.feature_names)

                min_val, max_val = 0.0, 0.0

                if feature == 'Pregnancies':
                    min_val, max_val = 1, 10
                elif feature == 'Glucose':
                    min_val, max_val = 65, 183
                elif feature == 'Insulin':
                    min_val, max_val = 45, 271
                elif feature == 'BMI':
                    min_val, max_val = 20, 44
                elif feature == 'Age':
                    min_val, max_val = 22, 52
                elif feature ==  'SkinThickness':
                    min_val, max_val = 19, 42
                elif feature == 'DiabetesPedigreeFunction':
                    min_val, max_val = 0.12, 1.05

                threshold = random.uniform(min_val, max_val)

                left = create_full_individual(depth + 1)
                right = create_full_individual(depth + 1)
                return DecisionNode(feature=feature, threshold=threshold, left=left, right=right)

        def create_individual(depth):
            if method == 'grow':
                #print("method: " + 'grow') 
                return create_grow_individual(depth)
            elif method == 'full':
                #print("method: " + 'full') 
                return create_full_individual(depth)

        self.root = create_individual(0)

    def perform_crossover(self, parent1, parent2):
        # Make copies of parent trees
        parent1_copy = parent1.root.copy_node()
        parent2_copy = parent2.root.copy_node()

        # Randomly select crossover points in each tree
        crossover_point1 = random.choice(self.get_all_non_leaf_nodes_bft(parent1_copy))
        crossover_point2 = random.choice(self.get_all_non_leaf_nodes_bft(parent2_copy))

        # print("Crossover Point 1:")
        # print(f"Feature: {crossover_point1.feature}, Threshold: {crossover_point1.threshold}, Value: {crossover_point1.value}")

        # print("\nCrossover Point 2:")
        # print(f"Feature: {crossover_point2.feature}, Threshold: {crossover_point2.threshold}, Value: {crossover_point2.value}")

        # Swap subtrees rooted at crossover points
        temp_subtree = crossover_point1.copy_node()
        crossover_point1.feature = crossover_point2.feature
        crossover_point1.threshold = crossover_point2.threshold
        crossover_point1.value = crossover_point2.value
        crossover_point1.left = crossover_point2.left
        crossover_point1.right = crossover_point2.right

        crossover_point2.feature = temp_subtree.feature
        crossover_point2.threshold = temp_subtree.threshold
        crossover_point2.value = temp_subtree.value
        crossover_point2.left = temp_subtree.left
        crossover_point2.right = temp_subtree.right

        # Create offspring trees
        offspring1 = Tree(self.feature_names)
        offspring1.root = parent1_copy
        offspring2 = Tree(self.feature_names)
        offspring2.root = parent2_copy
    
        # self.visualize_tree(parent1.root, "parent1_tree")
        # self.visualize_tree(parent2.root, "parent2_tree")
        # self.visualize_tree(offspring1.root, "offspring1_tree")
        # self.visualize_tree(offspring2.root, "offspring2_tree")

        # print("Offstring 1: ")
        # offspring1.print_tree()
        # print("Offstring 2: ")
        # offspring2.print_tree()

        # Perform depth-based pruning on offspring trees
        self.prune_tree_max_height(offspring1.root,5)
        self.prune_tree_max_height(offspring2.root,5)

        # self.visualize_tree(offspring1.root, "offspring1_tree_pruned")
        # self.visualize_tree(offspring2.root, "offspring2_tree_pruned")

        # Return the pruned offspring trees
        return offspring1, offspring2
    
    def prune_tree_max_height(self, node, max_height):
        if node is None:
            return

        # Check if the current node is a leaf node or if the max height constraint is reached
        if node.left is None and node.right is None or max_height <= 1:
            return

        # Recursively prune the subtrees
        self.prune_tree_max_height(node.left, max_height - 1)
        self.prune_tree_max_height(node.right, max_height - 1)

        # Check if the height of the pruned subtrees exceeds the maximum height constraint
        if self.get_tree_height(node.left) > max_height - 1:
            # Prune the left subtree
            node.left = None
            node.left = DecisionNode(value=self.get_majority_class(node))
            # Set the majority class of the pruned node as its value
            #node.value = self.get_majority_class(node)
        if self.get_tree_height(node.right) > max_height - 1:
            # Prune the right subtree
            node.right = None
            node.right = DecisionNode(value=self.get_majority_class(node))

            # Set the majority class of the pruned node as its value
            #node.value = self.get_majority_class(node)

    def get_majority_class(self, node):
        if random.random() > 0.5:  
            return 1
        else:
            return 0

    #add mutation
    def perform_mutation(self, mutation_rate):
        """Perform mutation by randomly selecting a node and mutating its attributes."""
        if self.root is None:
            return  
                
        if random.random() > mutation_rate:
            return 


        all_nodes = self.get_all_nodes_bft(self.root)

        if not all_nodes:
            return  # No mutation if there are no nodes

        # Randomly select a node for mutation
        node_to_mutate = random.choice(all_nodes)

        # print("Selected node for mutation:")
        # print(f"Feature: {node_to_mutate.feature}, Threshold: {node_to_mutate.threshold}, Value: {node_to_mutate.value}")

        if node_to_mutate.is_leaf():
            #print(f"Before mutation, Value: {node_to_mutate.value}")
            # Toggle the value of the leaf node
            if node_to_mutate.value == 1:
                node_to_mutate.value = 0
            else:
                node_to_mutate.value = 1
            #print(f"After mutation, Value: {node_to_mutate.value}")
        else:
            
            random_feature = self.generate_random_feature()
            node_to_mutate.feature = random_feature
            node_to_mutate.threshold = self.generate_random_threshold(random_feature)
            #print(f"Mutated feature: {random_feature}, Mutated threshold: {node_to_mutate.threshold}")

        return node_to_mutate

    def generate_random_feature(self):
        return random.choice(self.feature_names)

    def generate_random_threshold(self, feature):
        min_val, max_val = 0.0, 0.0

        if feature == 'Pregnancies':
            min_val, max_val = 1, 10
        elif feature == 'Glucose':
            min_val, max_val = 65, 183
        elif feature == 'Insulin':
            min_val, max_val = 45, 271
        elif feature == 'BMI':
            min_val, max_val = 20, 44
        elif feature == 'Age':
            min_val, max_val = 22, 52
        elif feature ==  'SkinThickness':
            min_val, max_val = 19, 42
        elif feature == 'DiabetesPedigreeFunction':
            min_val, max_val = 0.12, 1.05

        threshold = random.uniform(min_val, max_val)

        return threshold
    
    def get_all_non_leaf_nodes_bft(self, root):
        """Return a list of all non-leaf nodes in the tree.BFT excluding root"""
        non_leaf_nodes = []

        if root is None:
            return non_leaf_nodes
        
        current_level = [root]

        while current_level:
            next_level = []

            for node in current_level:
                if (node.left or node.right) and node is not root:  
                    non_leaf_nodes.append(node)

                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            
            current_level = next_level

        random.shuffle(non_leaf_nodes)

        return non_leaf_nodes
    
    def get_all_nodes_bft(self, root):
        """Return a list of all nodes in the tree.BFT, excluding root"""
        all_nodes = []

        if root is None:
            return all_nodes
        
        current_level = [root]

        while current_level:
            next_level = []

            for node in current_level:
                if node is not root:  # Skip adding the root node
                    all_nodes.append(node)

                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            
            current_level = next_level

        random.shuffle(all_nodes)

        return all_nodes



        
            
    # def get_all_nodes_bft(self, node):
    #     """Return a list of all nodes in the tree.BFT"""
    #     all_nodes = []

    #     if node is None:
    #         return all_nodes
        
    #     current_level = [node]

    #     while current_level:
    #         next_level = []

    #         for node in current_level:
    #             all_nodes.append(node)

    #             if node.left:
    #                 next_level.append(node.left)
    #             if node.right:
    #                 next_level.append(node.right)
            
    #         current_level = next_level

    #     random.shuffle(all_nodes)

    #     return all_nodes

    # def get_all_non_leaf_nodes_bft(self, root):
    #     """Return a list of all non-leaf nodes in the tree.BFT"""
    #     non_leaf_nodes = []

    #     if root is None:
    #         return non_leaf_nodes
        
    #     current_level = [root]

    #     while current_level:
    #         next_level = []

    #         for node in current_level:
    #             if (node.left or node.right) and node is not root:  
    #                 non_leaf_nodes.append(node)

    #             if node.left:
    #                 next_level.append(node.left)
    #             if node.right:
    #                 next_level.append(node.right)
            
    #         current_level = next_level

    #     random.shuffle(non_leaf_nodes)

    #     return non_leaf_nodes


    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root

        if node.is_leaf():
            print(indent + "Leaf: Predicted Value =", node.value)
        else:
            print(indent + "Split on Feature", node.feature, "at Threshold", node.threshold)
            print(indent + "--> Left:")
            self.print_tree(node.left, indent + "    ")
            print(indent + "--> Right:")
            self.print_tree(node.right, indent + "    ")

    def get_tree_height(self, node):
        if node is None:
            return 0
        else:
            left_height = self.get_tree_height(node.left)
            right_height = self.get_tree_height(node.right)
            return max(left_height, right_height) + 1


    