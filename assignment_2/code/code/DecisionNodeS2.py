class DecisionNodeS2:
    
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None, level = 1):
        self.feature = feature       # Index of the feature to split on
        self.threshold = threshold   # Threshold value for the feature
        self.value = value           # Value (class label) if the node is a leaf
        self.left = left             # Left child node
        self.right = right           # Right child node
        self.level = level

    #deep copy
    def copy_node(self):
        # Create a new node with the same attributes
        new_node = DecisionNodeS2(self.feature, self.threshold, self.value)

        # If it's a leaf node, copy the leaf label
        if self.value is not None:
            new_node.value = self.value

        # Recursively copy left and right children
        if self.left:
            new_node.left = self.left.copy_node()
        if self.right:
            new_node.right = self.right.copy_node()

        return new_node

    def evaluate_tree(self, sample):
        if self.value is not None:
            return self.value
        else:
            if sample[self.feature] <= self.threshold:
                return self.left.evaluate_tree(sample)
            else:
                return self.right.evaluate_tree(sample)

    def equals(self, other):
        """
        Check if two nodes have the same data and children.
        """
        if self.is_leaf() and other.is_leaf():
            return self.value == other.value
        elif not self.is_leaf() and not other.is_leaf():
            return self.feature == other.feature and self.threshold == other.threshold
        else:
            return False
    
    #is leaf
    def is_leaf(self):
        return self.left is None and self.right is None
    
    #change value
    def change_value(self, v):
        self.value = v

    #add left_child
    def add_left(self, l):
        self.left = l

    #add right_child
    def add_right(self, r):
        self.right = r

    def add_to_dot(self, dot):
        if self.value is not None:
            dot.node(str(id(self)), label=str(self.value))
        else:
            dot.node(str(id(self)), label=f"{self.feature} <= {self.threshold}")
            if self.left is not None:
                self.left.add_to_dot(dot)
                dot.edge(str(id(self)), str(id(self.left)), label="True")
            if self.right is not None:
                self.right.add_to_dot(dot)
                dot.edge(str(id(self)), str(id(self.right)), label="False")
    
    

    
        




