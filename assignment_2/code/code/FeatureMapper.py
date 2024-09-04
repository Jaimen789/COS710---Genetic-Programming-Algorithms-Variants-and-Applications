import random

class FeatureMapper:
    def __init__(self, source_features, target_features):
        self.source_features = source_features
        self.target_features = target_features
        
        # Dictionary to map source features to target features and threshold ranges
        self.feature_mapping = {
            "Glucose": {"feature": "AGE", "threshold_range": (20, 70)},
            "Insulin": {"feature": "HbA1c", "threshold_range": (1, 13)},
            "SkinThickness": {"feature": "Chol", "threshold_range": (1, 10)},
            "DiabetesPedigreeFunction": {"feature": "TG", "threshold_range": (0.5, 8)}
        }
    
    def map_and_replace_features_s1_t(self, node):
        # Perform DFS traversal to update features and threshold values in the tree
        stack = [node]
        while stack:
            current_node = stack.pop()
            # Perform feature mapping and replacement
            if current_node.feature in self.source_features:
                if current_node.feature != "BMI":  # Check if the feature is not "BMI"
                    mapping_info = self.feature_mapping.get(current_node.feature)
                    if mapping_info:
                        new_feature = mapping_info["feature"]
                        new_threshold = random.uniform(*mapping_info["threshold_range"])
                        current_node.feature = new_feature
                        current_node.threshold = new_threshold
                    else:
                        # Handle case where feature is not present in target dataset
                        pass
            # Add child nodes to the stack for further processing
            if current_node.left:
                stack.append(current_node.left)
            if current_node.right:
                stack.append(current_node.right)
        
        return node

