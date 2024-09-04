import random

class FeatureMapper2:
    def __init__(self, source_features, target_features):
        self.source_features = source_features
        self.target_features = target_features
        self.feature_mapping = {
            "bmi": {"feature": "BMI", "threshold_range": None},
            "age": {"feature": "AGE", "threshold_range": None},
            "HbA1c_level" : {"feature": "HbA1c", "threshold_range": None},
            "blood_glucose_level": {"feature": ["Chol", "TG"], "threshold_range": {"Chol": (1, 8), "TG": (0.5, 9)}}
        }

    def map_and_replace_features_s2_t(self, node):
        stack = [node]
        while stack:
            current_node = stack.pop()
            if current_node.feature is not None and current_node.feature.lower() in map(str.lower, self.source_features):
                # print("Current Feature before mapping: ", current_node.feature)
                mapping_info = self.feature_mapping.get(current_node.feature.lower())
                if mapping_info:
                    new_feature = mapping_info["feature"]
                    # print("new_feature: " + new_feature[0])
                    if isinstance(new_feature, str):
                        current_node.feature = new_feature
                    elif isinstance(new_feature, list):
                        current_node.feature = random.choice(new_feature)
                    if mapping_info["threshold_range"]:
                        if isinstance(current_node.feature, str):
                            threshold_range = mapping_info["threshold_range"].get(current_node.feature)
                            if threshold_range:
                                new_threshold = random.uniform(*threshold_range)
                                current_node.threshold = new_threshold
                        else:
                            print(f"Feature '{current_node.feature}' not handled for threshold mapping.")

                elif current_node.feature.lower() == "hba1c_level":
                        # print("mapping Hba1c_level")
                        current_node.feature = "HbA1c"
                else:
                    print("error mapping!")

            if current_node.left:
                stack.append(current_node.left)
            if current_node.right:
                stack.append(current_node.right)
        return node