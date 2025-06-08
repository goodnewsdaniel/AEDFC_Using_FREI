
"""
# Implementation Plan

1. Core AEDFC Algorithm with FREI stopping criterion
2. Feature Reduction Entropy Index (FREI) implementation
3. Adversarial Robustness Testing framework with multiple baselines
4. Experimental Setup for DDoS detection using CIC-DDoS2019 dataset

"""


import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import CarliniL2Method
from tqdm import tqdm


class AEDFC:
    def __init__(self, redundancy_threshold=0.1, min_clusters=5, max_iter=100):
        """
        Initialize AEDFC with FREI criterion

        Parameters:
        - redundancy_threshold: FREI threshold for stopping criterion
        - min_clusters: minimum number of clusters to allow
        - max_iter: maximum iterations to prevent infinite loops
        """
        self.redundancy_threshold = redundancy_threshold
        self.min_clusters = min_clusters
        self.max_iter = max_iter
        self.cluster_history = []
        self.frei_history = []

    def _compute_conditional_entropy_matrix(self, X):
        """Compute pairwise conditional entropy matrix for all features"""
        n_features = X.shape[1]
        cond_entropy_matrix = np.zeros((n_features, n_features))

        # Discretize continuous features for entropy calculation
        X_discrete = self._discretize_features(X)

        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    # Calculate H(Xi|Xj)
                    joint_probs = self._joint_probability(
                        X_discrete[:, i], X_discrete[:, j])
                    cond_entropy = 0
                    for xj_val in np.unique(X_discrete[:, j]):
                        mask = X_discrete[:, j] == xj_val
                        if np.sum(mask) > 0:
                            xi_cond = X_discrete[mask, i]
                            cond_entropy += np.sum(mask)/len(X) * \
                                entropy(np.bincount(
                                    xi_cond)/np.sum(mask))
                    cond_entropy_matrix[i, j] = cond_entropy

        # Make matrix symmetric by averaging
        sym_matrix = (cond_entropy_matrix + cond_entropy_matrix.T) / 2
        np.fill_diagonal(sym_matrix, 0)  # Zero diagonal
        return sym_matrix

    def _discretize_features(self, X, bins=10):
        """Discretize continuous features into bins for entropy calculation"""
        X_discrete = np.zeros_like(X, dtype=int)
        for i in range(X.shape[1]):
            if len(np.unique(X[:, i])) > bins:
                X_discrete[:, i] = pd.cut(X[:, i], bins=bins, labels=False)
            else:
                X_discrete[:, i] = X[:, i]
        return X_discrete

    def _joint_probability(self, x, y):
        """Compute joint probability distribution of two discrete variables"""
        xy = np.vstack([x, y]).T
        unique_rows, counts = np.unique(xy, axis=0, return_counts=True)
        joint_probs = counts / len(x)
        return joint_probs

    def _compute_frei(self, clusters, cond_entropy_matrix):
        """
        Compute Feature Reduction Entropy Index (FREI)

        FREI measures the average redundancy within clusters normalized by
        the entropy between clusters. Lower FREI indicates better redundancy reduction.
        """
        intra_cluster_redundancy = 0
        inter_cluster_redundancy = 0

        # Calculate intra-cluster redundancy (average conditional entropy within clusters)
        for cluster in clusters:
            if len(cluster) > 1:
                cluster_indices = list(cluster)
                sub_matrix = cond_entropy_matrix[np.ix_(
                    cluster_indices, cluster_indices)]
                intra_cluster_redundancy += np.sum(sub_matrix) / \
                    (len(cluster)**2 - len(cluster))

        if len(clusters) > 1:
            intra_cluster_redundancy /= len(clusters)

        # Calculate inter-cluster redundancy (average conditional entropy between clusters)
        all_cluster_indices = [list(cluster) for cluster in clusters]
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                sub_matrix = cond_entropy_matrix[np.ix_(
                    all_cluster_indices[i], all_cluster_indices[j])]
                inter_cluster_redundancy += np.mean(sub_matrix)

        if len(clusters) > 1:
            inter_cluster_redundancy /= (len(clusters)*(len(clusters)-1)/2)

        # FREI formula: ratio of intra-cluster to inter-cluster redundancy
        if inter_cluster_redundancy > 0:
            frei = intra_cluster_redundancy / inter_cluster_redundancy
        else:
            frei = intra_cluster_redundancy / 1e-10  # prevent division by zero

        return frei

    def fit(self, X):
        """Perform adaptive entropy-driven feature clustering"""
        # Compute conditional entropy matrix
        self.cond_entropy_matrix = self._compute_conditional_entropy_matrix(X)
        n_features = X.shape[1]

        # Initialize each feature as its own cluster
        clusters = [frozenset([i]) for i in range(n_features)]
        self.cluster_history.append(clusters.copy())

        # Compute initial FREI
        current_frei = self._compute_frei(clusters, self.cond_entropy_matrix)
        self.frei_history.append(current_frei)

        iteration = 0
        while (len(clusters) > self.min_clusters and
               iteration < self.max_iter and
               current_frei > self.redundancy_threshold):

            # Find the two most similar clusters to merge
            min_dissimilarity = float('inf')
            best_pair = (0, 1)

            # Compute dissimilarity between all cluster pairs
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    # Average dissimilarity between clusters
                    cluster_i = list(clusters[i])
                    cluster_j = list(clusters[j])
                    sub_matrix = self.cond_entropy_matrix[np.ix_(
                        cluster_i, cluster_j)]
                    avg_dissimilarity = np.mean(sub_matrix)

                    if avg_dissimilarity < min_dissimilarity:
                        min_dissimilarity = avg_dissimilarity
                        best_pair = (i, j)

            # Merge the two most similar clusters
            merged_cluster = clusters[best_pair[0]].union(
                clusters[best_pair[1]])
            new_clusters = [clusters[k]
                            for k in range(len(clusters)) if k not in best_pair]
            new_clusters.append(merged_cluster)
            clusters = new_clusters

            # Update FREI and history
            current_frei = self._compute_frei(
                clusters, self.cond_entropy_matrix)
            self.cluster_history.append(clusters.copy())
            self.frei_history.append(current_frei)

            iteration += 1

        self.final_clusters = clusters
        return self

    def transform(self, X, strategy='representative'):
        """
        Reduce features based on the final clusters

        Parameters:
        - X: input data (n_samples, n_features)
        - strategy: how to select features from clusters
                   'representative': choose feature with lowest average conditional entropy
                   'all': keep all features (just for analysis)
        """
        if strategy == 'all':
            return X

        selected_features = []
        feature_names = []

        for cluster in self.final_clusters:
            cluster_indices = list(cluster)

            if strategy == 'representative':
                # Select feature with minimum average conditional entropy to others in cluster
                if len(cluster_indices) == 1:
                    selected_idx = cluster_indices[0]
                else:
                    sub_matrix = self.cond_entropy_matrix[np.ix_(
                        cluster_indices, cluster_indices)]
                    avg_entropy = np.mean(sub_matrix, axis=1)
                    selected_idx = cluster_indices[np.argmin(avg_entropy)]

                selected_features.append(selected_idx)
                feature_names.append(f"Feature_{selected_idx}")

        return X[:, selected_features], feature_names

    def get_feature_clusters(self):
        """Return the final feature clusters"""
        return self.final_clusters


class DDoSDetector:
    def __init__(self, random_state=42):
        """Initialize DDoS detection framework"""
        self.random_state = random_state
        self.classifier = RandomForestClassifier(random_state=random_state)

    def load_data(self, dataset_path):
        """Load and preprocess CIC-DDoS2019 dataset"""
        # In practice, you would load the actual dataset here
        # This is a simplified version for demonstration
        data = pd.read_csv(dataset_path)

        # Preprocessing steps would include:
        # - Handling missing values
        # - Encoding categorical features
        # - Normalizing/standardizing numerical features
        # - Balancing classes if needed

        X = data.drop(columns=['Label']).values
        y = data['Label'].values

        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def evaluate_feature_selection(self, X_train, X_test, feature_selector):
        """Evaluate feature selection method"""
        # Apply feature selection
        feature_selector.fit(X_train)
        X_train_reduced, feature_names = feature_selector.transform(X_train)
        X_test_reduced, _ = feature_selector.transform(X_test)

        # Train classifier
        self.classifier.fit(X_train_reduced, self.y_train)

        # Calculate accuracy
        train_acc = self.classifier.score(X_train_reduced, self.y_train)
        test_acc = self.classifier.score(X_test_reduced, self.y_test)

        # Calculate redundancy score (average pairwise conditional entropy)
        cond_entropy_matrix = feature_selector.cond_entropy_matrix
        selected_indices = [int(name.split('_')[1]) for name in feature_names]
        sub_matrix = cond_entropy_matrix[np.ix_(
            selected_indices, selected_indices)]
        redundancy_score = np.mean(sub_matrix)

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'redundancy_score': redundancy_score,
            'num_features': len(feature_names),
            'feature_names': feature_names
        }

    def adversarial_robustness_test(self, X_train, X_test, feature_selector, attack_strength=0.1):
        """Evaluate robustness against adversarial attacks"""
        # Apply feature selection
        feature_selector.fit(X_train)
        X_train_reduced, _ = feature_selector.transform(X_train)
        X_test_reduced, _ = feature_selector.transform(X_test)

        # Train classifier
        self.classifier.fit(X_train_reduced, self.y_train)

        # Create ART classifier
        art_classifier = SklearnClassifier(model=self.classifier)

        # Generate adversarial examples
        attack = CarliniL2Method(classifier=art_classifier, confidence=0.0,
                                 targeted=False, learning_rate=0.01,
                                 max_iter=100, initial_const=0.01)
        X_test_adv = attack.generate(X_test_reduced)

        # Evaluate robustness
        clean_acc = self.classifier.score(X_test_reduced, self.y_test)
        adv_acc = self.classifier.score(X_test_adv, self.y_test)
        robustness = adv_acc / clean_acc

        return {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'robustness': robustness
        }


def compare_baselines(detector, X_train, X_test):
    """Compare AEDFC against other feature selection baselines"""
    results = {}

    # Define feature selection methods to compare
    methods = {
        'AEDFC': AEDFC(redundancy_threshold=0.15, min_clusters=10),
        'PearsonCorrelation': PearsonCorrelationSelector(threshold=0.9),
        'KMeansClustering': KMeansFeatureClusterer(n_clusters=10),
        'HierarchicalClustering': HierarchicalFeatureClusterer(n_clusters=10)
    }

    # Evaluate each method
    for name, method in methods.items():
        print(f"\nEvaluating {name}...")

        # Standard evaluation
        eval_results = detector.evaluate_feature_selection(
            X_train, X_test, method)

        # Adversarial robustness
        robustness_results = detector.adversarial_robustness_test(
            X_train, X_test, method)

        # Combine results
        results[name] = {**eval_results, **robustness_results}

    return results


class PearsonCorrelationSelector:
    """Baseline: Feature selection based on Pearson correlation"""

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X):
        corr_matrix = np.corrcoef(X.T)
        self.corr_matrix = corr_matrix
        return self

    def transform(self, X):
        n_features = X.shape[1]
        selected_features = []

        for i in range(n_features):
            keep = True
            for j in selected_features:
                if abs(self.corr_matrix[i, j]) > self.threshold:
                    keep = False
                    break
            if keep:
                selected_features.append(i)

        return X[:, selected_features], [f"Feature_{i}" for i in selected_features]


class KMeansFeatureClusterer:
    """Baseline: Feature clustering using K-means"""

    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters

    def fit(self, X):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(X.T)  # Cluster features, not samples
        self.labels_ = kmeans.labels_
        return self

    def transform(self, X):
        selected_features = []
        for cluster_id in np.unique(self.labels_):
            cluster_indices = np.where(self.labels_ == cluster_id)[0]
            # Select first feature in cluster as representative
            selected_features.append(cluster_indices[0])

        return X[:, selected_features], [f"Feature_{i}" for i in selected_features]


class HierarchicalFeatureClusterer:
    """Baseline: Feature clustering using hierarchical clustering"""

    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters

    def fit(self, X):
        # Use correlation-based distance
        corr_matrix = np.corrcoef(X.T)
        distance_matrix = 1 - np.abs(corr_matrix)

        # Perform hierarchical clustering
        hc = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            linkage='complete'
        )
        hc.fit(distance_matrix)
        self.labels_ = hc.labels_
        return self

    def transform(self, X):
        selected_features = []
        for cluster_id in np.unique(self.labels_):
            cluster_indices = np.where(self.labels_ == cluster_id)[0]
            # Select first feature in cluster as representative
            selected_features.append(cluster_indices[0])

        return X[:, selected_features], [f"Feature_{i}" for i in selected_features]


if __name__ == "__main__":
    # Example usage
    print("Implementing AEDFC for DDoS detection with FREI criterion")

    # Initialize detector
    detector = DDoSDetector(random_state=42)

    # Load data (in practice, use real dataset path)
    # dataset_path = "CIC-DDoS2019.csv"
    # For demonstration, we'll create synthetic data
    n_samples = 1000
    n_features = 50
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    X_test = np.random.randn(n_samples//2, n_features)
    y_test = np.random.randint(0, 2, n_samples//2)

    # Store in detector for compatibility
    detector.X_train = X_train
    detector.X_test = X_test
    detector.y_train = y_train
    detector.y_test = y_test

    # Compare methods
    results = compare_baselines(detector, X_train, X_test)

    # Display results
    print("\nComparison Results:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            if metric != 'feature_names':
                print(f"{metric}: {value:.4f}")

    # Example of analyzing AEDFC specifically
    aedfc = AEDFC(redundancy_threshold=0.15, min_clusters=10)
    aedfc.fit(X_train)

    print("\nAEDFC Feature Clusters:")
    for i, cluster in enumerate(aedfc.get_feature_clusters()):
        print(f"Cluster {i+1}: {list(cluster)}")

    print("\nFREI History:", aedfc.frei_history)

"""
# Key Components Implemented:

1. AEDFC Algorithm:
   - Computes pairwise conditional entropy matrix
   - Uses FREI(Feature Reduction Entropy Index) as stopping criterion
   - Implements iterative merging of most redundant features
   - Provides feature transformation with representative selection

2. Feature Reduction Entropy Index(FREI):
   - Measures ratio of intra-cluster to inter-cluster redundancy
   - Lower FREI indicates better redundancy reduction
   - Used as primary stopping criterion for AEDFC

3. Adversarial Robustness Testing:
   - Uses ART library(Adversarial Robustness Toolbox)
   - Implements Carlini & Wagner L2 attacks
   - Compares robustness across multiple baselines

4. Baseline Methods:
   - Pearson Correlation(filter method)
   - K-means Clustering
   - Hierarchical Clustering

5. Experimental Framework:
   - Integrated with Random Forest classifier
   - Measures standard accuracy and robustness metrics
   - Tracks redundancy scores and feature counts

# Notes on Implementation:
''''
1. The implementation uses synthetic data for demonstration purposes. In practice, you would:
   - Load the real CIC-DDoS2019 dataset
   - Perform proper data preprocessing (normalization, handling missing values, etc.)
   - Adjust hyperparameters based on validation performance

2. For the conditional entropy calculations:
   - Continuous features are discretized into bins
   - Joint probabilities are estimated empirically from data
   - The implementation ensures numerical stability

3. The adversarial robustness testing framework:
   - Uses state-of-the-art attack methods
   - Provides quantitative measures of model robustness
   - Can be extended with additional attack types

4. The FREI implementation:
   - Properly normalizes intra-cluster and inter-cluster redundancy
   - Provides a theoretically grounded stopping criterion
   - Tracks FREI values through the clustering process

This implementation satisfies both requested modifications: (1) using FREI as the redundancy criterion, and (2) comparing against multiple baselines for adversarial robustness testing. The code is modular and can be extended with additional feature selection methods or evaluation metrics as needed.

"""
