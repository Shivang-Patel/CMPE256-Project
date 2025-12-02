"""
CMPE 256 - Recommendation System Project
Implementation of multiple recommendation algorithms with hyperparameter tuning
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from typing import List, Dict, Tuple, Set
import time

class DataLoader:
    """Load and preprocess the user-item interaction data"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.user_items = {}
        self.item_users = defaultdict(set)
        self.all_items = set()
        self.n_users = 0
        self.n_items = 0

    def load_data(self):
        """Load data from file"""
        print("Loading data...")
        with open(self.filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                self.user_items[user_id] = items
                self.all_items.update(items)
                for item in items:
                    self.item_users[item].add(user_id)

        self.n_users = len(self.user_items)
        self.n_items = len(self.all_items)
        print(f"Loaded {self.n_users} users and {self.n_items} unique items")
        return self

    def get_statistics(self):
        """Get dataset statistics"""
        interactions = sum(len(items) for items in self.user_items.values())
        avg_items_per_user = interactions / self.n_users
        sparsity = 1 - (interactions / (self.n_users * self.n_items))

        items_per_user = [len(items) for items in self.user_items.values()]
        users_per_item = [len(users) for users in self.item_users.values()]

        stats = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_interactions': interactions,
            'avg_items_per_user': avg_items_per_user,
            'sparsity': sparsity,
            'min_items_per_user': min(items_per_user),
            'max_items_per_user': max(items_per_user),
            'median_items_per_user': np.median(items_per_user),
            'min_users_per_item': min(users_per_item),
            'max_users_per_item': max(users_per_item),
            'median_users_per_item': np.median(users_per_item)
        }
        return stats

    def train_test_split(self, test_size=2):
        """
        Split data into train and test sets
        Hold out last 'test_size' items per user for testing
        """
        train_data = {}
        test_data = {}

        for user_id, items in self.user_items.items():
            if len(items) <= test_size:
                # If user has too few items, keep all in training
                train_data[user_id] = items
                test_data[user_id] = []
            else:
                train_data[user_id] = items[:-test_size]
                test_data[user_id] = items[-test_size:]

        return train_data, test_data


class EvaluationMetrics:
    """Evaluation metrics for recommendation systems"""

    @staticmethod
    def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / k

    @staticmethod
    def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate Recall@K"""
        if len(relevant) == 0:
            return 0.0
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate NDCG@K"""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)

        # DCG
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # +2 because index starts at 0

        # IDCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate MAP@K"""
        if len(relevant) == 0:
            return 0.0

        recommended_k = recommended[:k]
        relevant_set = set(relevant)

        score = 0.0
        num_hits = 0.0

        for i, item in enumerate(recommended_k):
            if item in relevant_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(relevant), k)

    @staticmethod
    def evaluate_recommendations(recommendations: Dict[int, List[int]],
                                 test_data: Dict[int, List[int]],
                                 k_values: List[int] = [10, 20]) -> Dict:
        """Evaluate all metrics for given recommendations"""
        results = {f'precision@{k}': [] for k in k_values}
        results.update({f'recall@{k}': [] for k in k_values})
        results.update({f'ndcg@{k}': [] for k in k_values})
        results.update({f'map@{k}': [] for k in k_values})

        for user_id, relevant_items in test_data.items():
            if len(relevant_items) == 0 or user_id not in recommendations:
                continue

            recommended = recommendations[user_id]

            for k in k_values:
                results[f'precision@{k}'].append(
                    EvaluationMetrics.precision_at_k(recommended, relevant_items, k))
                results[f'recall@{k}'].append(
                    EvaluationMetrics.recall_at_k(recommended, relevant_items, k))
                results[f'ndcg@{k}'].append(
                    EvaluationMetrics.ndcg_at_k(recommended, relevant_items, k))
                results[f'map@{k}'].append(
                    EvaluationMetrics.map_at_k(recommended, relevant_items, k))

        # Average all metrics
        avg_results = {metric: np.mean(values) for metric, values in results.items()}
        return avg_results


class PopularityBaseline:
    """Popularity-based recommendation baseline"""

    def __init__(self, variant='global'):
        """
        variant: 'global' for simple popularity, 'weighted' for position-weighted
        """
        self.variant = variant
        self.item_popularity = None

    def fit(self, train_data: Dict[int, List[int]]):
        """Calculate item popularity from training data"""
        if self.variant == 'global':
            # Simple frequency count
            item_counts = Counter()
            for items in train_data.values():
                item_counts.update(items)
            self.item_popularity = item_counts

        elif self.variant == 'weighted':
            # Weight items by their position (recent items get higher weight)
            item_scores = defaultdict(float)
            for items in train_data.values():
                for idx, item in enumerate(items):
                    # Higher weight for items later in the sequence
                    weight = (idx + 1) / len(items)
                    item_scores[item] += weight
            self.item_popularity = item_scores

        return self

    def recommend(self, user_id: int, train_data: Dict[int, List[int]],
                  all_items: Set[int], n: int = 20) -> List[int]:
        """Generate top-N recommendations"""
        # Get items user has already interacted with
        user_items = set(train_data.get(user_id, []))

        # Sort items by popularity, excluding user's items
        candidates = [(item, score) for item, score in self.item_popularity.items()
                      if item not in user_items]
        candidates.sort(key=lambda x: x[1], reverse=True)

        recommendations = [item for item, score in candidates[:n]]

        # If not enough recommendations, fill with random popular items
        if len(recommendations) < n:
            remaining_items = list(all_items - user_items - set(recommendations))
            np.random.shuffle(remaining_items)
            recommendations.extend(remaining_items[:n - len(recommendations)])

        return recommendations[:n]


class ItemBasedCF:
    """Item-based Collaborative Filtering"""

    def __init__(self, similarity='cosine', k_similar=50):
        """
        similarity: 'cosine' or 'adjusted_cosine'
        k_similar: number of similar items to consider
        """
        self.similarity = similarity
        self.k_similar = k_similar
        self.item_similarity = None
        self.user_item_matrix = None

    def fit(self, train_data: Dict[int, List[int]], all_items: Set[int]):
        """Build item similarity matrix"""
        print(f"Fitting Item-based CF ({self.similarity})...")

        # Create user-item matrix
        n_users = max(train_data.keys()) + 1
        n_items = max(all_items) + 1

        self.user_item_matrix = lil_matrix((n_users, n_items), dtype=np.float32)
        for user_id, items in train_data.items():
            for item in items:
                self.user_item_matrix[user_id, item] = 1.0

        self.user_item_matrix = self.user_item_matrix.tocsr()

        # Calculate item-item similarity
        if self.similarity == 'cosine':
            # Simple cosine similarity
            item_matrix = self.user_item_matrix.T.tocsr()
            self.item_similarity = cosine_similarity(item_matrix, dense_output=False)

        elif self.similarity == 'adjusted_cosine':
            # Adjusted cosine: normalize by user mean
            user_means = np.array(self.user_item_matrix.mean(axis=1)).flatten()
            adjusted_matrix = self.user_item_matrix.copy().tolil()

            for user_id in range(n_users):
                user_items = adjusted_matrix[user_id].nonzero()[1]
                if len(user_items) > 0:
                    adjusted_matrix[user_id, user_items] -= user_means[user_id]

            adjusted_matrix = adjusted_matrix.tocsr()
            item_matrix = adjusted_matrix.T.tocsr()
            self.item_similarity = cosine_similarity(item_matrix, dense_output=False)

        return self

    def recommend(self, user_id: int, train_data: Dict[int, List[int]],
                  all_items: Set[int], n: int = 20) -> List[int]:
        """Generate top-N recommendations"""
        user_items = set(train_data.get(user_id, []))

        # Calculate scores for all items
        item_scores = defaultdict(float)

        for user_item in user_items:
            if user_item >= self.item_similarity.shape[0]:
                continue

            # Get similar items
            similar_items = self.item_similarity[user_item].toarray().flatten()
            top_similar_indices = np.argsort(similar_items)[::-1][:self.k_similar + 1]

            for similar_item in top_similar_indices:
                if similar_item != user_item and similar_item not in user_items:
                    item_scores[similar_item] += similar_items[similar_item]

        # Sort by score
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item for item, score in recommendations[:n]]

        # Fill with popular items if needed
        if len(recommendations) < n:
            remaining_items = list(all_items - user_items - set(recommendations))
            np.random.shuffle(remaining_items)
            recommendations.extend(remaining_items[:n - len(recommendations)])

        return recommendations[:n]


class MatrixFactorizationALS:
    """Matrix Factorization using Alternating Least Squares"""

    def __init__(self, n_factors=50, n_iterations=10, reg_param=0.01):
        """
        n_factors: number of latent factors
        n_iterations: number of ALS iterations
        reg_param: regularization parameter
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg_param = reg_param
        self.user_factors = None
        self.item_factors = None

    def fit(self, train_data: Dict[int, List[int]], all_items: Set[int]):
        """Train matrix factorization model using ALS"""
        print(f"Fitting ALS (factors={self.n_factors}, reg={self.reg_param})...")

        # Create user-item matrix
        n_users = max(train_data.keys()) + 1
        n_items = max(all_items) + 1

        R = lil_matrix((n_users, n_items), dtype=np.float32)
        for user_id, items in train_data.items():
            for item in items:
                R[user_id, item] = 1.0

        R = R.tocsr()

        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # ALS iterations
        for iteration in range(self.n_iterations):
            # Update user factors
            for user_id in range(n_users):
                user_items = R[user_id].nonzero()[1]
                if len(user_items) == 0:
                    continue

                A = self.item_factors[user_items].T @ self.item_factors[user_items] + \
                    self.reg_param * np.eye(self.n_factors)
                b = self.item_factors[user_items].T @ np.ones(len(user_items))

                self.user_factors[user_id] = np.linalg.solve(A, b)

            # Update item factors
            R_t = R.T.tocsr()
            for item_id in range(n_items):
                item_users = R_t[item_id].nonzero()[1]
                if len(item_users) == 0:
                    continue

                A = self.user_factors[item_users].T @ self.user_factors[item_users] + \
                    self.reg_param * np.eye(self.n_factors)
                b = self.user_factors[item_users].T @ np.ones(len(item_users))

                self.item_factors[item_id] = np.linalg.solve(A, b)

            if (iteration + 1) % 3 == 0:
                print(f"  Iteration {iteration + 1}/{self.n_iterations} completed")

        return self

    def recommend(self, user_id: int, train_data: Dict[int, List[int]],
                  all_items: Set[int], n: int = 20) -> List[int]:
        """Generate top-N recommendations"""
        if user_id >= len(self.user_factors):
            # Cold start: return popular items
            popular_items = list(all_items)
            np.random.shuffle(popular_items)
            return popular_items[:n]

        user_items = set(train_data.get(user_id, []))

        # Calculate scores for all items
        scores = self.user_factors[user_id] @ self.item_factors.T

        # Get top items excluding already seen
        item_scores = [(item, scores[item]) for item in all_items
                       if item not in user_items and item < len(scores)]
        item_scores.sort(key=lambda x: x[1], reverse=True)

        recommendations = [item for item, score in item_scores[:n]]

        # Fill with random items if needed
        if len(recommendations) < n:
            remaining_items = list(all_items - user_items - set(recommendations))
            np.random.shuffle(remaining_items)
            recommendations.extend(remaining_items[:n - len(recommendations)])

        return recommendations[:n]


class UserBasedCF:
    """User-based Collaborative Filtering"""

    def __init__(self, k_neighbors=20, similarity='cosine'):
        """
        k_neighbors: number of similar users to consider
        similarity: 'cosine' or 'jaccard'
        """
        self.k_neighbors = k_neighbors
        self.similarity = similarity
        self.user_similarity = None
        self.user_item_matrix = None

    def fit(self, train_data: Dict[int, List[int]], all_items: Set[int]):
        """Build user similarity matrix"""
        print(f"Fitting User-based CF (k={self.k_neighbors}, sim={self.similarity})...")

        # Create user-item matrix
        n_users = max(train_data.keys()) + 1
        n_items = max(all_items) + 1

        self.user_item_matrix = lil_matrix((n_users, n_items), dtype=np.float32)
        for user_id, items in train_data.items():
            for item in items:
                self.user_item_matrix[user_id, item] = 1.0

        self.user_item_matrix = self.user_item_matrix.tocsr()

        # Calculate user-user similarity
        if self.similarity == 'cosine':
            self.user_similarity = cosine_similarity(self.user_item_matrix, dense_output=False)

        elif self.similarity == 'jaccard':
            # Jaccard similarity for binary data
            self.user_similarity = lil_matrix((n_users, n_users), dtype=np.float32)

            for i in range(min(n_users, 1000)):  # Limit for efficiency
                i_items = set(self.user_item_matrix[i].nonzero()[1])
                if len(i_items) == 0:
                    continue

                for j in range(i + 1, n_users):
                    j_items = set(self.user_item_matrix[j].nonzero()[1])
                    if len(j_items) == 0:
                        continue

                    intersection = len(i_items & j_items)
                    union = len(i_items | j_items)

                    if union > 0:
                        sim = intersection / union
                        self.user_similarity[i, j] = sim
                        self.user_similarity[j, i] = sim

                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1} users...")

            self.user_similarity = self.user_similarity.tocsr()

        return self

    def recommend(self, user_id: int, train_data: Dict[int, List[int]],
                  all_items: Set[int], n: int = 20) -> List[int]:
        """Generate top-N recommendations"""
        if user_id >= self.user_similarity.shape[0]:
            # Cold start
            popular_items = list(all_items)
            np.random.shuffle(popular_items)
            return popular_items[:n]

        user_items = set(train_data.get(user_id, []))

        # Get similar users
        user_sims = self.user_similarity[user_id].toarray().flatten()
        similar_users = np.argsort(user_sims)[::-1][1:self.k_neighbors + 1]  # Exclude self

        # Aggregate items from similar users
        item_scores = defaultdict(float)
        for similar_user in similar_users:
            sim_score = user_sims[similar_user]
            if sim_score <= 0:
                continue

            similar_user_items = set(train_data.get(similar_user, []))
            for item in similar_user_items:
                if item not in user_items:
                    item_scores[item] += sim_score

        # Sort by score
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item for item, score in recommendations[:n]]

        # Fill with random items if needed
        if len(recommendations) < n:
            remaining_items = list(all_items - user_items - set(recommendations))
            np.random.shuffle(remaining_items)
            recommendations.extend(remaining_items[:n - len(recommendations)])

        return recommendations[:n]


def main():
    """Main execution function"""
    print("="*80)
    print("CMPE 256 - Recommendation System Project")
    print("="*80)

    # Load data
    data_loader = DataLoader('train.txt')
    data_loader.load_data()

    # Get statistics
    stats = data_loader.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Number of users: {stats['n_users']}")
    print(f"  Number of items: {stats['n_items']}")
    print(f"  Number of interactions: {stats['n_interactions']}")
    print(f"  Sparsity: {stats['sparsity']:.4f}")
    print(f"  Avg items per user: {stats['avg_items_per_user']:.2f}")
    print(f"  Items per user - Min: {stats['min_items_per_user']}, Max: {stats['max_items_per_user']}, Median: {stats['median_items_per_user']:.0f}")
    print(f"  Users per item - Min: {stats['min_users_per_item']}, Max: {stats['max_users_per_item']}, Median: {stats['median_users_per_item']:.0f}")

    # Split data
    print("\nSplitting data into train and test sets...")
    train_data, test_data = data_loader.train_test_split(test_size=2)
    print(f"  Training users: {len(train_data)}")
    print(f"  Test users with held-out items: {sum(1 for items in test_data.values() if len(items) > 0)}")

    # Save statistics
    with open('dataset_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("\nSaved dataset statistics to 'dataset_statistics.json'")

    # Initialize results dictionary
    all_results = {}

    # Algorithm 1: Popularity-based baselines
    print("\n" + "="*80)
    print("Algorithm 1: Popularity-based Baselines")
    print("="*80)

    for variant in ['global', 'weighted']:
        print(f"\nVariant: {variant}")
        model = PopularityBaseline(variant=variant)
        model.fit(train_data)

        # Generate recommendations
        recommendations = {}
        for user_id in train_data.keys():
            recommendations[user_id] = model.recommend(user_id, train_data,
                                                       data_loader.all_items, n=20)

        # Evaluate
        results = EvaluationMetrics.evaluate_recommendations(recommendations, test_data)
        all_results[f'popularity_{variant}'] = results

        print("Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

    # Save intermediate results
    with open('results_popularity.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("Completed initial analysis and baseline models")
    print("="*80)


if __name__ == "__main__":
    main()
