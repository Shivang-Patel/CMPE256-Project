import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filename):
        self.filename = filename
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.num_users = 0
        self.num_items = 0
        
    def load_data(self):
        """Loads data from text file and creates mappings."""
        users = []
        items = []
        
        with open(self.filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                u_raw = parts[0]
                if u_raw not in self.user_map:
                    self.user_map[u_raw] = self.num_users
                    self.reverse_user_map[self.num_users] = u_raw
                    self.num_users += 1
                u_idx = self.user_map[u_raw]
                
                for i_raw in parts[1:]:
                    if i_raw not in self.item_map:
                        self.item_map[i_raw] = self.num_items
                        self.reverse_item_map[self.num_items] = i_raw
                        self.num_items += 1
                    i_idx = self.item_map[i_raw]
                    
                    users.append(u_idx)
                    items.append(i_idx)
                    
        self.interactions = np.array([users, items]).T
        print(f"Loaded {len(self.interactions)} interactions.")
        print(f"Users: {self.num_users}, Items: {self.num_items}")
        
    def get_csr_matrix(self):
        """Returns the full user-item interaction matrix."""
        rows = self.interactions[:, 0]
        cols = self.interactions[:, 1]
        data = np.ones(len(self.interactions))
        return sp.csr_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items))
    
    def train_test_split(self, test_ratio=0.2, seed=42):
        """
        Splits interactions into train and test sets.
        Note: This is a global split. For a more robust evaluation, 
        we might want a per-user split, but global is faster for a first pass.
        However, for NDCG per user, we need to ensure users in test are in train.
        
        Better approach for Recommender Systems:
        For each user, mask some items for testing.
        """
        np.random.seed(seed)
        
        train_rows = []
        train_cols = []
        val_rows = []
        val_cols = []
        
        # Group items by user
        user_interactions = {}
        for u, i in self.interactions:
            if u not in user_interactions:
                user_interactions[u] = []
            user_interactions[u].append(i)
            
        for u, items in user_interactions.items():
            # If user has very few interactions, keep all in train to avoid cold start issues in validation
            if len(items) < 5:
                for i in items:
                    train_rows.append(u)
                    train_cols.append(i)
                continue
                
            # Randomly select 20% for validation
            n_val = int(len(items) * test_ratio)
            val_items = np.random.choice(items, n_val, replace=False)
            val_set = set(val_items)
            
            for i in items:
                if i in val_set:
                    val_rows.append(u)
                    val_cols.append(i)
                else:
                    train_rows.append(u)
                    train_cols.append(i)
                    
        train_data = np.ones(len(train_rows))
        val_data = np.ones(len(val_rows))
        
        train_mat = sp.csr_matrix((train_data, (train_rows, train_cols)), shape=(self.num_users, self.num_items))
        val_mat = sp.csr_matrix((val_data, (val_rows, val_cols)), shape=(self.num_users, self.num_items))
        
        return train_mat, val_mat
