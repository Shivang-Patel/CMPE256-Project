import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import implicit

class ItemKNN:
    def __init__(self, k=100, shrink=0):
        self.k = k
        self.shrink = shrink
        self.sim_matrix = None
        self.train_mat = None
        
    def fit(self, train_mat):
        self.train_mat = train_mat
        print("Calculating similarity matrix...")
        item_mat = normalize(train_mat.tocsc(), axis=0)
        
        # Compute full similarity: I^T * I
        # Note: This might be memory intensive. If it fails, we need a batched approach.
        sim = item_mat.T.dot(item_mat)
        
        # Zero out diagonal
        sim.setdiag(0)
        
        # Prune to top K
        print(f"Pruning to top {self.k} neighbors...")
        self.sim_matrix = self._prune(sim, self.k)
        
    def _prune(self, sim_matrix, k):
        # sim_matrix is a sparse matrix (likely CSR or CSC)
        # We want to keep only top K values per row.
        
        # Convert to CSR for efficient row slicing
        sim_matrix = sim_matrix.tocsr()
        
        # We can iterate over rows, but that's slow in Python.
        # A faster way is to use data/indices arrays directly if possible, 
        # but rows have variable length.
        
        # Let's try a list comprehension approach which is reasonably fast for 40k rows
        
        new_data = []
        new_indices = []
        new_indptr = [0]
        
        for i in range(sim_matrix.shape[0]):
            # Get row
            row_start = sim_matrix.indptr[i]
            row_end = sim_matrix.indptr[i+1]
            
            if row_end - row_start <= k:
                # Keep all
                data = sim_matrix.data[row_start:row_end]
                indices = sim_matrix.indices[row_start:row_end]
            else:
                # Find top K
                row_data = sim_matrix.data[row_start:row_end]
                row_indices = sim_matrix.indices[row_start:row_end]
                
                # argpartition to find top K indices in row_data
                top_k_idx = np.argpartition(row_data, -k)[-k:]
                
                data = row_data[top_k_idx]
                indices = row_indices[top_k_idx]
                
            new_data.extend(data)
            new_indices.extend(indices)
            new_indptr.append(len(new_data))
            
        return sp.csr_matrix((new_data, new_indices, new_indptr), shape=sim_matrix.shape)
        
    def predict_user(self, u_idx):
        user_vec = self.train_mat[u_idx]
        scores = user_vec.dot(self.sim_matrix)
        if sp.issparse(scores):
            scores = scores.toarray().ravel()
        else:
            scores = np.array(scores).ravel()
        return scores

class EASE:
    def __init__(self, lambda_=500):
        self.lambda_ = lambda_
        self.B = None
        self.train_mat = None
        
    def fit(self, train_mat):
        self.train_mat = train_mat
        print("Calculating G = X^T X...")
        G = train_mat.T.dot(train_mat).toarray()
        print("Adding lambda to diagonal...")
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.lambda_
        print("Inverting G...")
        P = np.linalg.inv(G)
        print("Calculating B...")
        B = P / (-np.diag(P))
        B[diag_indices] = 0
        self.B = B
        
    def predict_user(self, u_idx):
        user_vec = self.train_mat[u_idx].toarray().ravel()
        scores = user_vec.dot(self.B)
        return scores

class ALS:
    def __init__(self, factors=50, regularization=0.01, iterations=15):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, 
            regularization=regularization, 
            iterations=iterations,
            random_state=42
        )
        self.train_mat = None
        
    def fit(self, train_mat):
        self.train_mat = train_mat
        self.model.fit(train_mat)
        
    def predict_user(self, u_idx):
        user_factors = self.model.user_factors[u_idx]
        item_factors = self.model.item_factors
        scores = user_factors.dot(item_factors.T)
        return scores
