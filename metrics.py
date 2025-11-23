import numpy as np

def dcg_at_k(r, k):
    """Score is discounted cumulative gain (DCG) at rank k"""
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    """Score is normalized discounted cumulative gain (NDCG) at rank k"""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def evaluate_model(model, train_mat, val_mat, k=20):
    """
    Evaluates the model using NDCG@K.
    
    Args:
        model: The trained model (must have a predict or recommend method).
        train_mat: Training interactions (to mask seen items).
        val_mat: Validation interactions (ground truth).
        k: Top-K recommendations.
    """
    ndcg_scores = []
    
    # Get users in validation set
    val_users = np.unique(val_mat.nonzero()[0])
    
    print(f"Evaluating on {len(val_users)} users...")
    
    # Batch processing could be faster, but let's start simple loop
    # Depending on the model, we might get scores for all items or just top K
    
    # If model supports batch prediction, use it.
    # Assuming model.predict(user_ids) returns [n_users, n_items] scores
    
    # For ItemKNN, we usually compute scores = User_Vector * Similarity_Matrix
    
    for u_idx in val_users:
        # Ground truth items
        true_items = val_mat[u_idx].indices
        
        # Get scores for all items
        # This part depends on the model interface. 
        # Let's assume the model has a `predict_user` method that returns scores for all items.
        scores = model.predict_user(u_idx)
        
        # Mask items already in training set
        seen_items = train_mat[u_idx].indices
        scores[seen_items] = -np.inf
        
        # Get top K items
        # argpartition is faster than argsort for top K
        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
        
        # Create relevance array (1 if item in true_items, else 0)
        relevance = np.in1d(top_k_indices, true_items).astype(int)
        
        ndcg_scores.append(ndcg_at_k(relevance, k))
        
        if len(ndcg_scores) % 1000 == 0:
            print(f"Processed {len(ndcg_scores)} users...")
            
    return np.mean(ndcg_scores)
