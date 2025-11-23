from recsys_utils import DataLoader
from models import ItemKNN
import numpy as np

def main():
    print("Loading data...")
    loader = DataLoader("train.txt")
    loader.load_data()
    
    # Use full matrix for training
    train_mat = loader.get_csr_matrix()
    
    print(f"Training ItemKNN on full data (Users: {loader.num_users}, Items: {loader.num_items})...")
    # Use best K from tuning (will update if needed, default 100)
    model = ItemKNN(k=500) 
    model.fit(train_mat)
    
    print("Generating recommendations...")
    
    with open("submission.txt", "w") as f:
        # Iterate over all users in order of their internal ID to keep it simple, 
        # but we need to write "UserID Item1 ..."
        # The order in output file doesn't strictly matter usually, but good to be consistent.
        
        for u_idx in range(loader.num_users):
            u_original = loader.reverse_user_map[u_idx]
            
            # Get scores
            scores = model.predict_user(u_idx)
            
            # Mask items already seen
            seen_items = train_mat[u_idx].indices
            scores[seen_items] = -np.inf
            
            # Get top 20
            top_k_indices = np.argpartition(scores, -20)[-20:]
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
            
            # Convert to original item IDs
            top_items = [loader.reverse_item_map[i] for i in top_k_indices]
            
            # Write to file
            f.write(f"{u_original} {' '.join(top_items)}\n")
            
            if u_idx % 1000 == 0:
                print(f"Processed {u_idx} users...")
                
    print("Submission file generated: submission.txt")

if __name__ == "__main__":
    main()
