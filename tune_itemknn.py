from recsys_utils import DataLoader
from models import ItemKNN
from metrics import evaluate_model
import numpy as np

def main():
    print("Loading data...")
    loader = DataLoader("train.txt")
    loader.load_data()
    
    print("Splitting data...")
    train_mat, val_mat = loader.train_test_split(test_ratio=0.2)
    
    k_values = [50, 100, 200, 500]
    
    for k in k_values:
        print(f"Training ItemKNN with k={k}...")
        model = ItemKNN(k=k)
        model.fit(train_mat)
        
        print(f"Evaluating k={k}...")
        ndcg = evaluate_model(model, train_mat, val_mat, k=20)
        print(f"Validation NDCG@20 (k={k}): {ndcg:.4f}")

if __name__ == "__main__":
    main()
