import sys, os
import pickle
import pandas as pd
from feature_engineering import process_data


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "submission.csv")

def main():
    train_df = pd.read_csv(DATA_PATH)
    test_df = pd.read_csv(TEST_PATH)
    passenger_ids = test_df["PassengerId"]
    
    _, test_eng = process_data(train_df, test_df)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    preds = model.predict(test_eng)
    
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds.astype(int) 
    })
    
    submission.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()