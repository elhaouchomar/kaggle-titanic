import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from feature_engineering import process_data


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "assets")

def main():
    train_df = pd.read_csv(DATA_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    train_eng, _ = process_data(train_df, test_df)
    
    X = train_eng.drop(columns=["Survived"])
    y = train_eng["Survived"]
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,       
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    
    acc = accuracy_score(y, y_pred_cv)
    prec = precision_score(y, y_pred_cv)
    rec = recall_score(y, y_pred_cv)
    f1 = f1_score(y, y_pred_cv)
    cm = confusion_matrix(y, y_pred_cv)
    
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 50)
    
    os.makedirs(ASSETS_PATH, exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Perished (0)', 'Survived (1)'],
                yticklabels=['Perished (0)', 'Survived (1)'])
    plt.title('Random Forest Confusion Matrix (5-Fold CV)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_plot_path = os.path.join(ASSETS_PATH, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()