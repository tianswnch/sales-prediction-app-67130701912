import joblib
import pandas as pd
import glob
import os

def load_model_auto(model_path=None):
    """
    Load the model from 'model-reg-xxx.pkl'.
    If model_path is not provided, automatically load the latest model-reg-*.pkl.
    """
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return joblib.load(model_path)
    else:
        files = glob.glob("model-reg-*.pkl")
        if not files:
            raise FileNotFoundError("No model-reg-xxx.pkl file found in current directory.")
        latest = max(files, key=os.path.getmtime)
        print(f"Auto-loaded latest model: {latest}")
        return joblib.load(latest)

def main():
    # Step 1: Load the model
    model = load_model_auto()

    # Step 2: Create new DataFrame
    new_data = pd.DataFrame([[50, 50, 50]], columns=["youtube", "tiktok", "instagram"])
    print("Input data:")
    print(new_data)

    # Step 3: Predict sales
    predicted_sales = model.predict(new_data)
    print(f"Predicted sales: {float(predicted_sales[0]):.4f}")

if __name__ == "__main__":
    main()
