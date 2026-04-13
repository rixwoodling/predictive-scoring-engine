import pandas as pd
import numpy as np

def main():
    print("Predictive Scoring Engine (ML Pipeline)")
    
    # tiny dummy dataset
    data = pd.DataFrame({
        "income": [50000, 60000, 40000, 80000],
        "credit_score": [600, 650, 580, 720],
        "default": [1, 0, 1, 0]
    })

    print("\nSample Data:")
    print(data)

    print("\nProject is set up correctly.")

if __name__ == "__main__":
    main()
