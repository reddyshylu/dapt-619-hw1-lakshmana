import numpy as np
import pandas as pd
import joblib

def test_scoring():
    model_path = './models/linear_regression_pipeline.joblib'
    pipeline = joblib.load(model_path)
    
    df = pd.DataFrame({"eruptions": [1.5, 2.0, 3.0]})
    preds = pipeline.predict(df[["eruptions"]])
    
    assert len(preds) == len(df["eruptions"]), \
        f"Number of predictions ({len(preds)}) does not match number of inputs ({len(df['eruptions'])})"
    
    assert np.all(np.isfinite(preds)), \
        "Some predicted values are not finite (NaN or inf detected)"
    
    assert np.all(preds > 0), \
        "Some predicted values are not positive (zero or negative values detected)"
    
    print(f"All tests passed! Predictions: {preds}")

if __name__ == "__main__":
    test_scoring()
