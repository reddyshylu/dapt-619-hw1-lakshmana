import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def main():
    model_path = './models/linear_regression_pipeline.joblib'
    data_path = './data/raw/geyser.tsv'
    scored_output_path = './data/scored/geyser_scored.csv'
    plot_output_path = './plots/geyser_predictions.png'
    
    print("Loading model...")
    model = joblib.load(model_path)
    
    print("Loading data...")
    data = pd.read_csv(data_path, sep="\t")
    
    print("Scoring data...")
    predictions = model.predict(data[['eruptions']])
    
    scored_df = data.copy()
    scored_df['predicted_waiting'] = predictions
    
    os.makedirs(os.path.dirname(scored_output_path), exist_ok=True)
    scored_df.to_csv(scored_output_path, index=False)
    print(f"Scored dataset exported to: {scored_output_path}")
    
    print("Creating plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(data['waiting'], predictions, alpha=0.6, edgecolors='k')
    
    min_val = min(data['waiting'].min(), predictions.min())
    max_val = max(data['waiting'].max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Waiting Time (minutes)', fontsize=12)
    plt.ylabel('Predicted Waiting Time (minutes)', fontsize=12)
    plt.title('Actual vs Predicted Waiting Times for Old Faithful Geyser', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_output_path}")
    plt.close()
    
    print("\nScoring complete!")
    print(f"Total records scored: {len(data)}")

if __name__ == "__main__":
    main()
