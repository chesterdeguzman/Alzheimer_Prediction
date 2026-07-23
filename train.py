from pathlib import Path
import argparse
from alzheimer_analytics.modeling import train


def main():
    parser = argparse.ArgumentParser(description="Train the metadata-only Alzheimer class baseline.")
    parser.add_argument("--input", default="data/raw/Alzheimer_Dataset_Details.csv")
    parser.add_argument("--model", default="models/random_forest.joblib")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--predictions", default="data/processed/test_predictions.csv")
    args = parser.parse_args()
    result = train(args.input, args.model, args.metrics, args.predictions)
    print(f"Test accuracy: {result.metrics['test']['accuracy']:.3f}")
    print(f"Test balanced accuracy: {result.metrics['test']['balanced_accuracy']:.3f}")
    print(f"Test macro F1: {result.metrics['test']['macro_f1']:.3f}")

if __name__ == "__main__":
    main()
