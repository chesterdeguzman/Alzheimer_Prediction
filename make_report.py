from alzheimer_analytics.data import load_dataset, validate_dataset
from alzheimer_analytics.eda import create_figures

if __name__ == "__main__":
    df = load_dataset("data/raw/Alzheimer_Dataset_Details.csv")
    validate_dataset(df)
    create_figures(df, "reports/figures")
    print("EDA figures saved to reports/figures")
