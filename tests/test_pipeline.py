import pandas as pd
from alzheimer_analytics.data import validate_dataset, split_dataset, SAFE_FEATURES
from alzheimer_analytics.modeling import build_model


def sample_frame():
    rows = []
    for split in ["train", "val", "test"]:
        for disease, size in [("Mild_Demented", 4.4), ("Non_Demented", 4.1), ("Very_Mild_Demented", 4.3)]:
            rows.append({
                "Split": split, "Disease": disease, "Filename": f"{split}_{disease}.jpg",
                "Extension": ".jpg", "Width": 128, "Height": 128, "Channels": 1,
                "Color Mode": "L", "File Size (KB)": size, "File Size (MB)": size/1024,
                "Full Path": f"/{split}/{disease}.jpg",
            })
    return pd.DataFrame(rows)


def test_validate_and_split():
    df = sample_frame()
    validate_dataset(df)
    train, val, test = split_dataset(df)
    assert len(train) == len(val) == len(test) == 3


def test_model_fits_safe_features():
    df = sample_frame()
    train, _, test = split_dataset(df)
    model = build_model(random_state=1)
    model.fit(train[SAFE_FEATURES], train["Disease"])
    assert len(model.predict(test[SAFE_FEATURES])) == len(test)
