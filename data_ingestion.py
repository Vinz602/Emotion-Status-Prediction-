from pathlib import Path
import pandas as pd

# Base directory = iris/
BASE_DIR = Path(__file__).parent

# Define folders
RAW_DIR = BASE_DIR #/folder
INGESTED_DIR = BASE_DIR /"ingested"

# Define files
INPUT_FILE_TRAIN = RAW_DIR / "train.csv"
INPUT_FILE_TEST = RAW_DIR / "test.csv"
INPUT_FILE_VAL = RAW_DIR / "val.csv"
OUTPUT_FILE_TRAIN = INGESTED_DIR / "ingested_data_train.csv"
OUTPUT_FILE_TEST = INGESTED_DIR / "ingested_data_test.csv"
OUTPUT_FILE_VAL = INGESTED_DIR / "ingested_data_val.csv"

def ingest_data():
    # Ensure output folder exists
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # Read raw data
    df_train = pd.read_csv(INPUT_FILE_TRAIN)
    df_test = pd.read_csv(INPUT_FILE_TEST)
    df_val = pd.read_csv(INPUT_FILE_VAL)

    # Basic validation
    assert not df_train.empty, "Training dataset is empty"
    assert not df_test.empty, "Test dataset is empty"
    assert not df_val.empty, "Validation dataset is empty"
    

    # Save ingested data
    df_train.to_csv(OUTPUT_FILE_TRAIN, index=False)
    df_test.to_csv(OUTPUT_FILE_TEST, index=False)
    df_val.to_csv(OUTPUT_FILE_VAL, index=False)

    print(f"✅ Data ingested from {INPUT_FILE_TRAIN} → {OUTPUT_FILE_TRAIN}")
    print(f"✅ Data ingested from {INPUT_FILE_TEST} → {OUTPUT_FILE_TEST}")
    print(f"✅ Data ingested from {INPUT_FILE_VAL} → {OUTPUT_FILE_VAL}")

if __name__ == "__main__":
    ingest_data()
