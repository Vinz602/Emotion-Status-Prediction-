import joblib
from sklearn.preprocessing import LabelEncoder
from data_ingestion import ingest_data
from training import train_model
from evaluation import evaluate
import pandas as pd

ACCURACY_THRESHOLD = 0.9

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()
    
    df_train = pd.read_csv("ingested/ingested_data_train.csv")
    df_test = pd.read_csv("ingested/ingested_data_test.csv")
    df_test2 = pd.read_csv("ingested/ingested_data_val.csv")
    df_test = pd.concat([df_test, df_test2], ignore_index=True)
    
    column_mapping = {
    "Daily_Usage_Time (minutes)": "Daily_Usage_Time_minutes",
    "Posts_Per_Day": "Posts_Per_Day",
    "Likes_Received_Per_Day": "Likes_Received_Per_Day",
    "Comments_Received_Per_Day": "Comments_Received_Per_Day",
    "Messages_Sent_Per_Day": "Messages_Sent_Per_Day"
    }
    
    df_train.rename(columns=column_mapping, inplace=True)
    df_test.rename(columns=column_mapping, inplace=True)
    
    mask = df_train['Age'] == 'Male'
    df_train.loc[mask, ['Age', 'Gender']] = df_train.loc[mask, ['Gender', 'Age']].values

    mask2 = df_train['Age'] == 'Female'
    df_train.loc[mask2, ['Age', 'Gender']] = df_train.loc[mask2, ['Gender', 'Age']].values

    mask3 = df_train['Age'] == 'Non-binary'
    df_train.loc[mask3, ['Age', 'Gender']] = df_train.loc[mask3, ['Gender', 'Age']].values
    
    mask = df_test['Age'] == 'Male'
    df_test.loc[mask, ['Age', 'Gender']] = df_test.loc[mask, ['Gender', 'Age']].values

    mask2 = df_test['Age'] == 'Female'
    df_test.loc[mask2, ['Age', 'Gender']] = df_test.loc[mask2, ['Gender', 'Age']].values

    mask3 = df_test['Age'] == 'Non-binary'
    df_test.loc[mask3, ['Age', 'Gender']] = df_test.loc[mask3, ['Gender', 'Age']].values
    
    df_train = df_train[df_train['Gender'].isin(['Male', 'Female', 'Non-binary'])]
    df_train['Gender'].value_counts()
    
    df_test = df_test[df_test['Gender'].isin(['Male', 'Female', 'Non-binary'])]
    df_test['Gender'].value_counts()
    
    df_train_cleaned = df_train[pd.to_numeric(df_train['Age'], errors='coerce').notnull()]

    df_test_cleaned = df_test[pd.to_numeric(df_test['Age'], errors='coerce').notnull()]
    
    #convert string to integer
    df_train_cleaned['Age']=df_train_cleaned['Age'].astype(str).astype(int)
    df_test_cleaned['Age']=df_test_cleaned['Age'].astype(str).astype(int)

    x_train = df_train_cleaned.drop('Dominant_Emotion', axis=1)
    y_train = df_train_cleaned['Dominant_Emotion']
    
    x_test = df_test_cleaned.drop('Dominant_Emotion', axis=1)
    y_test = df_test_cleaned['Dominant_Emotion']
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    joblib.dump(le, "artifacts/label_encoder.pkl")

    print("Step 2: Training")
    run_id = train_model(x_train, y_train_encoded)

    print("Step 3: Evaluation")
    accuracy,prec,recall = evaluate(x_test,y_test_encoded,run_id)

    if accuracy >= ACCURACY_THRESHOLD:
        print("Model approved for deployment")
    else:
        print("Model rejected")

if __name__ == "__main__":
    run_pipeline()