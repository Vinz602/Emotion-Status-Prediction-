import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


def train_model(x_train, y_train):
    os.makedirs("artifacts", exist_ok=True)
    
    cat_feat = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_feat = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    #for impute missing value numerical features and categorical
    numeric_preprocess = Pipeline([('num_imputer', SimpleImputer(strategy='mean'))])
    categorical_preprocess= Pipeline([('cat_imputer', SimpleImputer(strategy='most_frequent')),
                                ('cat_encoder', OrdinalEncoder(categories=[['Male', 'Female', 'Non-binary'],]))])
    onehot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocess=ColumnTransformer(
        transformers=[
            ('numPreprocess', numeric_preprocess, num_feat),
            ('ordinalPreprocess', categorical_preprocess,(['Gender'])),
            ('oneHotPreprocess', onehot_pipeline, (['Platform'])),
        ],
        remainder='drop'
    )

    emotion_status_pred = Pipeline([
        ('preprocessing', preprocess),
        ('classifier', XGBClassifier(random_state=42,
                        n_estimators=50,
                        max_depth=10,
                        learning_rate=0.06681591658732046,
                        subsample=0.7340310041219665,
                        colsample_bytree= 0.5621164979232791,
                        gamma=0.363146357292497,
                        use_label_encoder=False,))])

    # ===== MLflow tracking =====

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Emotion Status Prediction")

    with mlflow.start_run() as run:
        # log parameters
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("learning_rate", 0.06681591658732046)
        mlflow.log_param("subsample", 0.7340310041219665)
        mlflow.log_param("colsample_bytree", 0.5621164979232791)
        mlflow.log_param("gamma", 0.363146357292497)

        # train
        emotion_status_pred.fit(x_train, y_train)

        # save and log model
        joblib.dump(emotion_status_pred, "artifacts/emotion_status_prediction_pipeline.pkl")
        mlflow.sklearn.log_model(emotion_status_pred,artifact_path="model")

    return run.info.run_id

if __name__ == "__main__":
    train_model()