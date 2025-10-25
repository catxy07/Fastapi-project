# TRAINING SCRIPT
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
from training.train_utils import MODEL_DIR, MODEL_PATH
from training.train_utils import DATA_FILE_PATH
# Your pipeline creation here...
# rf_model = Pipeline([...])

# Save pipeline
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_model, os.path.join(MODEL_DIR, "model.joblib"))

# Save column info separately
cols_info = {"num_cols": num_cols, "cat_cols": cat_cols}
joblib.dump(cols_info, os.path.join(MODEL_DIR, "cols_info.joblib"))

print("Pipeline saved to model.joblib")
print("Column info saved to cols_info.joblib")
