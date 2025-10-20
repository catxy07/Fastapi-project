import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from training.train_utils import DATA_FILE_PATH, MODEL_PATH, MODEL_DIR

df = (
    pd.read_csv(DATA_FILE_PATH)
    .drop_duplicates()
    .drop(columns=['name', 'model', 'edition'])
)

X = df.drop(columns=['selling_price'])
y = df['selling_price'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = X_train.select_dtypes(include='object').columns.tolist()

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

regressor = RandomForestRegressor(
    n_estimators=10, max_depth=5, random_state=42
    )

rf_model = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('regressor', regressor)  
])

rf_model.fit(X_train, y_train)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_model, MODEL_PATH)



