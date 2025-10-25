import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor  
import joblib
from training.train_utils import DATA_FILE_PATH, MODEL_PATH, MODEL_DIR

# Load data
df = pd.read_csv(DATA_FILE_PATH).drop_duplicates().drop(columns=['name', 'model', 'edition'])

X = df.drop(columns=['selling_price'])
y = df['selling_price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
num_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = X_train.select_dtypes(include='object').columns.tolist()

# Pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
], remainder='passthrough')

# Full pipeline
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42))
])

# Train
rf_model.fit(X_train, y_train)

# Save **both pipeline + column info** in a dict
model_dict = {
    'pipeline': rf_model,
    'num_cols': num_cols,
    'cat_cols': cat_cols
}

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model_dict, MODEL_PATH)

print(f"Model and columns info saved at: {MODEL_PATH}")
