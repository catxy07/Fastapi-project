import joblib
import pandas as pd
from app.core.config import settings
from app.cache.redis_cache import get_cached_prediction, set_cached_prediction

# Load trained pipeline + column info
model_dict = joblib.load(settings.MODEL_PATH)
model = model_dict['pipeline']
expected_num_cols = model_dict['num_cols']
expected_cat_cols = model_dict['cat_cols']

def predict_car_price(data: dict) -> dict:
    # Ensure all expected numeric columns exist
    for col in expected_num_cols:
        if col not in data or data[col] is None:
            data[col] = 0.0

    # Ensure all expected categorical columns exist
    for col in expected_cat_cols:
        if col not in data or data[col] is None:
            data[col] = "missing"

    input_df = pd.DataFrame([data])

    # Check cache
    cache_key = str(data)
    cached_result = get_cached_prediction(cache_key)
    if cached_result:
        return {"predicted_price": float(cached_result)}

    # Predict
    prediction = model.predict(input_df)
    prediction_value = float(prediction[0])

    # Cache result
    set_cached_prediction(cache_key, prediction_value)

    return {"predicted_price": prediction_value}
