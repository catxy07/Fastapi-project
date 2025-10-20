import joblib
import pandas as pd
from app.core.config import settings
from app.cache.redis_cache import get_cached_prediction, set_cached_prediction

model = joblib.load(settings.MODEL_PATH)

def predict_car_price(data: dict) -> dict:
    input = pd.DataFrame([data])
    cache_key = str(data)
    
    cached_result = get_cached_prediction(cache_key)
    if cached_result:
        return cached_result
    
    prediction = model.predict(input)[0]
    # result = {"predicted_price": prediction[0]}
    
    set_cached_prediction(cache_key, prediction)
    
    return prediction