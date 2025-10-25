from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.core.dependencies import get_current_user, get_api_key
from app.services.model_service import predict_car_price

router = APIRouter()

class CarFeatures(BaseModel):
    company: str
    year: int
    owner: str
    fuel: str
    seller_type: str
    transmission: str
    km_driven: float
    mileage_mpg: float
    engine_cc: float
    torque_nm: float
    seats: float
    # Optional: Add other columns your model expects
    max_power_bhp: float | None = None  # handle missing numeric column
    # Add any other numeric/categorical columns that could be missing
    # Example:
    # some_column: str | None = None

@router.post("/predict")
def predict_price(
    car: CarFeatures,
    user=Depends(get_current_user),
    api_key=Depends(get_api_key)
):
    try:
        # Convert Pydantic model to dict
        car_data = car.model_dump()

        # Predict using model_service
        prediction_dict = predict_car_price(car_data)

        predicted_price = prediction_dict.get("predicted_price", 0.0)
        return {"predicted_price": f"{predicted_price:,.2f}"}

    except Exception as e:
        # Return a clear error
        raise HTTPException(status_code=400, detail=str(e))
