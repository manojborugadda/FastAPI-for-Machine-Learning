from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import uvicorn
import sklearn


# defining the class for input data validation
class InputData(BaseModel):
    #inputs in the model
    x1: Annotated[float, Field(...,gt=0, description="first input feature for the linear regression model", example=5.0)]
    x2: Annotated[float, Field(...,gt=0, description="second input feature for the linear regression model", example=3.0)]

app = FastAPI()

model = joblib.load("linear_regression_model.pkl")

@app.post("/predict")
async def predict(data: InputData):
    # converting input data into numpy array
    input_data = np.array([[data.x1, data.x2]])
    #making prediction using the loaded model
    prediction = model.predict(input_data)[0]
    # return the prediction 
    return {"prediction": prediction}
   
if __name__ == "__main__":
    uvicorn.run("linear_regression_api:app",host="0.0.0.0",port=8088)   
    
# run the app with below command in the terminal:
# uvicorn linear_regression_api:app --host 0.0.0.0 --port 8088