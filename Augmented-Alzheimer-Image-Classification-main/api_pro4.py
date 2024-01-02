from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from io import BytesIO
from starlette.responses import JSONResponse

app = FastAPI()

# Load the saved model
loaded_model = load_model(r"C:\Users\fady\Downloads\Machine Project\model1.h5", compile=False)

# Custom decoding for your case (4 classes)
class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def predict_image(img_array):
    # Preprocess the input image
    img_array = preprocess_input(img_array)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to the domain of your webpage or "*" for any
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        img = image.load_img(BytesIO(contents), target_size=(176, 176))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predicted_class = predict_image(img_array)

        return JSONResponse(content={"result": predicted_class}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
