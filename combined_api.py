from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sps_genai.image_api import app as cnn_app
from main import app as gan_app

app = FastAPI(title="CNN and GAN API")

app.mount("/cnn", cnn_app)
app.mount("/gan", gan_app)

@app.get("/")
def home():
    return {
        "message": "Combined CNN and GAN API is running",
        "cnn_endpoint": "/cnn/predict",
        "gan_endpoint": "/gan/generate"
    }
