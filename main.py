import io
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from torchvision.utils import make_grid
from PIL import Image

app = FastAPI(title="GAN MNIST API")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 7 * 7 * 128)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        return self.model(x)

generator = Generator()
generator.load_state_dict(
    torch.load("generator.pth", map_location="cpu", weights_only=True)
)
generator.eval()

@app.get("/")
def home():
    return {"message": "GAN MNIST API is running"}

@app.get("/generate")
def generate():
    with torch.no_grad():
        noise = torch.randn(16, 100)
        images = generator(noise)
        images = (images + 1) / 2
        grid = make_grid(images, nrow=4)

    image = Image.fromarray(
        (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
