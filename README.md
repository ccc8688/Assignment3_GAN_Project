# Assignment 3 — GAN and CNN FastAPI

This project trains a GAN on MNIST handwritten digits and integrates the trained generator with an existing CNN classifier in a FastAPI application.

## Project Files

- `Assignment3_GAN.ipynb`: GAN architecture and MNIST training
- `generator.pth`: trained GAN generator weights
- `cnn_cifar10.pth`: trained CNN weights
- `main.py`: GAN API implementation
- `combined_api.py`: combined FastAPI application
- `sps_genai/`: CNN implementation
- `Dockerfile.combined`: Docker configuration
- `requirements.txt`: Python dependencies

## Run with Docker

Build the Docker image:

```sh
docker build -f Dockerfile.combined -t assignment3-combined .

```

Start the Docker container:

```bash
docker run --rm -p 8002:8000 assignment3-combined
```

## Test the API

Combined API documentation: http://localhost:8002/docs

GAN image generation: http://localhost:8002/gan/generate

CNN API documentation: http://localhost:8002/cnn/docs

CNN prediction: POST http://localhost:8002/cnn/predict

Upload an image through the CNN Swagger documentation to test classification.

## GitHub Repository

https://github.com/ccc8688/Assignment3_GAN_Project
