import torch
import iresnet
import os
import numpy as np
import cv2
from io import BytesIO
import boto3
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import multiprocessing

# 모델 로드
model_dir = '/opt/ml/model/'
model = None

class S3URL(BaseModel):
    s3_url: str

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic here
    global model
    model = model_fn(model_dir)
    yield
    # Shutdown logic here (if needed)

app = FastAPI(lifespan=lifespan)

@app.get('/ping')
def ping():
    health = model is not None
    if not health:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}

@app.post('/invocations')
def invocations(s3_url: S3URL):
    img = input_fn(s3_url.s3_url)
    feat = predict_fn(img, model)
    feat = output_fn(feat)
    return {"prediction": feat}

# S3에서 이미지 다운로드 함수
def download_image_from_s3(s3_url):
    s3 = boto3.client('s3')
    bucket_name = s3_url.split('/')[2]
    key = '/'.join(s3_url.split('/')[3:])

    response = s3.get_object(Bucket=bucket_name, Key=key)
    img_data = response['Body'].read()
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img

# Preprocessing function to prepare the input image
def preprocesing(img):
    height, width = img.shape[:2]
    crop_height, crop_width = int(height * 0.01), int(width * 0.01)
    img = img[crop_height:height - crop_height, crop_width:width - crop_width]
    img1 = cv2.resize(img, (224, 224))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = np.transpose(img1, (2, 0, 1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = torch.from_numpy(img1).unsqueeze(0).float().to(device)
    img1.div_(255).sub_(0.5).div_(0.5)
    return img1

# Postprocessing to calculate the distance between two features
def postprocesing(feat1,feat2):
    diff = np.subtract(feat1, feat2)
    dist = np.sqrt(np.sum(np.square(diff), 1))
    return dist

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'arcbook_model.pt')
    model = iresnet.iresnet18(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
  
    model.eval()

    return model

def predict_fn(input_data, model):
    with torch.no_grad():
        prediction = model(input_data)
    return prediction

def input_fn(s3_url):
    img = download_image_from_s3(s3_url)
    preprocessed_image = preprocesing(img)
    return preprocessed_image

def output_fn(prediction):
    return prediction.cpu().numpy().tolist()

if __name__ == "__main__":
    #cpu_count = multiprocessing.cpu_count()
    workers = 4
    uvicorn.run("sagemaker_inference:app", host="0.0.0.0", port=8080, workers=workers)
