import torch
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
from bookyolo_seg import bookyolo_seg, model_int

# 모델 로드
model_dir = '/opt/ml/model/'
model = None

class S3URL(BaseModel):
    s3_url: str
    s3_bucket: str
    s3_folder: str
    thr: float
    worker: int

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
def invocations(s3_data: S3URL):
    img = input_fn(s3_data.s3_url)
    coodi = predict_fn(img, model, s3_data.s3_bucket, s3_data.s3_folder, s3_data.thr, s3_data.worker)
    coodi = output_fn(coodi)
    return {"prediction": coodi}

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
def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'bookYOLO_seg_model.pt')
    model = model_int(model_path)

    return model

def predict_fn(input_data, model, s3_bucket, s3_folder, thr = 0.5, worker = 4):
    crop_coordinates_list = bookyolo_seg(
            model,
            input_data,
            conf_thres=thr,
            median_mode=False,
            s3_bucket_name=s3_bucket,
            s3_folder=s3_folder,
            worker=worker
    )
    return crop_coordinates_list

def input_fn(s3_url):
    img = download_image_from_s3(s3_url + 'ori.jpg')
    return img

def output_fn(coodi):
    return coodi

if __name__ == "__main__":
    #cpu_count = multiprocessing.cpu_count()
    workers = 1
    uvicorn.run("sagemaker_inference:app", host="0.0.0.0", port=8080, workers=workers)
