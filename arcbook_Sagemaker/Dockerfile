# 베이스 이미지 설정
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

# 타임존 및 필수 패키지 설치
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    python3-pip \
    libgl1-mesa-glx \
    curl \
    git \
    wget \
    vim \
    tzdata

# pip 업그레이드
RUN python -m pip install --upgrade pip

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt /opt/ml/code/requirements.txt
RUN python -m pip install -r /opt/ml/code/requirements.txt

# 코드 파일들을 컨테이너에 복사
COPY sagemaker_inference.py /opt/ml/code/sagemaker_inference.py
COPY iresnet.py /opt/ml/code/iresnet.py

ENTRYPOINT ["python", "/opt/ml/code/sagemaker_inference.py"]
