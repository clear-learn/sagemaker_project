# 베이스 이미지 설정
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

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

# 3) Tesseract 5 PPA 추가
RUN add-apt-repository ppa:alex-p/tesseract-ocr5 && apt-get update

# 4) Tesseract 5 + 언어팩 설치
#    (영문 + 한글 + OSD 등 필요한 것)
RUN apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor \
    tesseract-ocr-osd \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*
    
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5.00/tessdata

# 외부 코드 파일
COPY sagemaker_inference.py /opt/ml/code/sagemaker_inference.py
COPY bookyolo_seg.py /opt/ml/code/bookyolo_seg.py

# 내부 코드 파일
COPY bookyolo_seg.py /opt/ml/code/

ENTRYPOINT ["python", "/opt/ml/code/sagemaker_inference.py"]
