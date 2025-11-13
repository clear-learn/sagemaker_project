# arcbook_Sagemaker

## 개요
**Sagemaker를 활용해 arcbook API를 생성하는 Dockerfile과 python 포함 패키지**

## Sagemaker 사용법

### 1. Sagemaker deploy code example
```
import sagemaker
from sagemaker.model import Model

sagemaker_session = sagemaker.Session()
model_data = 'S3 버킷에 저장된 모델 아티팩트 (학습이 완료된 모델 파일의 경로)'
image_uri = 'ECR에 저장된 Docker 이미지 경로'
instance = 'ml.g4dn.xlarge'

role = 'SageMaker-MLAI'

model = Model(
    image_uri=image_uri,
    model_data=model_data,
    role=role,
    sagemaker_session=sagemaker_session,
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=instance,
    endpoint_name='arcbook'
)

# s3가 아닌 알라딘 NAS 모델 장소 https://nas.aladin.co.kr/drive/d/f/10t2GtEq9JL7zEXmq5DBtVy6H8qer5XC
# s3 모델 장소 s3://aladin-ai-models/arcbook_model.tar.gz
# ECR docker 장소 예시 381492228102.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker/arcbook:latest
```
### 2. Sagemaker inference code example
```
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

endpoint_name = 'arcbook'

# S3 URL
s3_url = '요청할 이미지의 S3 url'

# 요청 데이터 준비
payload = json.dumps({"s3_url": s3_url})

# 예측 요청
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=payload
)

# 결과 파싱
result_1 = json.loads(response['Body'].read().decode())
print(result_1)

# 요청할 책등 이미지 url 장소 예시 s3://aladin-ai-models/295604650_9.jpg
```

### 3. Sagemaker ECR Docker image upload example
반드시 dockerfile은 해당 폴더를 벗어나서 빌드하지 말것.
```
# 개인계정에따라 바뀔 수 있으므로 AWS ECR 에서 확인할것
# in terminal
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 381492228102.dkr.ecr.ap-northeast-2.amazonaws.com

docker build . -t arcbook

docker tag arcbook:latest 381492228102.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker/arcbook:latest

docker push 381492228102.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker/arcbook:latest

```
