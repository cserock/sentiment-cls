# sentiment classification

## local 개발환경 설정
### uv 설치
$ brew install uv
### python 3.11.9 환경 구성
$ cd sentiment-cls/
$ uv venv --python 3.11
### 가상환경 시작
$ source .venv/bin/activate
### 가상환경 종료
$ deactivate

### 모덷 다운로드
$ https://drive.google.com/file/d/10Yyg_SkYDR_fQDpE7nW5OmgkOlkkdtCe/view?usp=sharing

### 모델 설치 경로
$ mkdir ~/sentiment-cls/packages/model/sentiment_classification/results
$ cd ~/sentiment-cls/packages/model/sentiment_classification/results  
$ unzip KoBERT_128_CrossEntropyLoss.zip

## Docker BUILD
### 공통
$ docker build --no-cache -t sentiment-cls-api:0.1 -f Dockerfile .
### MAC
$ docker build --no-cache --platform linux/arm64/v8 -t sentiment-cls-api:0.1 -f Dockerfile .
### Linux
$ docker build --no-cache --platform linux/amd64 -t sentiment-cls-api:0.1 -f Dockerfile .

## Docker RUN
### 개발
$ docker run --rm --env-file .env.dev --name sentiment-cls-api -p 8088:8088 -v ${PWD}:/app sentiment-cls-api:0.1
### 운영
$ docker run --rm --env-file .env --name sentiment-cls-api -p 8088:8088 sentiment-cls-api:0.1 >> ~/docker.log &

## Swagger 접속
- local : http://127.0.0.1:8088/docs
- Header Authorization : Bearer 05ac3793-8a82-4e5e-9e24-b084a77042b7

## Docker console
$ docker exec -it sentiment-cls-api bash