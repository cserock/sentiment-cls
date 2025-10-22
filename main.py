import os, argparse
from fastapi import FastAPI, status
from packages.routers.classify_router import classify
from packages.config import HealthCheckData
from packages import FastAPIRunner
from dotenv import load_dotenv

# 로컬개발환경에서 .env.dev를 사용하도록 설정
if os.environ.get('ENV') == 'local':
    load_dotenv('.env.dev')

app = FastAPI(
    title='Sentiment Classification Model API',
    description='Sentiment Classification 모델 서빙 API입니다.',
)

app.include_router(classify)

@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheckData,
)
def get_health() -> HealthCheckData:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheckData(status=f"OK-{os.environ.get('ENV')}")

if __name__ == "__main__":
    # python main.py --host 127.0.0.1 --port 8000
    parser = argparse.ArgumentParser()
    parser.add_argument('--host')
    parser.add_argument('--port')
    args = parser.parse_args()
    api = FastAPIRunner(args)
    api.run()