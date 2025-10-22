import json
import re
from fastapi import APIRouter, Depends
from packages.config import ClassifyData
from packages.services.classify_sentiment import ClassifySentimentService
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey
import auth

classify = APIRouter(prefix='/classification')

@classify.post("/sentiment")
def classify_sentiment(inp: ClassifyData, api_key: APIKey = Depends(auth.get_api_key)):
    sentence = inp.sentence

    # 특수문자 제거
    sentence = re.sub('[-=+#^@*\"※~ㆍ』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', sentence)
    # 줄바꿈 제거
    sentence = re.sub(r"\n", "", sentence)

    model = ClassifySentimentService()
    result_json = model.classify(sentence)
    result_response = JSONResponse(json.loads(result_json))
    return result_response
