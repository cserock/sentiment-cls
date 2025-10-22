from pydantic import BaseSettings
from pydantic import BaseModel
from pydantic import Field
from pydantic import validator
from typing import Dict, List, Optional
from functools import lru_cache

class VariableConfig:
    def __init__(self):
        self.host_list = ['127.0.0.1', '0.0.0.0']
        self.port_list = ['8000', '8088']

class Settings(BaseSettings):
    API_KEY: str = 'Bearer 05ac3793-8a82-4e5e-9e24-b084a77042b7'

class APIEnvConfig(BaseSettings):
    host: str = Field(default='0.0.0.0', env='api host')
    port: int = Field(default='8000', env='api server port')

    # host 점검
    @validator("host", pre=True)
    def check_host(cls, host_input):
        if host_input == 'localhost':
            host_input = "127.0.0.1"
        if host_input not in VariableConfig().host_list:
            raise ValueError("host error")
        return host_input
    
    # port 점검
    @validator("port", pre=True)
    def check_port(cls, port_input):
        if port_input not in VariableConfig().port_list:
            raise ValueError("port error")
        return port_input

class APIConfig(BaseModel):
    api_name: str = 'main:app'
    api_info: APIEnvConfig = APIEnvConfig()

class HealthCheckData(BaseModel):
    status: str = "OK"

class ClassifyData(BaseModel):
    sentence: str

# New decorator for cache
@lru_cache()
def get_settings():
    return Settings()