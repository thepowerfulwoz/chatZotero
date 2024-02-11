from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_key: str
    library_type: str
    library_id: str
    qdrant_url: str

    model_config = SettingsConfigDict(env_file="../.env")