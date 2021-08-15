from pydantic import BaseModel


class FTR(BaseModel):
    file_path: str = ""