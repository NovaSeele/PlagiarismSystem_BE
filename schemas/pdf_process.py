from pydantic import BaseModel

class TextRequest(BaseModel): # Input của văn bản
    text: str

class KeywordsResponse(BaseModel): # Output gồm dãy các từ khoá
    keywords: list[str]