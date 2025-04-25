from pydantic import BaseModel
from typing import Optional, List

class PDFMetadata(BaseModel):
    filename: str
    user: str
    content: str
    upload_at: str
    
class PDFMetadataInDB(PDFMetadata):
    pass
    