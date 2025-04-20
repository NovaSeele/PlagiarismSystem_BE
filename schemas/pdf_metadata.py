from pydantic import BaseModel
from typing import Optional, List

class PDFMetadata(BaseModel):
    filename: str
    user: str
    description: Optional[str] = None
    content: str
    upload_at: str
    categories: Optional[List[str]] = None
    
class PDFMetadataCategory(PDFMetadata):
    category: Optional[List[str]] = None
    
    
class PDFMetadataInDB(PDFMetadata):
    pass
    