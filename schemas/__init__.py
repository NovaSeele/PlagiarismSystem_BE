##chỗ này để thiết kế database cho các trường của bảng

from schemas.pdf_metadata import PDFMetadata, PDFMetadataInDB
from schemas.pdf_process import TextRequest, KeywordsResponse
from schemas.user import (
    User,
    UserInDB,
    Token,
    UserCreate,
    UserUpdateAvatar,
    UserChangePassword,
    UserMSVUpdate,
)
from schemas.plagiarism import (
    FileNameComparisonRequest,
    PairPlagiarismRequest,
    MultiPlagiarismRequest,
    PlagiarismResponse,
    LayeredPlagiarismResponse,
    FilesComparisonRequest,
)
from schemas.websocket import LogEntry

# This file exposes all models from the schemas package
# This way, other modules can import from schemas directly
# Example: from schemas import User, PDFMetadata
