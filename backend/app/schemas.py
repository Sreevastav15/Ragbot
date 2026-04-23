from pydantic import BaseModel
from datetime import datetime

class DocumentCreate(BaseModel):
    filename: str
    original_path: str
    vector_path: str

class DocumentOut(BaseModel):
    id: int
    filename: str
    upload_date: datetime

    class Config:
        orm_mode=True

class QuestionOut:
    id: int
    question_text: str

    class Config:
        orm_mode=True

class AnswerOut:
    question_text: str
    answer_text: str
