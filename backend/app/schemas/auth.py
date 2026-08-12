from typing import Optional
from pydantic import BaseModel

class DoctorLoginRequest(BaseModel):
    username: str
    password: str
    session_id: Optional[str] = ""
    patient_id: Optional[str] = ""

class PatientLoginRequest(BaseModel):
    patient_id: str
    password: str

class PatientRegisterRequest(BaseModel):
    name: str
    age: int
    gender: str
    email: str
    password: str
