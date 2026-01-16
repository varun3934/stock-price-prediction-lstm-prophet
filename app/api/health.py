#routs to keep backed clean
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status_code":"ok"}