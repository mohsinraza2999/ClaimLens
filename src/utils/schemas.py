from pydantic import BaseModel
from dataclasses import dataclass

class RawData(BaseModel):
    claim_status: str
    video_id: int
    video_duration_sec: int
    video_transcription_text: str
    verified_status: int
    author_ban_status: int
    video_view_count: int
    video_like_count: int
    video_share_count: int
    video_download_count: int
    video_comment_count: int

class ProcessData(BaseModel):
    claim_status: str
    video_duration_sec: int
    video_transcription_text: str
    verified_status: int
    author_ban_status: int
    video_view_count: int
    video_like_count: int
    video_share_count: int
    video_download_count: int
    video_comment_count: int

class HealthResponse(BaseModel):
    status: str
    sevice: str
    timestamp: str


class APIResponse(BaseModel):
    claim_status: str
    video_duration_sec: int
    video_transcription_text: str
    verified_status: int
    author_ban_status: int
    video_view_count: int
    video_like_count: int
    video_share_count: int
    video_download_count: int
    video_comment_count: int

class PredictionResponse(BaseModel):
    timestamp: str
    prediction: str
    latency_ms: str