import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
import os
import time
import warnings
from pathlib import Path
import tempfile
import subprocess
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore")

class Config:
    SAMPLE_RATE = 16000
    N_MELS = 80
    MAX_DURATION = 7200.0
    MAX_CHUNK_DURATION = 15.0
    HOP_LENGTH = 160
    N_FFT = 512

config = Config()

class DeepSpeechVocabulary:
    def __init__(self):
        chars = [
            ' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н',
            'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '!', '?', ',', '.', '-', ':', ';', '(', ')', '"', "'"
        ]
        self.char_to_idx = {'<blank>': 0}
        for i, ch in enumerate(chars, start=1):
            self.char_to_idx[ch] = i
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        self.blank_id = 0

    def indices_to_text(self, indices: List[int]) -> str:
        return ''.join(self.idx_to_char[idx] for idx in indices if idx != self.blank_id)

class DeepSpeech2(nn.Module):
    def __init__(self, vocab_size, n_mels=80, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, 100)
            out = self.conv(dummy)
            rnn_input_size = out.size(1) * out.size(2)
        self.rnn = nn.LSTM(rnn_input_size, 320, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(640, vocab_size)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, h, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * h)
        x, _ = self.rnn(x)
        return self.fc(x)

class AudioProcessor:
    @staticmethod
    def convert_video_to_wav(input_path: str) -> str:
        output_path = tempfile.mktemp(suffix='.wav')
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ar', str(config.SAMPLE_RATE),
            '-ac', '1',
            '-acodec', 'pcm_s16le',
            '-y', output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            error_msg = f"FFmpeg error converting '{input_path}': {result.stderr}"
            print(f"ERROR: {error_msg}")
            raise Exception(error_msg)
        return output_path

    @staticmethod
    def load_audio_from_file(file_path: str) -> np.ndarray:
        if os.path.getsize(file_path) == 0:
            error_msg = f"File is empty: {file_path}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        ext = Path(file_path).suffix.lower()
        try:
            if ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                wav_path = AudioProcessor.convert_video_to_wav(file_path)
                audio, sr = sf.read(wav_path)
                os.unlink(wav_path)
            else:
                audio, sr = sf.read(file_path)
        except Exception as e:
            error_msg = f"Error reading audio file '{file_path}': {str(e)}"
            print(f"ERROR: {error_msg}")
            raise Exception(error_msg)
        
        if sr != config.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio

    @staticmethod
    def extract_log_mel(audio: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=config.SAMPLE_RATE, n_mels=config.N_MELS,
            n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel + 100) / 100.0
        return log_mel.astype(np.float32)

    @staticmethod
    def split_into_chunks(audio: np.ndarray, max_chunk_duration: float = 30.0) -> List[np.ndarray]:
        duration = len(audio) / config.SAMPLE_RATE
        if duration <= max_chunk_duration:
            return [audio]
        
        chunk_samples = int(max_chunk_duration * config.SAMPLE_RATE)
        chunks = []
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks

MODEL = None
VOCAB = None
DEVICE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, VOCAB, DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    
    MODEL_PATH = 'Deepsearch_model.pth'
    if os.path.exists(MODEL_PATH):
        try:
            VOCAB = DeepSpeechVocabulary()
            MODEL = DeepSpeech2(VOCAB.vocab_size, n_mels=config.N_MELS).to(DEVICE)
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            MODEL.load_state_dict(state_dict)
            MODEL.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            MODEL = None
    else:
        print(f"Model not found: {MODEL_PATH}")
    
    yield
    
    if MODEL is not None:
        del MODEL
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan, title="DeepSpeech2 API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if MODEL is not None else "model_missing",
        "device": str(DEVICE),
        "max_duration_hours": 2.0,
        "max_chunk_duration_sec": config.MAX_CHUNK_DURATION
    }

@app.get("/api/info")
async def api_info():
    return {
        "service": "DeepSpeech2 Transcribation API",
        "max_duration_hours": 2.0,
        "sample_rate": config.SAMPLE_RATE,
        "supported_formats": [
            "wav", "opus", "mp3", "ogg", "flac", "m4a", "aac",
            "mp4", "mov", "avi", "mkv", "webm"
        ]
    }

@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    output_path: str = Form(None)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File not specified")
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    use_custom_path = False
    final_output_path = None
    if output_path and output_path.strip():
        final_output_path = output_path.strip()
        if not final_output_path.endswith('.txt'):
            final_output_path += '.txt'
        use_custom_path = True

    suffix = Path(file.filename).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        if len(content) == 0:
            os.unlink(tmp.name)
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        try:
            audio = AudioProcessor.load_audio_from_file(tmp_path)
        except Exception as e:
            error_msg = f"Error loading audio '{file.filename}': {str(e)}"
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Unsupported or corrupted audio file: {str(e)}")
        
        duration = len(audio) / config.SAMPLE_RATE
        if duration > config.MAX_DURATION:
            error_msg = f"Audio too long: {duration:.1f} sec (max {config.MAX_DURATION} sec)"
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=400, detail="Audio too long (max 2 hours)")
        
        chunks = AudioProcessor.split_into_chunks(audio, config.MAX_CHUNK_DURATION)
        
        if MODEL is None:
            return {"error": "Model not loaded", "text": "[DEMO MODE]"}
        
        full_text_parts = []
        for chunk in chunks:
            log_mel = AudioProcessor.extract_log_mel(chunk)
            features = torch.from_numpy(log_mel).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = MODEL(features)
                pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
                
                decoded = []
                prev = VOCAB.blank_id
                for p in pred:
                    if p != prev and p != VOCAB.blank_id:
                        decoded.append(p)
                    prev = p
                chunk_text = VOCAB.indices_to_text(decoded)
                full_text_parts.append(chunk_text)
        
        full_text = " ".join(full_text_parts).strip()
        
        if use_custom_path:
            try:
                os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                with open(final_output_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                result_file = final_output_path
            except PermissionError as e:
                error_msg = f"Permission denied writing to: {final_output_path}"
                print(f"ERROR: {error_msg}")
                raise HTTPException(status_code=403, detail="Permission denied: cannot write to the specified path")
            except OSError as e:
                error_msg = f"Invalid output path: {final_output_path} - {str(e)}"
                print(f"ERROR: {error_msg}")
                raise HTTPException(status_code=400, detail=f"Invalid output path: {str(e)}")
        else:
            result_file = tempfile.mktemp(suffix='.txt')
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
        
        return {
            "filename": file.filename,
            "duration_sec": duration,
            "text": full_text,
            "output_file": result_file,
            "chunks_processed": len(chunks)
        }
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)