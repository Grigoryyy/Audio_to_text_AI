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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import Optional
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore")

# Конфигурация
class Config:
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 4.0
    
config = Config()

# Словарь
class Vocabulary:
    def __init__(self):
        self.char_to_idx = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2, ' ': 3,
            'а': 4, 'б': 5, 'в': 6, 'г': 7, 'д': 8, 'е': 9, 'ё': 10, 'ж': 11,
            'з': 12, 'и': 13, 'й': 14, 'к': 15, 'л': 16, 'м': 17, 'н': 18,
            'о': 19, 'п': 20, 'р': 21, 'с': 22, 'т': 23, 'у': 24, 'ф': 25,
            'х': 26, 'ц': 27, 'ч': 28, 'ш': 29, 'щ': 30, 'ъ': 31, 'ы': 32,
            'ь': 33, 'э': 34, 'ю': 35, 'я': 36,
            '0': 37, '1': 38, '2': 39, '3': 40, '4': 41, '5': 42, '6': 43,
            '7': 44, '8': 45, '9': 46, '!': 47, '?': 48, ',': 49, '.': 50,
            '-': 51, ':': 52, ';': 53, '(': 54, ')': 55, '"': 56, "'": 57
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def indices_to_text(self, indices):
        text = []
        for idx in indices:
            if idx == 2:
                break
            if idx in self.idx_to_char and idx not in [0, 1]:
                text.append(self.idx_to_char[idx])
        return ''.join(text)

# архитектура модели 
class HybridAudioEncoder(nn.Module):
    def __init__(self, input_dim=1, feature_dim=80, hidden_dim=192):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 96, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(96, feature_dim, kernel_size=5, stride=4, padding=1),
            nn.BatchNorm1d(feature_dim), nn.ReLU(),
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.projection = nn.Linear(feature_dim, hidden_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        return x

class HybridEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.projection(x)
        return x

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim=128):
        super().__init__()
        self.W = nn.Linear(encoder_dim, attention_dim)
        self.U = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
        
    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        scores = self.v(torch.tanh(self.W(encoder_outputs) + self.U(decoder_hidden)))
        attn_weights = F.softmax(scores, dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context, attn_weights.squeeze(-1)

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, decoder_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size, decoder_dim, padding_idx=0)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim)
        
        self.lstm = nn.LSTM(
            input_size=decoder_dim + encoder_dim,
            hidden_size=decoder_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.output_layer = nn.Linear(decoder_dim + encoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, encoder_outputs, targets=None, max_length=100):
        batch_size = encoder_outputs.size(0)
        hidden = None
        input_token = torch.full((batch_size,), 1, device=encoder_outputs.device)
        outputs = []
        
        if targets is not None:
            for t in range(targets.size(1) - 1):
                output, hidden = self._step(input_token, hidden, encoder_outputs)
                outputs.append(output)
                input_token = targets[:, t + 1]
        else:
            for t in range(max_length):
                output, hidden = self._step(input_token, hidden, encoder_outputs)
                outputs.append(output)
                input_token = torch.argmax(output, dim=-1)
                if (input_token == 2).all():
                    break
        
        return torch.stack(outputs, dim=1)
    
    def _step(self, input_token, hidden, encoder_outputs):
        embedded = self.embedding(input_token)
        if hidden is None:
            context = torch.zeros(
                encoder_outputs.size(0),
                encoder_outputs.size(2),
                device=encoder_outputs.device
            )
        else:
            decoder_hidden = hidden[0][-1]
            context, _ = self.attention(decoder_hidden, encoder_outputs)
        
        lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        lstm_out = lstm_out.squeeze(1)
        output_input = torch.cat([lstm_out, context], dim=-1)
        output = self.output_layer(output_input)
        return output, hidden

class HybridCTCAttention(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        self.audio_encoder = HybridAudioEncoder(
            input_dim=1, 
            feature_dim=80,  
            hidden_dim=192   
        )
        
        self.encoder = HybridEncoder(
            input_dim=192,  
            hidden_dim=192, 
            num_layers=4,   
            dropout=0.5     
        )
        
        self.ctc_head = nn.Linear(192, vocab_size)  
        
        self.attention_decoder = AttentionDecoder(
            vocab_size=vocab_size,
            encoder_dim=192,   
            decoder_dim=192,   
            num_layers=2,       
            dropout=0.5        
        )
        
    def forward(self, audio_input, text_input=None):
        audio_features = self.audio_encoder(audio_input)
        encoder_out = self.encoder(audio_features)
        ctc_logits = self.ctc_head(encoder_out)
        
        if text_input is not None:
            attn_logits = self.attention_decoder(encoder_out, text_input)
            return ctc_logits, attn_logits, encoder_out.size(1)
        else:
            return ctc_logits
    
    def decode(self, audio_input, method='attention', max_length=150):
        self.eval()
        with torch.no_grad():
            audio_features = self.audio_encoder(audio_input)
            encoder_out = self.encoder(audio_features)
            
            if method == 'ctc':
                ctc_logits = self.ctc_head(encoder_out)
                ctc_probs = F.softmax(ctc_logits, dim=-1)
                predictions = torch.argmax(ctc_probs, dim=-1)
                
                decoded = []
                for seq in predictions:
                    chars = []
                    prev_char = None
                    for char_idx in seq:
                        char_idx = char_idx.item()
                        if char_idx != 0 and char_idx != prev_char:
                            if char_idx != 2:
                                chars.append(char_idx)
                        prev_char = char_idx
                    decoded.append(chars)
                return decoded
            else:
                attn_logits = self.attention_decoder(encoder_out, max_length=max_length)
                predictions = torch.argmax(attn_logits, dim=-1)
                return [seq.tolist() for seq in predictions]

# обработка аудио
class AudioProcessor:
    @staticmethod
    def convert_to_wav(input_path: str, output_path: str = None) -> str:
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.wav')
        
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ar', str(config.SAMPLE_RATE),
                '-ac', '1',
                '-acodec', 'pcm_s16le',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
            
            return output_path
        except Exception as e:
            raise Exception(f"Conversion failed: {str(e)}")
    
    @staticmethod
    def load_audio(file_content: bytes, filename: str) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            ext = Path(filename).suffix.lower()
            
            if ext in ['.wav', '.opus', '.ogg', '.flac']:
                audio, sr = sf.read(tmp_path)
            elif ext in ['.mp3', '.m4a', '.aac', '.mp4', '.mov', '.avi', '.mkv', '.webm']:
                wav_path = AudioProcessor.convert_to_wav(tmp_path)
                audio, sr = sf.read(wav_path)
                os.unlink(wav_path)
            else:
                try:
                    audio, sr = sf.read(tmp_path)
                except:
                    wav_path = AudioProcessor.convert_to_wav(tmp_path)
                    audio, sr = sf.read(wav_path)
                    os.unlink(wav_path)
            
            if sr != config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            
            return audio
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @staticmethod
    def prepare_audio(audio: np.ndarray) -> torch.Tensor:
        max_samples = int(config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE)
        
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return torch.FloatTensor(audio)

# глобальные переменные для модели
MODEL = None
VOCAB = None
DEVICE = None

# Lifespan менеджер
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, VOCAB, DEVICE
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    
    MODEL_PATH = 'final_hybrid_model2.pth'
    
    if os.path.exists(MODEL_PATH):
        try:
            VOCAB = Vocabulary()
            MODEL = HybridCTCAttention(config, VOCAB.vocab_size).to(DEVICE)
            
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            
            MODEL.load_state_dict(state_dict, strict=True)
            MODEL.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to load with strict=False...")
            try:
                MODEL.load_state_dict(state_dict, strict=False)
                print("Model loaded with strict=False")
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                MODEL = None
    else:
        print(f"Model not found: {MODEL_PATH}")
        print("Running in demo mode")
    
    yield
    
    if MODEL is not None:
        del MODEL
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def fix_state_dict_names(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('decoder.'):
            new_key = 'attention_decoder.' + key[8:]  
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# FastAPI приложение
app = FastAPI(lifespan=lifespan, title="API для распознавания аудио")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if MODEL is not None else "demo_mode",
        "model": "loaded" if MODEL is not None else "not_found",
        "device": str(DEVICE)
    }

@app.get("/api/info")
async def api_info():
    return {
        "service": "API для распознавания аудио",
        "max_audio_length": config.MAX_AUDIO_LENGTH,
        "sample_rate": config.SAMPLE_RATE,
        "supported_formats": [
            "wav", "opus", "mp3", "ogg", "flac", "m4a", "aac",
            "mp4", "mov", "avi", "mkv", "webm"
        ]
    }

@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="файл не указан")
    
    # проверка размера файла
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=400, detail="аудио слишком длинное (макс 50мб)")
    
    start_time = time.time()
    
    try:
        contents = await file.read()
        print(f"обработка: {file.filename} ({len(contents)} байт)")
        
        audio = AudioProcessor.load_audio(contents, file.filename)
        duration = len(audio) / config.SAMPLE_RATE
        print(f"длина: {duration:.2f} секунд")
        
        audio_tensor = AudioProcessor.prepare_audio(audio)
        audio_tensor = audio_tensor.unsqueeze(0).to(DEVICE)
        
        processing_time = (time.time() - start_time) * 1000
        
        # если модель не загружена, то запущено демо
        if MODEL is None or VOCAB is None:
            return {
                "filename": file.filename,
                "duration": duration,
                "ctc_text": "[DEMO MODE] Model не загружена",
                "attention_text": "[DEMO MODE] Place final_hybrid_model2.pth",
                "processing_time_ms": processing_time,
                "message": "запущен демо режим"
            }
        
        # Распознавание
        with torch.no_grad():
            # CTC декодирование
            ctc_indices = MODEL.decode(audio_tensor, method='ctc')
            ctc_text = VOCAB.indices_to_text(ctc_indices[0]) if ctc_indices[0] else ""
            
            # attention декодирование
            attn_indices = MODEL.decode(audio_tensor, method='attention')
            attn_text = VOCAB.indices_to_text(attn_indices[0]) if attn_indices[0] else ""
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"распознано за {total_time:.0f} мс")
        print(f"CTC: {ctc_text}")
        print(f"Attention: {attn_text}")
        
        return {
            "filename": file.filename,
            "duration": duration,
            "ctc_text": ctc_text if ctc_text else "[пусто]",
            "attention_text": attn_text if attn_text else "[пусто]",
            "processing_time_ms": total_time,
            "message": f"аудио обрезано до {config.MAX_AUDIO_LENGTH} секунд" 
                       if duration > config.MAX_AUDIO_LENGTH else None
        }
        
    except Exception as e:
        error_msg = f"ошибка обработки аудио: {str(e)}"
        print(f"ошибка: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

if __name__ == "__main__":
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )