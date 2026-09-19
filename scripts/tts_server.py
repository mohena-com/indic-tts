#!/usr/bin/env python3

import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HF_HOME = PROJECT_ROOT / "models" / "huggingface"
os.environ["HF_HOME"] = str(HF_HOME)

import torch
import soundfile as sf

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


MODEL_ID = "ai4bharat/indic-parler-tts"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


VOICE_DESCRIPTIONS = {

    "divya_page3": (
        "Divya is a polished Indian female entertainment presenter "
        "with a glamorous Page 3 celebrity-news style. "
        "She speaks Hindi in a lively, warm and conversational manner, "
        "like a popular Bollywood entertainment channel host. "
        "Her voice is confident, youthful and engaging, "
        "with a subtle sense of excitement and curiosity. "
        "She uses natural pauses and expressive emphasis on celebrity names, "
        "movie titles and important entertainment news. "
        "Her delivery is smooth and energetic without sounding like "
        "a formal television newsreader. "
        "She has clear Hindi pronunciation, pleasant balanced pitch, "
        "moderately fast speaking speed and polished studio-quality audio "
        "with no background noise."
    ),

    "divya": (
        "Divya speaks in a clear, warm and expressive female Hindi voice. "
        "She sounds like a professional Indian entertainment news presenter. "
        "Her delivery is confident, natural and engaging, "
        "with a moderate speaking rate and balanced pitch. "
        "The voice is very clear and high quality, "
        "with no background noise."
    ),

    "rohit": (
        "Rohit speaks in a clear, natural and confident male Hindi voice. "
        "He sounds like a professional Indian entertainment news presenter. "
        "His delivery is engaging and expressive, "
        "with a moderate speaking rate and balanced pitch. "
        "The voice is very clear and high quality, "
        "with no background noise."
    ),
}


def get_device():

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


DEVICE = get_device()


print("=" * 60)
print("Indic TTS API")
print("=" * 60)
print(f"Model : {MODEL_ID}")
print(f"Device: {DEVICE}")
print("=" * 60)

print("Loading model...")

MODEL = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID
).to(DEVICE)

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_ID
)

DESCRIPTION_TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL.config.text_encoder._name_or_path
)

print("Model loaded.")
print("=" * 60)


app = FastAPI(
    title="Indic Hindi TTS",
    version="1.0.0"
)


class TTSRequest(BaseModel):

    text: str
    voice: str = "divya_page3"


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "voices": list(VOICE_DESCRIPTIONS.keys())
    }


@app.post("/tts")
def generate_tts(request: TTSRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    voice = request.voice.lower().strip()

    if voice not in VOICE_DESCRIPTIONS:

        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unknown voice: {voice}",
                "available_voices": list(
                    VOICE_DESCRIPTIONS.keys()
                )
            }
        )

    description = VOICE_DESCRIPTIONS[voice]

    print()
    print("Generating:")
    print(f"Voice: {voice}")
    print(f"Text : {text}")

    description_inputs = DESCRIPTION_TOKENIZER(
        description,
        return_tensors="pt"
    ).to(DEVICE)

    prompt_inputs = TOKENIZER(
        text,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():

        generation = MODEL.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    audio = generation.cpu().numpy().squeeze()

    sample_rate = MODEL.config.sampling_rate

    file_id = uuid.uuid4().hex

    filename = f"{file_id}.wav"

    output_file = OUTPUT_DIR / filename

    sf.write(
        str(output_file),
        audio,
        sample_rate
    )

    duration = len(audio) / sample_rate

    print(
        f"Generated {filename} "
        f"({duration:.2f}s)"
    )

    return {
        "success": True,
        "file": filename,
        "audio_url": f"/audio/{filename}",
        "duration": round(duration, 2),
        "sample_rate": sample_rate,
        "voice": voice
    }


@app.get("/audio/{filename}")
def get_audio(filename: str):

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Audio file not found."
        )

    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )
