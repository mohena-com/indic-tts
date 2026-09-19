#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

HF_HOME = PROJECT_ROOT / "models" / "huggingface"

# Hugging Face cache
os.environ["HF_HOME"] = str(HF_HOME)

# ============================================================
# Imports
# ============================================================

import torch
import soundfile as sf

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "ai4bharat/indic-parler-tts"

DEFAULT_VOICE = "divya"


VOICE_DESCRIPTIONS = {
    "divya": (
    	"Divya is a polished Indian female entertainment presenter "
	"with a glamorous Page 3 celebrity-news style. "
    	"She speaks Hindi in a lively, warm and conversational manner, "
    	"like a popular Bollywood entertainment channel host. "
    	"Her voice is confident, youthful and engaging, "
   	"with a subtle sense of excitement and curiosity. "
    	"She uses natural pauses and expressive emphasis on celebrity names, "
    	"movie titles and important entertainment news. "
    	"Her delivery is smooth and energetic without sounding dramatic "
    	"or like a formal television newsreader. "
    	"She has clear Hindi pronunciation, a pleasant balanced pitch, "
    	"moderately fast speaking speed and a polished studio-quality voice "
    	"with no background noise."
    ),
    "rohit": (
        "Rohit speaks in a clear, natural and confident male Hindi voice. "
        "He sounds like a professional Indian entertainment news presenter. "
        "His delivery is engaging and expressive, "
        "with a moderate speaking rate and balanced pitch. "
        "The voice is very clear, close sounding and high quality, "
        "with no background noise."
    ),
}


# ============================================================
# Device
# ============================================================

def get_device():
    """
    Prefer Apple Silicon MPS, then CUDA, then CPU.
    """

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ============================================================
# Model
# ============================================================

def load_model(device):
    """
    Load Indic-Parler-TTS model and tokenizers.
    """

    print()
    print("=" * 60)
    print("Loading Indic Parler-TTS")
    print("=" * 60)
    print(f"Model        : {MODEL_ID}")
    print(f"Device       : {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"HF cache     : {HF_HOME}")
    print("=" * 60)

    start = time.time()

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_ID
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    elapsed = time.time() - start

    print(f"Model loaded in {elapsed:.1f} seconds")

    return model, tokenizer, description_tokenizer


# ============================================================
# Text
# ============================================================

def read_text(args):
    """
    Read input text either from --text or --input.
    """

    if args.text:
        text = args.text.strip()

    elif args.input:
        input_file = Path(args.input)

        if not input_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_file}"
            )

        text = input_file.read_text(
            encoding="utf-8"
        ).strip()

    else:
        raise ValueError(
            "Either --text or --input must be supplied."
        )

    if not text:
        raise ValueError(
            "Input text is empty."
        )

    return text


# ============================================================
# Voice
# ============================================================

def get_voice_description(voice):
    """
    Return the description used to condition the speaker.
    """

    voice = voice.lower().strip()

    if voice not in VOICE_DESCRIPTIONS:
        available = ", ".join(VOICE_DESCRIPTIONS.keys())

        raise ValueError(
            f"Unknown voice '{voice}'. "
            f"Available voices: {available}"
        )

    return VOICE_DESCRIPTIONS[voice]


# ============================================================
# Generate
# ============================================================

def generate_voice(
    text,
    voice="divya",
    output_file=None,
):
    """
    Generate Hindi speech.

    Returns:
        Path to generated WAV file.
    """

    device = get_device()

    voice = voice.lower().strip()

    description = get_voice_description(
        voice
    )

    print()
    print("=" * 60)
    print("Indic Parler-TTS")
    print("=" * 60)
    print(f"Voice        : {voice}")
    print(f"Device       : {device}")
    print(f"Characters   : {len(text)}")
    print("=" * 60)

    model, tokenizer, description_tokenizer = load_model(
        device
    )

    print()
    print("Text:")
    print(text)
    print()

    print("Tokenizing...")

    description_inputs = description_tokenizer(
        description,
        return_tensors="pt"
    ).to(device)

    prompt_inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    print("Generating speech...")

    start = time.time()

    with torch.no_grad():

        generation = model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    elapsed = time.time() - start

    audio = (
        generation
        .cpu()
        .numpy()
        .squeeze()
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    if output_file:

        output_path = Path(output_file)

        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    else:

        output_path = (
            PROJECT_ROOT
            / "output"
            / f"{voice}_voice.wav"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    sample_rate = model.config.sampling_rate

    sf.write(
        str(output_path),
        audio,
        sample_rate
    )

    duration = len(audio) / sample_rate

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"Voice        : {voice}")
    print(f"Output       : {output_path}")
    print(f"Sample rate  : {sample_rate}")
    print(f"Duration     : {duration:.2f} sec")
    print(f"Generation   : {elapsed:.2f} sec")
    print(f"Characters   : {len(text)}")
    print("=" * 60)

    return output_path


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Hindi speech using "
            "AI4Bharat Indic-Parler-TTS."
        )
    )

    input_group = parser.add_mutually_exclusive_group(
        required=True
    )

    input_group.add_argument(
        "--text",
        help="Hindi text to convert to speech."
    )

    input_group.add_argument(
        "--input",
        help="UTF-8 text file containing Hindi text."
    )

    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=sorted(VOICE_DESCRIPTIONS.keys()),
        help=(
            "Speaker voice. "
            f"Default: {DEFAULT_VOICE}"
        )
    )

    parser.add_argument(
        "--output",
        help=(
            "Output WAV file. "
            "Relative paths are resolved from the project root."
        )
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    try:

        text = read_text(args)

        output_file = generate_voice(
            text=text,
            voice=args.voice,
            output_file=args.output,
        )

        print()
        print(f"Generated: {output_file}")

    except KeyboardInterrupt:

        print("\nGeneration interrupted.")

        sys.exit(130)

    except Exception as exc:

        print()
        print("ERROR")
        print("=" * 60)
        print(str(exc))
        print("=" * 60)

        sys.exit(1)


if __name__ == "__main__":
    main()
