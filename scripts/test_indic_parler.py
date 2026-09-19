import os

# Keep Hugging Face cache inside this project.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HF_HOME = os.path.join(
    PROJECT_ROOT,
    "models",
    "huggingface"
)

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME

import torch
import soundfile as sf

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


MODEL_ID = "ai4bharat/indic-parler-tts"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def main():

    device = get_device()

    print("=" * 60)
    print("Indic Parler-TTS")
    print("=" * 60)
    print(f"Device       : {device}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"HF cache     : {HF_HOME}")
    print(f"Model        : {MODEL_ID}")
    print("=" * 60)

    print("\nLoading model...")

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_ID
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    # Hindi text
    prompt = (
        "         "
        "          "
        "        "
    )

    # Consistent Hindi female voice.
    # Divya is one of the recommended Hindi speakers.
    description = (
        "Divya speaks in a clear, warm and expressive female Hindi voice. "
        "She sounds like a professional Indian entertainment news presenter. "
        "Her delivery is confident, natural and engaging, "
        "with a moderate speaking rate and balanced pitch. "
        "The voice is very clear, close sounding and high quality, "
        "with no background noise."
    )

    print("\nTokenizing...")

    description_inputs = description_tokenizer(
        description,
        return_tensors="pt"
    ).to(device)

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    print("Generating speech...")
    print("This may take a while on the first run.\n")

    with torch.no_grad():

        generation = model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    audio = generation.cpu().numpy().squeeze()

    output_dir = os.path.join(
        PROJECT_ROOT,
        "output"
    )

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "test_divya_hindi.wav"
    )

    sf.write(
        output_file,
        audio,
        model.config.sampling_rate
    )

    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"Output      : {output_file}")
    print(f"Sample rate : {model.config.sampling_rate}")
    print(f"Samples     : {len(audio)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
