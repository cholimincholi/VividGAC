"""Model paths, generation settings, and relevance / audio-grounding thresholds."""
import os

M = os.environ.get("VIVID_MODELS", "models")

VLMS = {
    "qwen2.5-vl-7b": f"{M}/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-3b": f"{M}/Qwen2.5-VL-3B-Instruct",
    "internvl3-8b": f"{M}/InternVL3-8B",
    "internvl2.5-8b": f"{M}/InternVL2_5-8B",
    "videochat-r1-7b": f"{M}/VideoChat-R1_7B_caption",
    "smolvlm2-2.2b": f"{M}/SmolVLM2-2.2B-Instruct",
}
ALM = f"{M}/Qwen2-Audio-7B-Instruct"
LLMS = {
    "qwen2.5-7b": f"{M}/Qwen2.5-7B-Instruct",
    "llama3.1-8b": f"{M}/Llama-3.1-8B-Instruct",
    "mistral-7b": f"{M}/Mistral-7B-Instruct-v0.3",
}

CLIP = os.environ.get("VIVID_CLIP", f"{M}/CLIP-ViT-B-32-laion2B-s34B-b79K")

CLAP_BACKEND = os.environ.get("VIVID_CLAP", "laion-general")
CLAP_PATHS = {
    "msclap-2022": f"{M}/msclap/CLAP_weights_2022.pth",
    "msclap-2023": f"{M}/msclap/CLAP_weights_2023.pth",
    "laion-unfused": f"{M}/clap-htsat-unfused",
    "laion-general": f"{M}/larger_clap_general",


    "laion-general-win3": f"{M}/larger_clap_general",
}

TAU_R = float(os.environ.get("VIVID_TAU_R", 0.85))
TAU_A = float(os.environ.get("VIVID_TAU_A", 0.10))


QUESTION_TERMS = os.environ.get("VIVID_QTERMS", "default")

QUESTION_POS = ("NOUN", "ADJ")
QUESTION_STOP = {"video", "audio"}

CUE_POS = ("NOUN",)
CUE_STOP = {"right", "left", "performance", "setting", "video", "top", "bottom", "side", "room", "manner",
            "technique", "proficiency", "passion"}
KEEP_QUESTION_KEYWORDS = True

REASON_STYLE = os.environ.get("VIVID_REASON_STYLE", "code")

CAPTION_RETRIES = 3


KW_RELEVANCE = os.environ.get("VIVID_KWREL", "code")


CAPTION_GEN_PAPER = {
    "qwen2.5-vl": dict(temperature=1e-6, top_p=1.0, top_k=50, repetition_penalty=1.05),
    "smolvlm2": dict(temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0),
    "videochat-r1": dict(temperature=0.1, top_p=0.001, top_k=1, repetition_penalty=1.05),
    "qwen2-audio": dict(temperature=0.7, top_p=0.5, top_k=20, repetition_penalty=1.1),

    "internvl": dict(),
}

DECODING = os.environ.get("VIVID_DECODING", "default")


if DECODING == "paper":
    REASON_STYLE = os.environ.get("VIVID_REASON_STYLE", "paper")
    KW_RELEVANCE = os.environ.get("VIVID_KWREL", "qnoun_max")
    QUESTION_TERMS = os.environ.get("VIVID_QTERMS", "nouns")
    KEEP_QUESTION_KEYWORDS = False


CAPTION_GEN_DEFAULT = {
    "qwen2.5-vl": dict(max_new_tokens=80, do_sample=True, temperature=1e-6, top_p=1.0, top_k=50, repetition_penalty=1.05),
    "internvl": dict(max_new_tokens=80, do_sample=True, temperature=1.0),
    "smolvlm2": dict(max_new_tokens=80, do_sample=True, **CAPTION_GEN_PAPER["smolvlm2"]),
    "videochat-r1": dict(max_new_tokens=80, do_sample=True, **CAPTION_GEN_PAPER["videochat-r1"]),
    "qwen2-audio": dict(max_new_tokens=100, do_sample=False, repetition_penalty=1.1),
}


def caption_gen_kwargs(family):
    if DECODING == "default":
        return dict(num_beams=1, **CAPTION_GEN_DEFAULT[family])
    base = dict(max_new_tokens=100, num_beams=1)
    if DECODING == "greedy":
        return dict(base, do_sample=False)
    return dict(base, do_sample=True, **CAPTION_GEN_PAPER[family])


REASON_GEN = dict(max_new_tokens=30, num_beams=1, do_sample=False)
REASON_MAX_DEFAULT = {"music_avqa": 64, "music_val": 64, "avqa_mcq": 64, "avqa_dev": 64, "avhbench": 200, "cmm": 200}


def reason_gen(dataset):
    if DECODING == "default" and dataset in REASON_MAX_DEFAULT:
        return dict(REASON_GEN, max_new_tokens=REASON_MAX_DEFAULT[dataset])
    return REASON_GEN

DESCRIBE_GEN = dict(max_new_tokens=77, num_beams=1, do_sample=False)


BATCH = int(os.environ.get("VIVID_BATCH", 1))

REUSE_GREEDY = os.environ.get("VIVID_REUSE", "1") == "1"


FRAMES = {"music_avqa": 16, "music_val": 16, "avqa_mcq": 16, "avqa_dev": 16, "avhbench": 32, "cmm": 32, "avh_cap": 32}

AUDIO_MAX_SEC = 30
VIDEO_FPS = 1.0
VIDEO_MAX_FRAMES = 16
VIDEO_MAX_PIXELS = 360 * 420
INTERNVL_SEGMENTS = 8


def family_of(vlm_name):
    for k in ("qwen2.5-vl", "internvl", "videochat-r1", "smolvlm2"):
        if vlm_name.startswith(k):
            return k
    raise ValueError(vlm_name)


DATA = os.environ.get("VIVID_DATA", "data")


def fixed_video_path(v):
    return os.path.join(DATA, "_fixed_videos", os.path.relpath(os.path.abspath(v), os.path.abspath(DATA)))


def fixed_video(v):
    p = fixed_video_path(v)
    return p if os.path.exists(p) else v
