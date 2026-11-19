"""Load audio-visual question data into unified records: qid, dataset, subset, video, audio, question, choices, answer."""
import ast
import json
import os
import random
import re

ROOT = os.environ.get("VIVID_ROOT", ".")
DATA = os.environ.get("VIVID_DATA", f"{ROOT}/data")
AUDIO_CACHE = f"{DATA}/_audio16k"


AVQA_MCQ_N = 2000
AVQA_MCQ_SEED = 0


def _wav(dataset, video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return f"{AUDIO_CACHE}/{dataset}/{stem}.wav"


def load_music_avqa():
    qa = json.load(open(f"{DATA}/music_avqa/repo/data/json/avqa-test.json"))
    out, seen = [], {}
    for x in qa:


        k = seen[x["question_id"]] = seen.get(x["question_id"], -1) + 1
        q = x["question_content"]
        vals = ast.literal_eval(x["templ_values"]) if x["templ_values"] else []
        for v in vals:
            q = re.sub(r"<[^>]+>", v, q, count=1)
        video = f"{DATA}/music_avqa/hf/videos/{x['video_id']}.mp4"
        out.append(dict(qid=f"music_{x['question_id']}" + (f"_dup{k}" if k else ""), dataset="music_avqa",
                        subset="/".join(ast.literal_eval(x["type"])), video=video,
                        audio=_wav("music_avqa", video), question=q, choices=None,
                        answer=x["anser"]))
    return out


def load_music_val():
    """Validation questions whose videos do not occur in the test split; retain one occurrence per question ID."""
    D_ = f"{DATA}/music_avqa/repo/data/json"
    test_vids = {x["video_id"] for x in json.load(open(f"{D_}/avqa-test.json"))}
    out, seen = [], set()
    for x in json.load(open(f"{D_}/avqa-val.json")):
        if x["video_id"] in test_vids or x["question_id"] in seen:
            continue
        seen.add(x["question_id"])
        q = x["question_content"]
        for v in (ast.literal_eval(x["templ_values"]) if x["templ_values"] else []):
            q = re.sub(r"<[^>]+>", v, q, count=1)
        video = f"{DATA}/music_avqa/hf/videos/{x['video_id']}.mp4"
        out.append(dict(qid=f"mval_{x['question_id']}", dataset="music_val",
                        subset="/".join(ast.literal_eval(x["type"])), video=video,
                        audio=_wav("music_avqa", video), question=q, choices=None, answer=x["anser"]))
    return out


def _avqa_video_dir():

    return os.environ.get("AVQA_VIDEO_DIR", f"{DATA}/avqa/videos")


def _avqa_pool():
    qa = json.load(open(f"{DATA}/avqa/repo/data/annotation/val_qa.json"))
    vdir = _avqa_video_dir()
    return [x for x in qa if os.path.exists(f"{vdir}/{x['video_name']}.mp4")] or qa


def load_avqa_mcq(n=AVQA_MCQ_N, seed=AVQA_MCQ_SEED):
    qa = _avqa_pool()
    rng = random.Random(seed)
    qa = sorted(rng.sample(qa, min(n, len(qa))), key=lambda x: x["id"])
    return _avqa_records(qa, "avqa_mcq")


AVQA_DEV_N = 500
AVQA_DEV_SEED = 1


def load_avqa_dev(n=AVQA_DEV_N, seed=AVQA_DEV_SEED):
    test = load_avqa_mcq()
    test_videos = {os.path.basename(r["video"]) for r in test}
    pool = [x for x in _avqa_pool() if f"{x['video_name']}.mp4" not in test_videos]
    qa = sorted(random.Random(seed).sample(pool, min(n, len(pool))), key=lambda x: x["id"])
    return _avqa_records(qa, "avqa_dev")


def _avqa_records(qa, dataset):
    vdir = _avqa_video_dir()
    out = []
    for x in qa:
        video = f"{vdir}/{x['video_name']}.mp4"
        out.append(dict(qid=f"avqa_{x['id']}", dataset=dataset, subset=x["question_type"],
                        video=video, audio=_wav("avqa_mcq", video), question=x["question_text"],
                        choices=x["multi_choice"], answer=x["multi_choice"][x["answer"]]))
    return out


AVH_TASKS = {"Video-driven Audio Hallucination": "V2A",
             "Audio-driven Video Hallucination": "A2V",
             "AV Matching": "AVMatch"}


def load_avhbench(include_captioning=False):
    qa = json.load(open(f"{DATA}/avhbench/QA.json"))
    out = []
    for i, x in enumerate(qa):
        if x["task"] not in AVH_TASKS and not (include_captioning and x["task"] == "AV Captioning"):
            continue
        video = f"{DATA}/avhbench/videos/{x['video_id']}.mp4"
        out.append(dict(qid=f"avh_{i}", dataset="avhbench",
                        subset=AVH_TASKS.get(x["task"], "AVCaption"), video=video,
                        audio=_wav("avhbench", video), question=x["text"], choices=None,
                        answer=x["label"]))
    return out


def load_avh_cap():
    """Load audiovisual captioning records for caption generation."""
    return [dict(r, dataset="avh_cap") for r in load_avhbench(include_captioning=True)
            if r["subset"] == "AVCaption"]


CMM_SUBSETS = {"overrely_audio_ignore_visual": "Audio-biased",
               "overrely_visual_ignore_audio": "Visual-biased",
               "visual-audio-language": "Cross-modal"}


def load_cmm():
    qa = json.load(open(f"{DATA}/cmm/hf/all_data_final_reorg.json"))
    out = []
    for i, x in enumerate(qa):
        if x["sub_category"] not in CMM_SUBSETS:
            continue
        rel = lambda p: os.path.normpath(f"{DATA}/cmm/{p}")
        out.append(dict(qid=f"cmm_{i}", dataset="cmm", subset=CMM_SUBSETS[x["sub_category"]],
                        video=rel(x["video_path"]), audio=rel(x["audio_path"]),
                        question=x["question"], choices=None, answer=x["answer"]))
    return out


LOADERS = {"music_avqa": load_music_avqa, "avqa_mcq": load_avqa_mcq, "avqa_dev": load_avqa_dev,
           "avhbench": load_avhbench, "avh_cap": load_avh_cap, "cmm": load_cmm,
           "music_val": load_music_val}


def load(name, **kw):
    return LOADERS[name](**kw)
