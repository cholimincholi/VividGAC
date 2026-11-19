"""Prompt templates for visual captioning, grounded audio captioning, and audiovisual reasoning."""


VISUAL_PROMPT = (
    "\n        Focusing on the Keywords, please describe the video in detail.\n"
    "        Keywords: {noun_phrases}\n        "
)

VISUAL_PROMPT_NO_NP = "Please describe the video in detail."


VISUAL_PROMPT_A2V = (
    "{audio_caption}\n"
    "Focusing on the Keywords, please describe the video in detail.\n"
    "Keywords: {noun_phrases}"
)


AUDIO_NAIVE = "Describe the sound in detail."
AUDIO_NAIVE_CONF = "Please describe this audio in detail."


AUDIO_VIVIDAC = (
    "{video_caption}\n"
    "Provide a detailed description of the sound according to the Keywords: {keywords}."
)


AUDIO_VCI = AUDIO_VIVIDAC


AUDIO_DECOUPLED = "Describe the sound in detail. Focus especially on: {keywords}."


AUDIO_PROMPT_V1 = (
    "{video_caption}\n"
    "Given the Video Caption, Provide a detailed description related to the Question{question}"
)
AUDIO_PROMPT_V2 = (
    "{video_caption}\n"
    "Provide a detailed description of the sound related to the Question: {question}"
)


CLAP_TEXT_QUERY = "the sound of {cue}"


REASON_SYSTEM = (
    "You are a helpful assistant who can answer questions according to the given "
    "video and audio descriptions."
)

REASON_USER = (
    "Audio Description: {audio_caption}\n"
    "Video Description: {video_caption}\n"
    "Question: {question}"
)

MCQ_SUFFIX = "\n{options}Answer with the option's letter from the given choices directly."
MCQ_LETTERS = "ABCD"


REASON_SYSTEM_CODE = "Based on the provided context (audio and video descriptions), answer the question directly."
REASON_USER_CODE = (
    "\n            Audio Description: {audio_caption}\n"
    "            Video Description: {video_caption}\n\n"
    "            {question}?\n            "
)

CLIP_TEXT_TEMPLATE = "This is {}"
