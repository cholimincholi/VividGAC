"""Model wrappers for visual captioning, audio captioning, reasoning, CLIP, and CLAP."""
import os

import torch

import config as C


def device():
    if os.environ.get("VIVIDGAC_ALLOW_GPU") == "1" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def require_gpu_permission():
    if os.environ.get("VIVIDGAC_ALLOW_GPU") != "1":
        raise SystemExit("GPU stage blocked: set VIVIDGAC_ALLOW_GPU=1 (and CUDA_VISIBLE_DEVICES) "
                         "to enable GPU execution.")


class VisualAgent:
    def __init__(self, name):
        require_gpu_permission()
        self.name, self.family, path = name, C.family_of(name), C.VLMS[name]
        self.gen = C.caption_gen_kwargs(self.family)
        if self.family in ("qwen2.5-vl", "videochat-r1"):
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
            self.proc = AutoProcessor.from_pretrained(path)
        elif self.family == "smolvlm2":
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.model = AutoModelForImageTextToText.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
            self.proc = AutoProcessor.from_pretrained(path)
        else:
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                   trust_remote_code=True, device_map="cuda").eval()
            self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    @torch.no_grad()
    def caption(self, video, prompt, nframes=None):
        """nframes: uniformly sampled frames (16 / 32 per dataset); None -> fps-based sampling"""
        video = C.fixed_video(video)
        if self.family in ("qwen2.5-vl", "videochat-r1"):
            from qwen_vl_utils import process_vision_info
            vid = {"type": "video", "video": video, "max_pixels": C.VIDEO_MAX_PIXELS}
            vid.update({"nframes": nframes} if nframes else {"fps": C.VIDEO_FPS, "max_frames": C.VIDEO_MAX_FRAMES})
            msgs = [{"role": "user", "content": [vid, {"type": "text", "text": prompt}]}]
            text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids, vkw = process_vision_info(msgs, return_video_kwargs=True)
            inp = self.proc(text=[text], images=imgs, videos=vids, return_tensors="pt", **vkw).to("cuda")
            out = self.model.generate(**inp, **self.gen)
            return self.proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if self.family == "smolvlm2":
            msgs = [{"role": "user", "content": [{"type": "video", "path": video},
                                                 {"type": "text", "text": prompt}]}]
            inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                return_dict=True, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
            out = self.model.generate(**inp, **self.gen)
            return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        pix, npl = internvl_load_video(video, nframes or C.INTERNVL_SEGMENTS)
        q = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(npl))) + prompt
        gen = {k: v for k, v in self.gen.items() if k != "num_beams"}
        return self.model.chat(self.tok, pix.to(torch.bfloat16).cuda(), q, gen,
                               num_patches_list=npl).strip()


def internvl_load_video(path, num_segments, input_size=448):
    """InternVL model-card video loader: uniform segments, one 448px tile per frame."""
    import numpy as np
    import torchvision.transforms as T
    from decord import VideoReader, cpu
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode
    tf = T.Compose([T.Lambda(lambda im: im.convert("RGB")),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    n = len(vr)
    idx = np.linspace(0, n - 1, num_segments + 2)[1:-1].round().astype(int)
    frames = [tf(Image.fromarray(vr[i].asnumpy())).unsqueeze(0) for i in idx]
    return torch.cat(frames), [1] * len(frames)


class AudialAgent:
    def __init__(self):
        require_gpu_permission()
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(C.ALM)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            C.ALM, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.gen = C.caption_gen_kwargs("qwen2-audio")
        self.sr = self.proc.feature_extractor.sampling_rate
        self.proc.tokenizer.padding_side = "left"

    def caption(self, audio, prompt):
        return self.caption_batch([audio], [prompt])[0]

    @torch.no_grad()
    def caption_batch(self, audios, prompts):
        """batch of 1 = paper setting (no padding); larger batches use left padding (VIVID_BATCH)."""
        import librosa
        texts = [self.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "audio", "audio_url": a}, {"type": "text", "text": p}]}],
            add_generation_prompt=True, tokenize=False) for a, p in zip(audios, prompts)]
        wavs = [librosa.load(a, sr=self.sr, duration=C.AUDIO_MAX_SEC)[0] for a in audios]
        inp = self.proc(text=texts, audios=wavs, sampling_rate=self.sr, return_tensors="pt", padding=True).to("cuda")
        out = self.model.generate(**inp, **self.gen)
        return [t.strip() for t in self.proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)]


class ReasonLLM:
    def __init__(self, name):
        require_gpu_permission()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = C.LLMS[name]
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                          device_map="cuda").eval()

    def _template(self, system, user):
        try:
            return self.tok.apply_chat_template([{"role": "system", "content": system},
                                                 {"role": "user", "content": user}],
                                                tokenize=False, add_generation_prompt=True)
        except Exception:
            return self.tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}],
                                                tokenize=False, add_generation_prompt=True)

    def answer(self, system, user, gen=None):
        return self.answer_batch(system, [user], gen)[0]

    @torch.no_grad()
    def answer_batch(self, system, users, gen=None):
        texts = [self._template(system, u) for u in users]
        inp = self.tok(texts, return_tensors="pt", add_special_tokens=False, padding=True).to("cuda")
        out = self.model.generate(**inp, **(gen or C.REASON_GEN), pad_token_id=self.tok.pad_token_id)
        return [t.strip() for t in self.tok.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)]


class ClipText:
    def __init__(self):
        from transformers import CLIPModel, CLIPTokenizer
        self.dev = device()
        self.tok = CLIPTokenizer.from_pretrained(C.CLIP)
        self.model = CLIPModel.from_pretrained(C.CLIP).to(self.dev).eval()

    @torch.no_grad()
    def embed(self, texts):
        t = self.tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.dev)
        return torch.nn.functional.normalize(self.model.get_text_features(**t), dim=-1).cpu()


class Clap:
    def __init__(self, backend=C.CLAP_BACKEND):
        self.backend, self.dev = backend, device()
        if backend.startswith("msclap"):
            from msclap import CLAP
            ver = backend.split("-")[1]
            self.m = CLAP(model_fp=C.CLAP_PATHS[backend], version=ver, use_cuda=self.dev == "cuda")
        else:
            from transformers import ClapModel, ClapProcessor
            self.proc = ClapProcessor.from_pretrained(C.CLAP_PATHS[backend])
            self.m = ClapModel.from_pretrained(C.CLAP_PATHS[backend]).to(self.dev).eval()

    @torch.no_grad()
    def scores(self, audio, texts):
        """cosine similarity between one audio file and each text"""
        if self.backend.startswith("msclap"):
            a = torch.nn.functional.normalize(self.m.get_audio_embeddings([audio]), dim=-1)
            t = torch.nn.functional.normalize(self.m.get_text_embeddings(texts), dim=-1)
            return (t @ a.T).squeeze(-1).cpu().tolist()
        import librosa
        wav, _ = librosa.load(audio, sr=48000)
        if self.backend.endswith("-win3"):
            win = 10 * 48000
            wav = wav[:C.AUDIO_MAX_SEC * 48000]
            wins = [wav[i:i + win] for i in range(0, len(wav), win) if len(wav[i:i + win]) >= 48000 // 2] or [wav]
            ai = self.proc(audios=wins, sampling_rate=48000, return_tensors="pt").to(self.dev)
            a = torch.nn.functional.normalize(self.m.get_audio_features(**ai), dim=-1).mean(0, keepdim=True)
            a = torch.nn.functional.normalize(a, dim=-1)
            ti = self.proc(text=texts, padding=True, return_tensors="pt").to(self.dev)
            t = torch.nn.functional.normalize(self.m.get_text_features(**ti), dim=-1)
            return (t @ a.T).squeeze(-1).cpu().tolist()
        ai = self.proc(audios=[wav], sampling_rate=48000, return_tensors="pt").to(self.dev)
        ti = self.proc(text=texts, padding=True, return_tensors="pt").to(self.dev)
        a = torch.nn.functional.normalize(self.m.get_audio_features(**ai), dim=-1)
        t = torch.nn.functional.normalize(self.m.get_text_features(**ti), dim=-1)
        return (t @ a.T).squeeze(-1).cpu().tolist()


