"""Audio-visual reasoning pipeline.

vcap: generate a visual caption conditioned on question terms.
keywords: extract candidate cues and score question relevance with CLIP.
clap: score candidate cues against the audio using CLAP.
acap: generate an audio caption from grounded cues; omit the full visual caption.
reason: answer the question using the visual and audio captions.

Alternative caption methods support independent audio, relevance filtering,
and separate grounding / context-decoupling configurations."""
import argparse
import glob
import json
import os
import re
import time

import config as C
import data as D
import prompts as P

RUNS = os.environ.get("VIVID_OUTPUTS", "outputs")
METHODS = ["naive", "vividac", "vci", "gate", "vividgac", "vci_dec", "clip_dec", "prompt_v1", "prompt_v2", "a2v"]
NEEDS_VCAP = {m for m in METHODS if m not in ("naive", "a2v")}


def read_jsonl(path_noext):
    rows = {}
    for p in sorted(glob.glob(path_noext + ".jsonl*")):
        if p.endswith(".meta.json"):
            continue
        for line in open(p):
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


class Writer:
    def __init__(self, path_noext, shard, nshard):
        os.makedirs(os.path.dirname(path_noext), exist_ok=True)
        self.path = path_noext + (".jsonl" if nshard == 1 else f".jsonl.part{shard}of{nshard}")
        self.done = set(read_jsonl(path_noext))
        self.f = open(self.path, "a")
        self.meta_path = self.path + ".meta.json"
        self.t0, self.n, self.busy, self.calls, self.reused = time.time(), 0, 0.0, 0, 0

    def write(self, row, sec, calls=1):
        row["sec"] = round(sec, 4)
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()
        if row.get("reused"):
            self.reused += 1
            return
        self.n += 1
        self.busy += sec
        self.calls += calls

    def close(self, load_sec, extra=None):
        self.f.close()
        meta = dict(n_new=self.n, n_reused=self.reused, load_sec=round(load_sec, 2), wall_sec=round(time.time() - self.t0, 2),
                    mean_sec_per_item=round(self.busy / max(self.n, 1), 4),
                    model_calls=self.calls, decoding=C.DECODING, **(extra or {}))
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                meta["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        except Exception:
            pass
        json.dump(meta, open(self.meta_path, "w"), indent=1)


_NLP = None


def nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def noun_phrases(text):
    return list(dict.fromkeys(c.text.lower() for c in nlp()(text).noun_chunks))


def nouns(text):
    return list(dict.fromkeys(t.lemma_.lower() for t in nlp()(text) if t.pos_ in ("NOUN", "PROPN")))


def question_keywords(text):
    """Question keywords: noun / adjective lemmas, noun lemmas, or noun phrases."""
    if C.QUESTION_TERMS == "noun_chunks":
        return noun_phrases(text)
    if C.QUESTION_TERMS == "nouns":
        return nouns(text)
    return list(dict.fromkeys(t.lemma_.lower() for t in nlp()(text)
                              if t.pos_ in C.QUESTION_POS and t.text.lower() not in C.QUESTION_STOP))


def question_nouns_code(text):
    """Question noun lemmas, excluding generic video / audio terms."""
    return list(dict.fromkeys(t.lemma_.lower() for t in nlp()(text)
                              if t.pos_ == "NOUN" and t.text.lower() not in C.QUESTION_STOP))


def clean_caption(text):
    """Normalize whitespace and retain complete caption sentences; discard the final complete sentence when the trailing sentence is unfinished."""
    text = str(text).replace("\n", " ").strip()
    sents = re.findall(r"[^.!?]*[.!?]", text)
    if text and text[-1] not in ".!?" and sents:
        sents = sents[:-1]
    sents = [x.strip() for x in sents if x.strip()]
    return " ".join(sents) if sents else " "


def generate_clean(fn):
    """Regenerate an empty cleaned caption up to CAPTION_RETRIES times."""
    for _ in range(C.CAPTION_RETRIES):
        cap = clean_caption(fn())
        if cap.strip():
            break
    return cap


def cue_candidates(text):
    """Candidate cue lemmas from a visual caption, with optional part-of-speech and stop-word filtering."""
    if C.QUESTION_TERMS != "default":
        return nouns(text)
    return list(dict.fromkeys(t.lemma_.lower() for t in nlp()(text)
                              if t.pos_ in C.CUE_POS and t.lemma_.lower() not in C.CUE_STOP))


def question_text(r):
    """Format a question and optional lettered answer choices."""
    q = r["question"]
    if r["choices"]:
        q += P.MCQ_SUFFIX.format(options="".join(f"{L}. {c}\n" for L, c in zip(P.MCQ_LETTERS, r["choices"])))
    return "Question: " + q if C.REASON_STYLE == "code" else q


def stage_vcap(recs, a, w):
    import models
    t = time.time(); vlm = models.VisualAgent(a.vlm); load = time.time() - t
    ac = read_jsonl(path(a, "acap", "naive")) if a.method == "a2v" else {}
    for r in recs:
        if a.dataset == "avh_cap":
            nps = []
        else:
            nps = question_keywords(r["question"])
        if a.method == "a2v":
            if r["qid"] not in ac:
                continue
            prompt = (P.VISUAL_PROMPT_A2V.format(audio_caption=ac[r["qid"]]["caption"], noun_phrases=", ".join(nps))
                      if nps else ac[r["qid"]]["caption"] + "\n" + P.VISUAL_PROMPT_NO_NP)
        else:

            prompt = P.VISUAL_PROMPT.format(noun_phrases=repr(nps)) if a.dataset != "avh_cap" else P.VISUAL_PROMPT_NO_NP
        t = time.time()
        cap = generate_clean(lambda: vlm.caption(r["video"], prompt, nframes=C.FRAMES.get(a.dataset)))
        w.write(dict(qid=r["qid"], np=nps, prompt=prompt, caption=cap), time.time() - t)
    return load


def stage_keywords(recs, a, w):
    import models
    vc = read_jsonl(path(a, "vcap", a.vlm))
    t = time.time(); clip = models.ClipText(); load = time.time() - t
    for r in recs:
        if r["qid"] not in vc:
            continue
        t = time.time()
        cand = cue_candidates(vc[r["qid"]]["caption"])
        qn = question_keywords(r["question"])
        qc = question_nouns_code(r["question"])
        sim, sim_qn, sim_code = {}, {}, {}
        if cand and qc:
            e = clip.embed([P.CLIP_TEXT_TEMPLATE.format(x) for x in qc + cand])
            m = e[len(qc):] @ e[:len(qc)].T
            for j, c in enumerate(cand):
                for i, q in enumerate(qc):
                    if q == c:
                        m[j, i] = -1.0
            sim_code = {n: round(float(v), 5) for n, v in zip(cand, m.max(1).values.tolist())}
        if cand:
            e = clip.embed([r["question"]] + cand + qn)
            ec = e[1:1 + len(cand)]
            sim = {n: round(float(s), 5) for n, s in zip(cand, (ec @ e[0]).tolist())}
            if qn:
                mx = (ec @ e[1 + len(cand):].T).max(1).values.tolist()
                sim_qn = {n: round(float(s), 5) for n, s in zip(cand, mx)}

        w.write(dict(qid=r["qid"], cand=cand, q_nouns=qn, q_code=qc, clip_sim=sim, clip_sim_qn=sim_qn,
                     clip_sim_code=sim_code), time.time() - t)
    return load


def stage_clap(recs, a, w):
    import models
    kw = read_jsonl(path(a, "keywords", a.vlm))
    t = time.time(); clap = models.Clap(); load = time.time() - t
    for r in recs:
        if r["qid"] not in kw:
            continue
        cand = kw[r["qid"]]["cand"]
        t = time.time()
        sc = clap.scores(r["audio"], [P.CLAP_TEXT_QUERY.format(cue=c) for c in cand]) if cand else []
        w.write(dict(qid=r["qid"], clap={c: round(float(s), 5) for c, s in zip(cand, sc)}), time.time() - t)
    return load


def audio_prompt(method, r, vc, kw, cl):
    cap = vc["caption"] if vc else ""
    if method == "naive":
        return P.AUDIO_NAIVE, []
    cand = kw["cand"]
    if method in ("vividac", "clip_dec"):
        rel = {"qnoun_max": kw["clip_sim_qn"], "code": kw.get("clip_sim_code", {})}.get(C.KW_RELEVANCE, kw["clip_sim"])
        ks = [c for c in cand if rel.get(c, -1) >= C.TAU_R]
        if C.KEEP_QUESTION_KEYWORDS:
            qk = kw.get("q_code", []) if C.KW_RELEVANCE == "code" else kw.get("q_nouns", [])
            ks = list(dict.fromkeys(qk + ks))
        if method == "clip_dec":
            return (P.AUDIO_DECOUPLED.format(keywords=", ".join(ks)) if ks else P.AUDIO_NAIVE), ks
        return P.AUDIO_VIVIDAC.format(video_caption=cap, keywords=", ".join(ks)), ks
    if method == "vci":
        return P.AUDIO_VCI.format(video_caption=cap, keywords=", ".join(cand)), cand
    if method == "vci_dec":
        return (P.AUDIO_DECOUPLED.format(keywords=", ".join(cand)) if cand else P.AUDIO_NAIVE), cand
    gated = [c for c in cand if cl["clap"].get(c, -1) >= C.TAU_A]
    if method == "gate":
        return P.AUDIO_VCI.format(video_caption=cap, keywords=", ".join(gated)), gated
    if method == "vividgac":
        return (P.AUDIO_DECOUPLED.format(keywords=", ".join(gated)) if gated else P.AUDIO_NAIVE), gated
    if method == "prompt_v1":
        return P.AUDIO_PROMPT_V1.format(video_caption=cap, question=r["question"]), []
    if method == "prompt_v2":
        return P.AUDIO_PROMPT_V2.format(video_caption=cap, question=r["question"]), []
    raise ValueError(method)


def stage_acap(recs, a, w):
    import models
    vc = read_jsonl(path(a, "vcap", a.vlm)) if a.method in NEEDS_VCAP else {}
    kw = read_jsonl(path(a, "keywords", a.vlm)) if a.method in NEEDS_VCAP else {}
    cl = read_jsonl(path(a, "clap", f"{a.vlm}__{C.CLAP_BACKEND}")) if a.method in ("gate", "vividgac") else {}

    cache = greedy_cache(a) if C.DECODING in ("greedy", "default") and C.REUSE_GREEDY else {}
    todo = []
    for r in recs:
        q = r["qid"]
        if a.method in NEEDS_VCAP and q not in kw:
            continue
        if a.method in ("gate", "vividgac") and q not in cl:
            continue
        prompt, ks = audio_prompt(a.method, r, vc.get(q), kw.get(q), cl.get(q))
        hit = cache.get((r["audio"], prompt))
        if hit:
            w.write(dict(qid=q, prompt=prompt, keywords=ks, caption=hit, reused=True), 0.0)
        else:
            todo.append((q, r["audio"], prompt, ks))
    if not todo:
        return 0.0
    t = time.time(); alm = models.AudialAgent(); load = time.time() - t
    for i in range(0, len(todo), C.BATCH):
        chunk = todo[i:i + C.BATCH]
        t = time.time(); caps = alm.caption_batch([c[1] for c in chunk], [c[2] for c in chunk])
        sec = (time.time() - t) / len(chunk)
        for (q, au, prompt, ks), cap in zip(chunk, caps):
            first = [cap]
            cap = generate_clean(lambda: first.pop() if first else alm.caption(au, prompt))
            w.write(dict(qid=q, prompt=prompt, keywords=ks, caption=cap), sec)
    return load


def greedy_cache(a):
    """(audio, prompt) -> caption from every finished acap file of this tag/dataset (greedy only)."""
    recs = {r["qid"]: r["audio"] for r in D.load(a.dataset)}
    cache = {}
    for p in glob.glob(path(a, "acap", "*") + ".jsonl*"):
        if p.endswith(".meta.json"):
            continue
        for line in open(p):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["qid"] in recs:
                cache.setdefault((recs[row["qid"]], row["prompt"]), row["caption"])
    return cache


def answer_post(ans):
    """Normalize the final answer text."""
    if "### Final Answer:" in ans:
        part = ans.split("### Final Answer:", 1)[-1].strip()
        return part or ans
    return ans.replace("Answer:", "").replace("Summary:", "").strip()


def stage_reason(recs, a, w):
    import models
    vc = read_jsonl(path(a, "vcap", f"{a.vlm}__a2v" if a.method == "a2v" else a.vlm))
    ac = read_jsonl(path(a, "acap", acap_name(a)))
    gen = C.DESCRIBE_GEN if a.dataset == "avh_cap" else C.reason_gen(a.dataset)
    t = time.time(); llm = models.ReasonLLM(a.llm); load = time.time() - t
    todo = [r for r in recs if r["qid"] in vc and r["qid"] in ac]
    for i in range(0, len(todo), C.BATCH):
        chunk = todo[i:i + C.BATCH]
        tmpl, system = ((P.REASON_USER_CODE, P.REASON_SYSTEM_CODE) if C.REASON_STYLE == "code"
                        else (P.REASON_USER, P.REASON_SYSTEM))
        users = [tmpl.format(audio_caption=ac[r["qid"]]["caption"], video_caption=vc[r["qid"]]["caption"],
                             question=question_text(r)) for r in chunk]
        t = time.time(); preds = [answer_post(x) for x in llm.answer_batch(system, users, gen)]
        sec = (time.time() - t) / len(chunk)
        for r, pred in zip(chunk, preds):
            w.write(dict(qid=r["qid"], subset=r["subset"], answer=r["answer"], choices=r["choices"], pred=pred), sec)
    return load


STAGES = dict(vcap=stage_vcap, keywords=stage_keywords, clap=stage_clap, acap=stage_acap, reason=stage_reason)


def acap_name(a):
    if a.method in ("naive", "a2v"):
        return "naive"
    name = f"{a.vlm}__{a.method}"
    if a.method in ("gate", "vividgac") and C.TAU_A != 0.10:
        name += f"__tauA{C.TAU_A:g}"
    if a.method in ("gate", "vividgac") and C.CLAP_BACKEND != "laion-general":
        name += f"__{C.CLAP_BACKEND}"
    if a.method in ("vividac", "clip_dec") and C.TAU_R != 0.85:
        name += f"__tauR{C.TAU_R:g}"
    if a.method == "vividac" and C.KW_RELEVANCE != "question":
        name += f"__{C.KW_RELEVANCE}"
    return name


def out_name(a):
    if a.stage == "reason":
        suffix = "a2v" if a.method == "a2v" else acap_name(a).split("__", 1)[-1]
        return f"{a.vlm}__{a.llm}__{suffix}"
    return {"vcap": f"{a.vlm}__a2v" if a.method == "a2v" else a.vlm, "keywords": a.vlm, "clap": f"{a.vlm}__{C.CLAP_BACKEND}",
            "acap": acap_name(a)}[a.stage]


def path(a, stage, name):
    return f"{RUNS}/{a.tag}/{a.dataset}/{stage}/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--dataset", required=True, choices=list(D.LOADERS))
    ap.add_argument("--vlm", choices=list(C.VLMS))
    ap.add_argument("--llm", choices=list(C.LLMS))
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--tag", default=f"{C.DECODING}_s0", help="run tag, e.g. greedy_s0 / paper_s1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N records")
    a = ap.parse_args()

    import torch
    import transformers
    transformers.set_seed(a.seed)
    torch.backends.cudnn.deterministic = True

    recs = D.load(a.dataset)
    if a.limit:
        recs = recs[:a.limit]
    recs = recs[a.shard::a.nshard]
    w = Writer(path(a, a.stage, out_name(a)), a.shard, a.nshard)
    recs = [r for r in recs if r["qid"] not in w.done]
    print(f"[{a.stage}] {a.dataset} {out_name(a)}: {len(recs)} to do", flush=True)
    load = STAGES[a.stage](recs, a, w) if recs else 0.0
    w.close(load, extra=dict(stage=a.stage, dataset=a.dataset, seed=a.seed,
                             tau_r=C.TAU_R, tau_a=C.TAU_A, batch=C.BATCH, clap=C.CLAP_BACKEND, clip=C.CLIP))


if __name__ == "__main__":
    main()
