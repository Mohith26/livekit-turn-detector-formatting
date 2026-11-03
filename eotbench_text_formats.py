"""Score LiveKit's text turn-detector models on eot-bench (English) under different
transcript formatting, using eot-bench's own span grid, message builder, adapter
and metrics code.

Needs eot-bench installed (pip install -e path/to/livekit/eot-bench) and the
English parquet from the livekit/eot-bench-data dataset on Hugging Face.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from eot_harness.io import DEFAULT_INFERENCE_INTERVAL, DEFAULT_TRANSCRIPT_LAG, _time_grid, build_messages
from eot_harness.livekit_text_adapter import LiveKitTextTurnDetectorAdapter, _normalize_text
from eot_harness.metrics import compute_metrics_from_predictions

MIN_SILENCE = 0.1
TERMINAL = (".", "?", "!")


def load_rows():
    path = hf_hub_download("livekit/eot-bench-data", "data/en/validation-00000-of-00001.parquet", repo_type="dataset")
    cols = [c for c in pq.read_schema(path).names if c != "audio"]
    return pq.read_table(path, columns=cols).to_pylist()


def prediction_points(rows):
    # same loop as eot_harness.io.iter_transformed_row, minus the audio slicing
    for row in rows:
        spans = row["silence_spans"]
        for span_index, span in enumerate(spans):
            start, end = float(span["start"]), float(span["end"])
            if end - start < MIN_SILENCE - 1e-6:
                continue
            label = "eot" if span_index == len(spans) - 1 else "hold"
            for t in _time_grid(start, end, DEFAULT_INFERENCE_INTERVAL):
                yield {
                    "id": row["id"], "language": "en", "span_index": span_index, "timestamp": t,
                    "silence_dur": round(t - start, 6), "label": label,
                    "messages": build_messages(row, t, transcript_lag=DEFAULT_TRANSCRIPT_LAG),
                }


class TextAdapter(LiveKitTextTurnDetectorAdapter):
    """eot-bench's text adapter with a pluggable formatting step."""

    def __init__(self, revision, fmt):
        super().__init__(revision=revision)
        self.fmt = fmt

    def _batch_text_probs(self, texts):
        # the English ONNX export has a fixed batch size of 1, so score one prompt at a time
        import numpy as np
        out = []
        for text in texts:
            enc = self.tokenizer(text, add_special_tokens=False, return_tensors="np", max_length=self.max_length, truncation=True)
            res = self.session.run(None, {"input_ids": enc["input_ids"].astype(np.int64)})[0]
            out.append(float(np.asarray(res).reshape(-1)[-1]))
        return out

    def _format_chat_input(self, messages):
        chat = []
        for i, m in enumerate(messages):
            if m.get("role") not in ("assistant", "user"):
                continue
            content = str(m.get("content") or "").strip()
            if self.fmt == "normalized":
                content = _normalize_text(content)
            elif self.fmt == "period" and i == len(messages) - 1 and m["role"] == "user":
                if content and not content.endswith(TERMINAL):
                    content = content.rstrip(",") + "."
            if content:
                chat.append({"role": m["role"], "content": content})
        if not chat:
            return ""
        text = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, add_special_tokens=False, tokenize=False)
        cut = text.rfind("<|im_end|>")
        return text[:cut] if cut != -1 else text


VARIANTS = [
    # (name, model revision, formatting)
    ("multilingual v0.4.1, normalized (eot-bench default)", "v0.4.1-intl", "normalized"),
    ("english v1.2.2, transcript as-is", "v1.2.2-en", "raw"),
    ("english v1.2.2, lowercase + no punctuation", "v1.2.2-en", "normalized"),
    ("english v1.2.2, period added at every pause", "v1.2.2-en", "period"),
]


def summarize(name, s):
    out = {"variant": name, "auc": round(s["auc"], 4), "ap": round(s["ap"], 4)}
    for k in ("5pct", "10pct"):
        op = s["operating_points"].get(k) or {}
        out[f"latency_at_{k}_cutoffs_s"] = None if op.get("mean_latency") is None else round(op["mean_latency"], 3)
    return out


def main():
    rows = load_rows()
    points = list(prediction_points(rows))
    print(f"{len(rows)} English turns, {len(points)} prediction points")

    # sanity check: the grid matches the committed span set used for the leaderboard
    ref = sys.argv[1] if len(sys.argv) > 1 else None
    if ref:
        committed = pd.read_parquet(ref)[["id", "span_index", "timestamp"]]
        mine = pd.DataFrame(points)[["id", "span_index", "timestamp"]]
        key = lambda df: set(zip(df.id, df.span_index, df.timestamp.round(6)))
        print("grid matches committed predictions:", key(committed) == key(mine), len(committed))

    results = []
    Path("eotbench_out").mkdir(exist_ok=True)
    for name, revision, fmt in VARIANTS:
        adapter = TextAdapter(revision, fmt)
        scores = []
        for i in range(0, len(points), 256):
            scores += adapter.predict_batch(points[i:i + 256])
        df = pd.DataFrame([{k: v for k, v in p.items() if k != "messages"} | {"p_eot": float(s)} for p, s in zip(points, scores)])
        _, summary = compute_metrics_from_predictions(df)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        df.to_parquet(f"eotbench_out/{slug}.parquet", index=False)
        results.append(summarize(name, summary))
        print(json.dumps(results[-1]))
    json.dump(results, open("eotbench_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
