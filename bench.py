"""Run LiveKit's turn-detector models on the same utterances rendered three ways.

Uses the plugin's own inference runners (livekit-plugins-turn-detector), so text
formatting, truncation and the ONNX session match what an agent runs.
"""

import json
import re
import statistics
import sys
import time

from livekit.plugins.turn_detector.base import HG_MODEL, _download_from_hf_hub
from livekit.plugins.turn_detector.english import _EUORunnerEn
from livekit.plugins.turn_detector.models import MODEL_REVISIONS
from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual
from livekit.plugins.turn_detector.version import __version__


def plain(text: str) -> str:
    # what an STT with punctuation and casing turned off would return
    text = re.sub(r"[^\w\s'-]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def render(fmt: str, text: str, complete: bool) -> str:
    if fmt == "punctuated":
        return text
    if fmt == "punctuated+period":
        # an STT that closes every final segment with a period, even mid-thought
        return text if complete else text.rstrip(",") + "."
    if fmt == "plain":
        return plain(text)
    raise ValueError(fmt)


FORMATS = ["punctuated", "punctuated+period", "plain"]


def threshold(model_type: str) -> float:
    path = _download_from_hf_hub(
        HG_MODEL, "languages.json", revision=MODEL_REVISIONS[model_type], local_files_only=True
    )
    with open(path) as f:
        return json.load(f)["en"]["threshold"]


def main() -> None:
    rows = json.load(open("utterances.json"))
    results = {"plugin_version": __version__, "n_pairs": len(rows), "models": {}}
    per_item = []
    for cls in (_EUORunnerEn, _EUORunnerMultilingual):
        mt = cls.model_type()
        cls._download_files()
        runner = cls()
        runner.initialize()
        thr = threshold(mt)
        model_res = {"revision": MODEL_REVISIONS[mt], "en_threshold": thr, "formats": {}}
        latencies = []
        for fmt in FORMATS:
            probs = {True: [], False: []}
            for i, row in enumerate(rows):
                for complete in (True, False):
                    text = render(fmt, row["complete" if complete else "incomplete"], complete)
                    ctx = [
                        {"role": "assistant", "content": row["agent"]},
                        {"role": "user", "content": text},
                    ]
                    t0 = time.perf_counter()
                    out = json.loads(runner.run(json.dumps({"chat_ctx": ctx}).encode()))
                    latencies.append((time.perf_counter() - t0) * 1000)
                    p = out["eou_probability"]
                    probs[complete].append(p)
                    per_item.append(
                        {"model": mt, "format": fmt, "i": i, "complete": complete,
                         "text": text, "eou_probability": round(p, 5)}
                    )
            n = len(rows)
            model_res["formats"][fmt] = {
                # complete turn scored at/above threshold -> agent replies after the normal delay
                "complete_recognized": sum(p >= thr for p in probs[True]) / n,
                # incomplete turn scored below threshold -> agent waits longer
                "incomplete_held": sum(p < thr for p in probs[False]) / n,
                "median_p_complete": round(statistics.median(probs[True]), 4),
                "median_p_incomplete": round(statistics.median(probs[False]), 4),
            }
        latencies.sort()
        model_res["latency_ms"] = {
            "p50": round(latencies[len(latencies) // 2], 2),
            "p95": round(latencies[int(len(latencies) * 0.95)], 2),
            "runs": len(latencies),
        }
        results["models"][mt] = model_res
    json.dump(results, open("results.json", "w"), indent=1)
    json.dump(per_item, open("per_item.json", "w"), indent=1)

    print(f"livekit-plugins-turn-detector {__version__}, {len(rows)} complete/incomplete pairs\n")
    print(f"{'model':<14}{'format':<20}{'complete ok':>12}{'incomplete held':>17}{'med p(c)':>10}{'med p(i)':>10}")
    for mt, m in results["models"].items():
        for fmt, r in m["formats"].items():
            print(f"{mt:<14}{fmt:<20}{r['complete_recognized']:>12.0%}{r['incomplete_held']:>17.0%}"
                  f"{r['median_p_complete']:>10}{r['median_p_incomplete']:>10}")
        print(f"{'':<14}threshold {m['en_threshold']}, latency p50 {m['latency_ms']['p50']} ms, p95 {m['latency_ms']['p95']} ms")


if __name__ == "__main__":
    sys.exit(main())
