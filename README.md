# Does STT formatting move LiveKit's text turn detector?

LiveKit's text end-of-turn models (`EnglishModel` and `MultilingualModel` in `livekit-plugins-turn-detector`) read the transcript and decide whether the user is done. The English model is fed the transcript as-is (`english.py` skips normalization because it was trained on unnormalized text), while the multilingual model lowercases and strips punctuation first. I wanted to know how much the English model depends on punctuation, since STTs differ a lot in how they format text.

## 1. On eot-bench (400 real English turns)

`eotbench_text_formats.py` runs on LiveKit's open [eot-bench](https://github.com/livekit/eot-bench) English validation set using the harness's own pieces: its span grid and `build_messages` (0.1 s steps, 0.5 s transcript lag), its text adapter, and `compute_metrics_from_predictions`. The only thing I change is how the user's words are formatted before scoring.

Sanity checks: my prediction grid is identical to the committed v1-mini `predictions.parquet` (same 11,191 points), and recomputing metrics from that committed file gives the published numbers (AUC 0.8902, 1.07 s latency at a 5% false-cutoff budget).

| text model and formatting | AUC | latency at 5% false cutoffs | latency at 10% false cutoffs |
|---|---|---|---|
| multilingual v0.4.1, normalized (what the harness's text adapter does) | 0.899 | 0.997 s | 0.698 s |
| english v1.2.2, words as they appear in the dataset (cased, punctuated) | 0.903 | 0.868 s | 0.634 s |
| english v1.2.2, lowercase, no punctuation | 0.754 | 1.292 s | 0.857 s |
| english v1.2.2, a period added at every pause | 0.736 | 1.226 s | 0.886 s |

Latency here is eot-bench's endpointing delay (dead air after the user finishes), not inference time.

The English model goes from the best of the four to clearly the worst depending only on formatting. Without punctuation, or with a period at every pause, it needs about 0.35 to 0.42 s more dead air to stay under 5% false cutoffs. The multilingual model is unaffected by punctuation by design.

Caveat: I don't know how the punctuation in the dataset's word list was produced. If it came from transcribing the whole turn, the English model may be getting a cleaner end-of-sentence signal than a live STT would give, so the "as-is" row could be optimistic.

## 2. A 40-pair check I wrote by hand

`bench.py` scores 40 hand-written complete/incomplete pairs ("I want to book a flight to Boston." vs "I want to book a flight to") through the plugin's own inference runners, and counts a turn as "held" when its probability is below the model's English `unlikely_threshold` from `languages.json`.

| model | format | complete turns recognized | incomplete turns held |
|---|---|---|---|
| en (threshold 0.0289) | punctuated | 100% | 100% |
| en | punctuated, period on incomplete turns | 100% | **0%** |
| en | lowercase, no punctuation | **75%** | 95% |
| multilingual (threshold 0.011) | all three | 100% | 92.5% |

A trailing period on a cut-off sentence ("I want to book a flight to.") put every incomplete turn above the threshold. With no punctuation, 10 of 40 finished answers fell below it (for example "yes i'm allergic to peanuts" scored 0.013). Per-turn scores are in `per_item.json`. Latency per call on Apple Silicon CPU: en p50 1.4 ms, multilingual p50 9.8 ms.

## Limits

- English only. The hand-written set is small and mine; the eot-bench run is the more meaningful number.
- "Period at every pause" is a stress case. I did not measure how often a particular STT does this.
- These text models are deprecated in favor of the audio `livekit.agents.inference.TurnDetector`, which I did not test. This mostly matters for agents still on `EnglishModel`.
- The English ONNX export takes a batch size of 1, so the script scores prompts one at a time.

## Run it

```
python3 -m venv env && ./env/bin/pip install -r requirements.txt
./env/bin/python bench.py

# eot-bench part (Python 3.12; eot-bench pins numpy<2, which needed pyarrow 20 here)
git clone https://github.com/livekit/eot-bench && python3.12 -m venv ebenv
./ebenv/bin/pip install -e eot-bench transformers onnxruntime jinja2 "pyarrow==20.0.0"
./ebenv/bin/python eotbench_text_formats.py
```

Both download model and dataset files from Hugging Face on first run (the English eot-bench parquet is about 160 MB). Results are in `results.json`, `eotbench_results.json` and `eotbench_out/`.
