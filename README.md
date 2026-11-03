# Does STT formatting move LiveKit's text turn detector?

LiveKit's text end-of-turn models (`EnglishModel` and `MultilingualModel` in `livekit-plugins-turn-detector`) score the user's transcript and decide whether the agent should wait for more. The English model is fed the transcript as-is (no normalization, per `english.py`), while the multilingual model lowercases and strips punctuation first. I wanted to see what that means when the same words arrive from STTs that format text differently.

## What I ran

40 hand-written English turns, each as a pair: a complete answer ("I want to book a flight to Boston.") and the same answer cut off mid-thought ("I want to book a flight to"), each after a short agent question. Every turn is rendered three ways:

- `punctuated`: cased and punctuated, incomplete turns have no final punctuation
- `punctuated+period`: same, but incomplete turns end with a period, like an STT that closes every final segment
- `plain`: lowercase, no punctuation

`bench.py` calls the plugin's own inference runners (`_EUORunnerEn`, `_EUORunnerMultilingual`), so formatting, truncation and the ONNX session match an agent. A turn counts as "held" when its probability is below the model's English `unlikely_threshold` from `languages.json` (the point where the agent waits longer).

## Results (livekit-plugins-turn-detector 1.8.10, Apple Silicon CPU)

| model | format | complete turns recognized | incomplete turns held | median p (complete) | median p (incomplete) |
|---|---|---|---|---|---|
| en (threshold 0.0289) | punctuated | 100% | 100% | 0.897 | 0.0003 |
| en | punctuated+period | 100% | **0%** | 0.897 | 0.758 |
| en | plain | **75%** | 95% | 0.061 | 0.0007 |
| multilingual (threshold 0.011) | all three | 100% | 92.5% | 0.612 | 0.0005 |

Latency per call: en p50 1.4 ms, multilingual p50 9.8 ms.

What I take from it:

- The English model leans heavily on the final punctuation mark. A trailing period on a cut-off sentence ("I want to book a flight to.") moved every one of the 40 incomplete turns above the threshold, so the agent would reply early every time.
- With no punctuation at all, 10 of 40 complete answers fell below the threshold (for example "yes i'm allergic to peanuts" scored 0.013), so the agent would wait the longer delay on finished turns.
- The multilingual model gives identical scores across all three renderings, because of its normalization step. Its 3 misses are two partial numbers ("It's 482", "It's A B 7 7") and "Window, unless the only ones left are".

Every per-turn score is in `per_item.json`; the summary is in `results.json` and `run_output.txt`.

## Limits

- 40 pairs I wrote myself, English only. This is a sensitivity check, not an accuracy benchmark.
- `punctuated+period` is a stress case. I did not measure how often a given STT provider actually ends incomplete segments with a period.
- The plugin README marks these text models as deprecated in favor of the audio `livekit.agents.inference.TurnDetector`. I couldn't test that one here (no speech synthesis in my environment), so this mainly matters for agents still on the text models.

## Run it

```
python3 -m venv env && ./env/bin/pip install -r requirements.txt
./env/bin/python bench.py
```

The first run downloads the model files from Hugging Face (`livekit/turn-detector`).
