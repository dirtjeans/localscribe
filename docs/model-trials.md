# Models tried, and what decided each one

Every model this project has measured for a stage of the pipeline, what the measurement said,
and how to repeat it. Kept so the next "is there a faster X?" starts from numbers rather than
from the same research again. Dates are when the measurement ran; machines are the M2 MacBook
(16 GB) and the Snapdragon X Elite laptop (12 cores, 64 GB). The fixtures are the podcast
(`MM Aug 10_v1-new.scrb`, 443 s, five speakers, studio audio) and the debate
(`Peterson debate.scrb`, 116 s, two speakers, constant crosstalk).

## How these were measured — read before adding a row

- **At Balanced, with the machine otherwise idle.** The Light pace halves the threads and leaves
  placement to the OS; the same build swung from 3:40 to 7:06 across runs there. A LocalScribe
  window transcribing in the background during a trial doubled both its times and the trial's.
  Run each candidate twice, alternating, and report both.
- **Accuracy without ground truth is agreement, and agreement is not correctness.** No fixture
  has a hand transcript. Text candidates are compared with Whisper and the disagreements printed
  for a person to listen to; timing candidates are graded against the MMS fp16 aligner, which
  `--check-words` verified against the audio (drift +0.00 in every fifth of both fixtures).
- **"Measured N of N" is not sync.** A run can place every word and still place them wrong;
  `--check-words` on a saved archive is what says whether the words land on the voice.

The instruments: `localscribe-doctor --aligner-trial <scrb>` (aligners and Whisper's word times),
`--asr-trial <scrb>` (a transcriber against Whisper), `--check-words <scrb>` (sync against the
audio), and `LocalScribe.Desktop --headless <file> [--save out.scrb]` for whole-pipeline timing,
with the stage log in the temp folder (`localscribe-stages.txt`).

## Transcription

| Model | Where | Verdict |
|---|---|---|
| Whisper large-v3-turbo, whisper.cpp + Core ML | Mac | **In use** |
| Whisper large-v3-turbo, QNN export | Snapdragon NPU | **In use** |
| Nemotron 3.5 ASR streaming 0.6B, int4 ONNX | Mac CPU | Rejected 2026-10-05 |

**Whisper on the Mac.** Encoder on the Neural Engine, decoder on the CPU. As the app runs it, at
five threads: podcast 55.8 s (7.9× real time), debate 15.5 s (7.5×). On the Snapdragon the QNN
export runs encoder and decoder both on the Hexagon NPU, leaving the CPU to the other stages.

**Nemotron 3.5 ASR** ([onnx-community int4](https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4),
cache-aware FastConformer-RNNT, 0.56 s chunks, 128-band NeMo features, decoded in the doctor
after onnxruntime-genai's reference implementation). Rejected on every count that mattered:

- **Slower.** Podcast 126.7 s (3.5×) against Whisper's 55.8 s; debate 26.7 s against 15.5 s. Of
  the podcast's time, 87 s was the encoder, which runs on the CPU in small chunks and alone
  outlasts all of Whisper. The decoding (35 s) could be cut to a few seconds by batching the
  joint network across blank frames; that would not change the verdict.
- **Weaker on names.** "Raghu Nandakumara" came out three different wrong ways, "Meta's
  engineers" as "medicineers", "Fung T Nguyen" as "funk two". Most other podcast differences
  were formatting ("vendor neutral", numbers spelled out). The debate lost the second speaker's
  short interjections under crosstalk and a few words mid-sentence; part of the 79-word gap
  there was Whisper writing two crosstalk passages twice.
- **Its word times are no better than Whisper's own.** A steady lag of +0.22 s (podcast) and
  +0.28 s (debate), no drift; with the lag removed, 68% of podcast words within 0.1 s of the
  aligner and 98% within 0.25 s. Whisper's heard words, free, scored 71% and 95%. So pairing it
  with the Nemotron speaker model would not have let the scan go.
- The 4-bit build is itself less accurate than NVIDIA's original by its own published table
  (Spanish 7.28% WER against 4.11%, French 12.55% against 9.03%); the table has no English row.
  It has no translation and covers 40 languages to Whisper's ~99.

Not tried, and what would reopen the question: the [Core ML INT8 build](https://huggingface.co/aoiandroid/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8)
might run on the Neural Engine and change the speed picture, but carries the same model's
accuracy. NVIDIA's [multitalker variant](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
transcribes each Sortformer speaker separately, overlap included — the debate's weak spot —
at one ASR instance per speaker.

## Word timing

| Model | Where | Verdict |
|---|---|---|
| MMS-300m forced aligner, 4-bit (`model_q4.onnx`) | Mac; Windows ARM64 | **In use** |
| MMS-300m forced aligner, fp16 | Windows x64; the trials' reference | In use there |
| MMS-300m, q4f16 | — | No better than q4, slower |
| MMS-300m, int8 builds | — | Do not load on ARM64 |
| Whisper's cross-attention (DTW) word times | Mac | **In use as the first look** |
| Nemotron 3.5 ASR's own token times | — | Rejected with the model (above) |

**MMS 4-bit** quantises only the MatMuls (MatMulNBits, which has ARM64 kernels). On the Mac,
against fp16: podcast 99% of words within 0.1 s and all within 0.25 s, debate 92% and 95%, no
drift. Alone it was 19–30% faster (podcast 164.8 s against 234.0 s); in the whole pipeline at
Balanced, start to finished scan, fp16 took 4:26 and 4:09 and q4 3:19 and 2:37. It holds about
1.6 GB while scanning against 3.0 GB, which is what mattered on 16 GB. Graded against the audio,
end to end on the podcast: 115 of 115 segments measured, drift +0.00 in every fifth. On the
Snapdragon: podcast 99% and 100%, debate 89% and 94%, in 62.5 s against 119.8 s and 37.4 s
against 69.9 s, with `--check-words` leaving the same words adrift as fp16.

**q4f16** matched q4's accuracy exactly and was slower on both fixtures (204.4 s and 45.6 s
against 164.8 s and 40.3 s). **The int8 builds** fail at load: their ConvInteger has no ARM64
kernel in ONNX Runtime.

**Whisper's word times** (whisper.cpp DTW with the large-v3-turbo alignment heads, read per
30-second window, since handed a whole file it stopped at 55 s of 116). A constant lag of +0.12
to +0.18 s with no drift — each window is timed afresh, so error cannot accumulate — and, with
the lag removed, median 0.06 s, 67–72% within 0.1 s, 87–93% within 0.25 s. Not good enough to
replace the scan (one word in ten more than a quarter-second off), good enough to click on, so
the Mac uses them, less 0.15 s, until the scan replaces them. Cost: transcription 68–70 s
against 60 s on the podcast, because the timed decode ends segments differently and the chunker
takes 18 windows instead of 16. Not available on the Snapdragon: the QNN decoder exports no
cross-attention weights.

Not tried: the aligner on the GPU through ONNX Runtime's Core ML provider (fp16 only — Core ML
cannot take MatMulNBits), which would let the scan run beside the CPU stages instead of
competing with them.

## Speakers

| Model | Where | Verdict |
|---|---|---|
| NVIDIA Nemotron 3 Diarization (Sortformer), int8 | ARM64, both machines | **In use** |
| pyannote segmentation + WeSpeaker embeddings (sherpa-onnx) | x64 | In use; tuning frozen |

**Nemotron 3** is one end-to-end model, at most eight speakers, on the CPU. On the Mac it
reproduced Windows: debate 358 of 388 words with the same speaker as on the Snapdragon, podcast
1167 of 1178 (99.1%), about 46× real time on its own. Its thread knee is 4; the aligner's is 6.
Found at load on both machines now — labels 11–18 s into transcription on the Mac, against
92–120 s when it waited for transcription to end. See [diarization.md](diarization.md) for the
pyannote pipeline, whose tuning is frozen and judged only by `--diarize` turn diffs.

## Cleanup

| Model | Where | Verdict |
|---|---|---|
| Qwen 2.5 7B Instruct, through Foundry Local | either | Optional, **off by default** |
| Qwen 2.5 1.5B | — | Not tried; made moot |

**Qwen 7B** was what Foundry already served (StyleHelper installed it). It holds about 7.2 GB,
which beside the scan pushed a 16 GB Mac into swap until every word fell back to an estimate;
and it takes roughly half the recording's length on the M2. Measured on the podcast it changed
about one token in eleven, nearly all punctuation Whisper already had right. Off by default on
the Mac first, then on Windows (podcast, done at about 150 s without it and 236 s with it). The
1.5B was the candidate for keeping cleanup on cheaply; making cleanup optional removed the need.
