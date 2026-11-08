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
| MMS-300m forced aligner, 4-bit (`model_q4.onnx`) | Windows ARM64; the Mac's fallback | **In use** |
| MMS-300m forced aligner, fp16 | Windows x64; the trials' reference | In use there |
| MMS-300m, q4f16 | — | No better than q4, slower |
| MMS-300m, int8 builds | — | Do not load on ARM64 |
| Whisper's cross-attention (DTW) word times | Mac | **In use as the first look** |
| MMS-300m fp16 on the GPU through Core ML | Mac | **In use** since 2026-11-07; the 4-bit CPU scan is its fallback |
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

**MMS fp16 on the GPU** (ONNX Runtime 1.29's Core ML provider, MLProgram format, 34 s windows at a
fixed shape; `localscribe-doctor --aligner-coreml <scrb> --units CPUAndGPU|ALL`). Core ML takes
1,385 of the graph's 1,391 nodes; the other six are the first feature-extractor convolutions, whose
input is longer than Core ML's 16,384 limit, and they run on the CPU. As exported, the model took 2–3
minutes to open every time, cache or not: Core ML transposes each MatMul weight for its `linear` op
and writes the result into the compiled program as text, 3.7 GB of it. Rewriting the 146 MatMuls
that have a weight as Reshape, Gemm with `transB=1` on the weight pre-transposed, Reshape (the same
arithmetic) keeps the weights in binary: open 6 s the first time, 0.6 s after, a 1.2 GB cache
instead of 4.1 GB. The app does this rewrite itself (`GemmRewrite`, `GpuAligner`), once, in the
background: 12 s on the M2, then 1.8 GB on disk for the rewritten model and Core ML's cache.

Measured with the machine far from idle (Defender and Spotlight, load averages from 9 to 100), so
compare within a row, where the runs were back to back:

| Recording | 4-bit on the CPU (the app) | fp16 on the GPU | Within 0.1 s of the reference, CPU / GPU |
|---|---|---|---|
| Debate, 1:56 | 47.1 s | 14.0 s | 92% / 97% |
| Podcast, 7:23 | 167.7 s | 47.0 s | 99% / 100% |

Quieter, earlier, on the debate: 24.8 s against 7.9 s. The GPU scan still kept about two cores busy
(94.6 s of CPU time over the podcast's 47 s), for the six CPU nodes and the windowing. Through
`ALL`, which adds the Neural Engine, it was no faster (52.0 s on the podcast), took 322 s to compile
the first time, and would contend with Whisper's encoder, which lives there.

The app's own path (`--aligner-coreml <scrb> --app`), the machine quieter: podcast scanned in
21.8 s with 52 s of CPU time, every one of 1,178 words within 0.1 s of the reference.

Whole runs on the podcast at Balanced (`LOCALSCRIBE_PACE=balanced`, headless), two each,
alternating:

| Order | Done | Labels | Peak memory |
|---|---|---|---|
| CPU 4-bit scan after transcription (before) | 3:42, 4:15 | 48 s, 42 s | 2.5 GB |
| GPU scan after transcription (adopted) | 1:35, 1:52 | 26 s, 41 s | 2.5 GB |
| GPU scan alongside transcription | 2:02, 1:47 | 118 s, 103 s | 2.9 GB |

Alongside, the scan slowed Whisper (raw transcript 1:44–1:53 against 0:58–1:10) and the speaker
model with it, so the Mac keeps scanning after transcription. Recordings shorter than one 34 s
window are scanned on the CPU.

## Speakers

| Model | Where | Verdict |
|---|---|---|
| NVIDIA Nemotron 3 Diarization (Sortformer), float32 | Mac | **In use** since 2026-11-03 |
| NVIDIA Nemotron 3 Diarization (Sortformer), int8 | Snapdragon | **In use**; invents speakers on the Mac |
| pyannote segmentation + WeSpeaker embeddings (sherpa-onnx) | x64 | In use; tuning frozen |

**Nemotron 3** is one end-to-end model, at most eight speakers, on the CPU. On the Mac it
reproduced Windows: debate 358 of 388 words with the same speaker as on the Snapdragon, podcast
1167 of 1178 (99.1%), about 46× real time on its own. Its thread knee is 4; the aligner's is 6.
Found at load on both machines now — labels 11–18 s into transcription on the Mac, against
92–120 s when it waited for transcription to end. See [diarization.md](diarization.md) for the
pyannote pipeline, whose tuning is frozen and judged only by `--diarize` turn diffs.

**The 8-bit build's answer depends on the machine.** Run 2026-11-03 on the M2, each build alone,
turns compared every 0.1 s (`--diarize-trial --only nemotron --turns`, with `--float32` for full
precision; thread count changed nothing, 0.00% between 2 and 5 threads):

| Recording | Speakers, int8 / float32 | Differ on | Took at 2 threads, int8 / float32 |
|---|---|---|---|
| Debate, 1:56 | **3** / 2 | 6.8% | 2.3 s / 4.3 s |
| Podcast Aug 10, 7:23 | 5 / 5 | 1.4% | 8.4 s / 17.0 s |
| Podcast Aug 17, 7:36 | 5 / 5 | 0.3% | 9.0 s / 19.3 s |
| Karl (phone), 21:02 | **7** / 6 | 4.3% | 41 s / 121 s |
| Zero Trust panel, 61:42 | 3 / 3 | 0.2% | 120 s / 286 s |

The debate has two people; int8's third is a 4.8 s phantom taken from one of them, and it reaches
the finished transcript. The Snapdragon's int8 found 2 on the debate and 6 on Karl, the counts
float32 finds here, so the difference is in how int8 executes on each machine (ONNX Runtime 1.22
there, 1.29 here; the cause has not been traced). On the debate and the podcast float32 is also
far closer to the Snapdragon's turns than the Mac's int8 is (0.9% against 5.9%, 0.2% against
1.4%); on Karl they tie. It is not better everywhere: at 3:59 on the podcast int8 labels 3.5 s of
voice-over that float32 leaves empty. On the two synthetic clips with true labels
(`samples/two-speakers.json`, `rapid.json`) both builds get every turn right. Peak memory is about
300 MB higher with float32 (roughly 750 MB against 450 MB), and the file is 396 MB against 103 MB.
Adopted on the Mac on 2026-11-03 (`SortformerModelSource.PreferFullPrecision`). In the app the
debate then finishes with two speakers. The Snapdragon stays on int8, whose counts were right there.

## Cleanup

| Model | Where | Verdict |
|---|---|---|
| Qwen 2.5 7B Instruct, through Foundry Local | either | **Removed** 2026-11-03 |
| Qwen 2.5 1.5B | — | Not tried; made moot |

**Qwen 7B** was what Foundry already served (StyleHelper installed it). It holds about 7.2 GB,
which beside the scan pushed a 16 GB Mac into swap until every word fell back to an estimate;
and it takes roughly half the recording's length on the M2. Measured on the podcast it changed
about one token in eleven, nearly all punctuation Whisper already had right. Off by default on
the Mac first, then on Windows (podcast, done at about 150 s without it and 236 s with it). The
1.5B was the candidate for keeping cleanup on cheaply; making cleanup optional removed the need.
After a stretch off by default with nobody missing it, the stage was removed, along with Foundry
Local, GenieX and the glossary, which only ever worked through it. The last commit with it is
`454d6ef`.
