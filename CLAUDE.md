# LocalScribe

Offline transcription for Snapdragon X Windows laptops. Whisper on the Hexagon NPU, a local
language model for cleanup, nothing leaving the machine.

**Read [docs/handoff.md](docs/handoff.md) first.** It says what has been verified, what has never
run, and what is most likely to be broken. This file is the short version.

## The state that matters

The core library and its tests are verified. `LocalScribe.App` has never been compiled, and
`LocalScribe.Onnx` has compiled but never executed a model — both were written in a Linux cloud
container with no Windows toolchain and no NPU.

## Build and test

```powershell
dotnet build -c Release -r win-arm64      # arm64 is required, see below
dotnet test                                # core only; no hardware needed
dotnet run --project src/LocalScribe.Doctor -c Release -r win-arm64
```

`LocalScribe.App` is not in `LocalScribe.sln`, because WinUI cannot restore on Linux and its
presence would break `dotnet build` for non-Windows contributors. Build it by path.

Everything else is in the solution, including `LocalScribe.Diarization` — sherpa-onnx restores
and compiles on any platform; only running it needs the native library.

## Invariants

Do not change these without reading the reasoning in `docs/handoff.md`:

- **Build native arm64.** Under x64 emulation the QNN provider cannot load, and the symptoms are
  identical to a missing driver.
- **`LocalScribe.Core` takes no external dependencies.** That is what makes the policy and the
  audio maths testable on any machine.
- **The decoder stays on the CPU** unless the NPU took the encoder. Dispatch overhead per decode
  step outweighs the compute saved.
- **Never hard-code the Foundry Local port.** It is dynamic; ask `foundry service status`.
- **Never rename downloaded model files.** Large ONNX graphs reference their weight sidecars by
  name. `localscribe-model.json` records the roles instead.
- **Never auto-install the Hexagon driver.** Signed kernel driver, account wall. Report it.
- **Diarization runs on the CPU, not the NPU.** The weights are pyannote's; the runtime is
  sherpa-onnx's own ONNX Runtime, which has no QNN provider. There is no "pyannote-NPU". See
  `docs/diarization.md`.
- **CPU threads are capped on purpose**, so the rest of Windows stays responsive. That is the
  product requirement, not a limitation to optimise away.

## Style

Comments explain why, not what. The existing code comments the non-obvious decisions — dispatch
overhead, reflect padding, silent fallback — and leaves the mechanical parts to speak for
themselves. Match that.

Tests check decisions and first principles rather than snapshots of current output. See the note
at the end of `docs/handoff.md` for why that matters here specifically.
