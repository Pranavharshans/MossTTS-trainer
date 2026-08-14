# Malayalam fine-tuning on FAU Alex

## Chosen approach

This workflow directly fine-tunes MOSS-TTS-Nano with an ID-preserving
Malayalam SentencePiece extension. It does not run a competing byte-tokenizer
training arm.

The original tokenizer can represent Malayalam through UTF-8 byte fallback,
but ordinary Malayalam sentences often consume many more text positions than
English. That has four practical effects:

- fewer text/audio examples fit a fixed packed sequence;
- prompt processing and first-audio latency increase;
- gradients are spread across byte fragments instead of useful Malayalam
  character and word patterns;
- learning pronunciation and grapheme-to-sound mappings requires more data and
  steps.

Transformer attention cost grows approximately with the square of sequence
length. The end-to-end penalty is smaller than squaring the text-only ratio
because target audio frames also occupy the sequence, but byte fallback is
still an avoidable source of memory use, dropped examples, and slower
convergence.

The extension appends 2,048 Malayalam pieces after the complete 16,384-token
base vocabulary. IDs `0..16383` and all non-Malayalam tokenization stay
unchanged. Each new embedding row is initialized as the mean of that piece's
original byte-token embeddings, the tied text output head is resized, and both
vocabulary-size fields in the model configuration are updated. Byte fallback
remains available for unseen text.

This changes checkpoint compatibility: every Malayalam checkpoint must travel
with its extended `tokenizer.model`. It adds about 1.57 million tied embedding
parameters (`2048 x 768`), but avoids invalidating the pretrained rows.

## Fixed inputs

- Training manifest: `/anvme/workspace/v123be62-voxcpm-ml/voxcpm-runtime/datasets/rasa-malayalam/train.jsonl`
- Validation manifest: `/anvme/workspace/v123be62-voxcpm-ml/voxcpm-runtime/datasets/rasa-malayalam/val.jsonl`
- Evaluation reference: `eval_prompt.flac` and `eval_prompt.txt` in the same directory
- Base model revision: `44502f80dbf9743528fa921cc544d662c685ebec`
- Codec revision: `6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`
- Default runtime root: `/anvme/workspace/v123be62-voxcpm-ml/moss-tts-runtime`

The existing audio is 16 kHz mono FLAC. Preparation decodes it and feeds it to
the model's 48 kHz codec, which performs the required resampling/channel
conversion. Upsampling does not recreate frequencies already removed by the
16 kHz source, so output fidelity remains bounded by the source recordings.

## Submit the complete Slurm pipeline

On an Alex login node, clone or update this repository and run from its root:

```bash
bash scripts/slurm/submit_malayalam_alex.sh
```

Explicit paths can be supplied when needed:

```bash
bash scripts/slurm/submit_malayalam_alex.sh \
  /absolute/path/to/MossTTS-trainer \
  /anvme/workspace/v123be62-voxcpm-ml/moss-tts-runtime \
  /anvme/workspace/v123be62-voxcpm-ml/voxcpm-runtime/datasets/rasa-malayalam
```

The submission script creates an `afterok` dependency chain:

1. `setup`: creates the persistent Conda environment, downloads pinned model
   snapshots, trains the Malayalam tokenizer extension, and initializes the
   resized model.
2. `prepare`: encodes target audio with the MOSS codec and performs exact
   packed-length filtering.
3. `train`: full-model BF16 SFT for three epochs on one A40, with validation
   loss and permanent epoch checkpoints.
4. `eval`: generates the seven-prompt matched audio suite for the initialized
   extended model and every epoch checkpoint.

The command prints all four job IDs and exact `squeue`, `sacct`, and `tail`
monitoring commands. Logs are stored under
`/anvme/workspace/v123be62-voxcpm-ml/moss-tts-runtime/logs` by default.

## Resume behavior and outputs

Audio preparation is divided into 32 deterministic training shards. Each
accepted manifest, rejected manifest, and completion report is written
atomically. Re-running a failed preparation job skips every shard whose input
fingerprint and settings still match.

Training writes model, tokenizer, optimizer, scheduler, and random-number state
at every completed epoch. `--resume-from-checkpoint auto` restores the newest
complete epoch after requeue or resubmission. It intentionally resumes only at
an epoch boundary; a partially completed epoch is repeated rather than guessed.

Important paths below the runtime root:

- `tokenizer/report.json`: full-corpus before/after token-length statistics
- `models/MOSS-TTS-Nano-Malayalam`: initialized extended checkpoint
- `prepared/train.rank*-of-00032.jsonl`: accepted training shards
- `prepared/*.rejected.jsonl`: excluded samples with reasons
- `prepared/*.report.json`: resumability and packed-length reports
- `runs/malayalam-sft/checkpoint-epoch-*`: permanent epoch checkpoints
- `runs/malayalam-sft/metrics.jsonl`: validation loss per epoch
- `runs/malayalam-sft/evaluation`: generated WAVs and duration/timing summary

Silent truncation is disabled in both preparation and training. If a packed
sample exceeds 1,024 positions, preparation records it as rejected; if an
unfiltered overlength row reaches training, training fails loudly.

## Training defaults

- Full-model SFT, not LoRA
- One A100, BF16, SDPA attention
- Per-device batch 4, gradient accumulation 4, effective batch 16
- Learning rate `1e-5`, 3% warmup, weight decay `0.1`
- Three epochs, validation after each epoch
- Text/audio channel weighting `1,32`

These are safe starting settings for this 100M-parameter model. The generated
evaluation suite should determine which epoch is deployed; `checkpoint-last`
is a convenience alias, not an automatic quality decision.

## Acceptance checks

Listen to the same prompt set for each epoch and inspect
`evaluation_summary.json`.

1. Malayalam must be intelligible without empty, clipped, repeated, or runaway
   output.
2. Pronunciation and rhythm should improve across most Malayalam prompts.
3. Speaker identity should remain stable with the fixed reference clip.
4. Durations should remain plausible and not jump by more than about 15%
   between adjacent checkpoints without an audible reason.
5. The English retention prompt should remain intelligible.

Validation loss is useful for detecting divergence, but it does not select the
best TTS checkpoint by itself. Audio quality and duration are the final gates.
