# Malayalam SFT run results — 2026-08-15

## Outcome

**Rejected. Do not deploy or resume from any checkpoint produced by this run.**

All generated Malayalam samples from epochs 1, 2, and 3 were judged to be
gibberish and unintelligible. The three checkpoints therefore fail the primary
acceptance criterion regardless of their validation losses.

## Run configuration

- Method: full-model supervised fine-tuning (SFT), not LoRA
- Dataset: Rasa Malayalam
- Training examples: 30,466
- Validation examples: 500
- Rejected during preparation: 0
- Epochs: 3
- Effective batch size: 16
- Learning rate: `1e-5` with 3% warmup and linear decay
- Precision: BF16
- Maximum packed length: 1,024
- Text/audio channel weighting: `1,32`
- Hardware: one NVIDIA A40
- Training runtime: 1 hour 36 minutes 45 seconds

## Validation results

| Epoch | Global step | Validation loss | Qualitative result |
| ---: | ---: | ---: | --- |
| 1 | 1,905 | 5.287093 | Rejected — gibberish |
| 2 | 3,810 | 5.282271 | Rejected — gibberish |
| 3 | 5,715 | 5.282065 | Rejected — gibberish |

The loss improved by only `0.005028` from epoch 1 to epoch 3, approximately
0.095%. This small numerical change did not correspond to intelligibility or
usable speech quality. Epoch 3 must not be selected merely because it has the
lowest validation loss.

## Evaluation artifacts and observed failure signals

The evaluation completed for the initialized extended base model and all three
epoch checkpoints. Seven fixed prompts were generated per checkpoint: six
Malayalam prompts and one English-retention prompt.

Generated duration was unstable between checkpoints for identical text. For
example, `ml_short_01` produced:

- epoch 1: 8.24 seconds
- epoch 2: 1.28 seconds
- epoch 3: 3.76 seconds

This large variation, together with universal unintelligibility, indicates
unstable generation/termination behavior that validation loss did not expose.

## Current diagnosis

The run proves that a finite, slowly decreasing teacher-forced loss is not a
sufficient quality gate for this workflow. The failure has not yet been
isolated to one confirmed root cause. The leading hypotheses are:

1. Training used plain text/audio pairs without reference-audio codes, while
   evaluation used voice-clone conditioning. This may be a training/inference
   contract mismatch.
2. Evaluation used stochastic sampling (`do_sample=1`); decoding settings may
   amplify an already weak or unstable model.
3. Prepared codec tokens may not reconstruct the source audio correctly, or
   source audio and transcripts may be misaligned.
4. Full-model Malayalam-only SFT may have damaged pretrained generation
   behavior before learning a reliable Malayalam text-to-audio mapping.
5. The extended tokenizer/model initialization may preserve IDs yet still be
   incompatible with useful pretrained generation and must be tested against
   the untouched official model.

These are hypotheses, not confirmed findings.

## Required gates before another full run

Do not launch another full training run until all of the following pass on a
small, representative diagnostic:

1. Decode prepared codec tokens for randomly selected records and confirm that
   they reproduce the corresponding source speech intelligibly.
2. Manually confirm transcript/audio alignment for a sampled subset.
3. Verify the untouched official model on supported-language and English
   prompts using the same inference stack.
4. Compare the initialized extended model with the untouched model on English
   retention before training.
5. Evaluate plain TTS and voice-clone modes separately, matching the inference
   prompt format to the training examples.
6. Compare greedy decoding (`do_sample=0`) with the sampling configuration.
7. Run only a 100–300-step pilot and require intelligible generated Malayalam
   before authorizing a complete run.
8. Select checkpoints by matched listening tests and intelligibility metrics,
   not validation loss alone.

## Keep/reject decision

- Epoch 1: reject
- Epoch 2: reject
- Epoch 3 / checkpoint-last: reject
- Training recipe: retain only as a reproducibility record; do not reuse
  unchanged
- Prepared data and tokenizer extension: quarantine for diagnostic validation
  before reuse

