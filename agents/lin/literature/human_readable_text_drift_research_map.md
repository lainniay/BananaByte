# Human-readable text drift in generative image restoration

Updated: 2026-08-25

## 1. Provisional research question

The current observation should be treated as a hypothesis, not an established phenomenon:

> Under degradations for which humans can still transcribe scene text with measurable accuracy, a generative restoration or super-resolution model sometimes produces a sharp but semantically different character or word.

Working name: **human-readable text drift**. This name is deliberately provisional. The first contribution may be an operational definition and prevalence study rather than an explanation of an already-established failure mode.

The central comparison is not simply LR versus SR image quality. It is:

- human transcription and confidence on the degraded input;
- model output text identity and visual fidelity;
- whether the output remains supported by the input evidence;
- when and how a plausible lexical prior replaces the observed glyph evidence.

## 2. What the current literature already supports

Direct evidence is emerging but fragmented.

- Undermind's deep search returned 53 candidates and 41 available full texts. Its highest-ranked papers were NCAP, a low-resolution text extraction framework, C3-STISR, Hallucination Score, PRISM, and TextSR. The search summary explicitly classified the exact human-readable/model-wrong mismatch as unmeasured rather than established.
- NCAP identifies overconfidence in text-prior modalities as a generalization problem, making it a direct prior-dominance reference.
- PRISM states that degraded-input text conditions can themselves be unreliable and introduces uncertainty-aware stroke-boundary modeling. It is currently one of the closest mechanistic-method papers to the proposed question, although it does not perform the required human-versus-model threshold experiment.
- TextSR reports that general diffusion SR has difficulty localizing text and modeling character shapes, producing inconsistent or hallucinated textures. It also treats OCR errors as a remaining bottleneck.
- GLYPH-SR argues that ordinary perceptual metrics are insensitive to character-level errors and that scene text is often handled as generic texture.
- TIGER explicitly describes a conflict between generative image quality and textual readability and separates glyph restoration from later image enhancement.
- C3-STISR establishes an older and important failure mechanism: a recognition prior extracted from degraded text can itself be wrong and then mislead the SR module.
- Hallucination Score and HalluGen show that hallucination in restoration is not captured reliably by PSNR, SSIM, LPIPS, or no-reference quality alone. HalluGen further distinguishes measurement-inconsistent intrinsic hallucinations from measurement-consistent but inverse-ambiguous extrinsic hallucinations.
- Trustworthy SR uses digits and letters as examples where multiple feasible diffusion outputs are not sufficient and human feedback is needed to select a trustworthy reconstruction.
- Evaluating and Preserving High-level Fidelity and the 2026 generative-restoration study frame the broader transition from missing detail to uncontrolled or semantically incorrect generated detail.
- Human reading studies show that letter identification is interactive rather than purely bottom-up. Real-word context, semantic priming, and word-level expectations can improve identification of weak letters. This makes “a human can still read it” a psychophysical statement that must be measured, not inferred from an OCR score.
- Uncertainty-aware OCR work shows a related asymmetry: VLMs can emit fluent but incorrect text from lossy images without indicating uncertainty. This suggests that forced commitment is itself a testable mechanism.

What is **not** yet established is the prevalence of the exact human-readable/model-wrong mismatch across restoration models, languages, fonts, degradations, and scene contexts.

## 3. Why a human may read the text while a generative model changes it

These are competing, testable hypotheses rather than a single explanation.

1. **Different use of context.** Humans combine partial strokes, lexical knowledge, semantics, and scene context while retaining uncertainty. A generative model must commit to a specific pixel realization.
2. **Information bottleneck.** Downsampling, patchification, a VAE, or an image encoder may collapse distinctions between similar glyphs before generation begins. Recent DiT-SR work directly identifies VAE compression as weakening fine spatial evidence.
3. **Prior dominance.** Sampling and guidance can favor a frequent or contextually plausible word over a rare name, nonword, or visually supported alternative.
4. **Objective mismatch.** Perceptual and generative objectives reward realism and texture plausibility, not exact discrete character identity.
5. **Recognizer-prior contamination.** An OCR or VLM prior may make the first semantic error and then render that error sharply.
6. **Uncertainty collapse.** The system has no explicit abstention or alternative-hypothesis channel, so ambiguity becomes a confident-looking output.

## 4. Research routes that do not require model internals

### 4.1 Human-model psychophysics

Construct controlled minimal pairs that differ by one glyph, for example common word versus rare word, word versus nonword, and proper name versus frequent lexical neighbor. Apply blur, downsampling, JPEG compression, noise, and occlusion in calibrated sweeps.

For each stimulus, collect:

- human transcription accuracy and confidence;
- human character-level confusion distribution;
- model exact match, CER, and WER;
- unsupported insertions, deletions, and substitutions;
- output stability across seeds and sampling settings.

The critical region is where human accuracy remains above a preregistered threshold but the generative output changes identity.

### 4.2 Black-box causal probes

The following can be run through APIs before the GPU server arrives:

- full scene versus text crop;
- intact context versus masked context;
- real word versus matched nonword;
- common word versus rare word or proper name;
- English/Chinese/mixed-script conditions;
- prompt absent versus restoration-only prompt versus explicit transcription constraint;
- fixed input across multiple seeds, guidance values, and edit strengths;
- LR input alone versus LR plus an independently supplied candidate transcription.

These interventions localize whether the failure depends on scene semantics, lexical frequency, language, prompt prior, or sampling instability without inspecting weights.

### 4.3 Pipeline localization without full circuit analysis

Once an open model runs locally, test progressively:

1. interpolation-only control;
2. VAE encode-decode reconstruction without denoising;
3. conditioning/image-encoder reconstruction;
4. early, middle, and late denoising checkpoints;
5. decoder-only variants where possible.

The earliest stage at which the correct glyph becomes unrecoverable sharply narrows the causal target.

### 4.4 External mitigation

- split text and non-text regions, using conservative or specialist SR for text;
- generate multiple candidates and rerank using independent OCR ensembles plus measurement consistency;
- permit abstention or return several candidate strings when evidence is insufficient;
- use OCR only as a proposal and verify it against glyph evidence rather than treating it as ground truth;
- evaluate text identity separately from global image quality.

## 5. If studying model internals: one model or cross-model commonality?

The correct design is **one-model mechanism plus cross-model behavioral generalization**.

Low-level circuits are architecture-specific. A U-Net latent diffusion model, MM-DiT, autoregressive image model, and unified multimodal editor do not expose the same modules. Even DiTs with similar task accuracy can implement different circuits when the text encoder changes. Therefore, a claim such as “head 7 in layer 18 causes the error” should remain model-specific.

Cross-model hypotheses can be expressed at shared functional interfaces:

- loss of glyph evidence at the input representation bottleneck;
- increasing dominance of lexical or semantic priors over measurement evidence;
- absence of an exact discrete-character constraint;
- forced commitment under ambiguity;
- amplification of an early error during iterative sampling or decoding.

Recommended scope:

- select one open-weight anchor model with accessible encoder/VAE, denoising trajectory, and reproducible inference;
- perform VAE controls, activation/attention tracing, patching, and causal interventions on that model;
- validate the same behavioral signatures on two or three architecturally distinct models, including a closed API model if useful;
- claim a shared mechanism only when the intervention is defined at a shared functional level and supported across models.

Candidate internal tools include cross-attention attribution (DAAM), norm-based causal prompt interventions, OV-subspace interventions, DiT concept maps, and transcoder-based circuit tracing. These methods are starting points; none directly proves the text-restoration mechanism without new experiments.

## 6. Recommended staged study

### Stage A — define and verify the phenomenon now

1. Build 200-500 controlled stimuli with one-character minimal pairs and known HR references.
2. Generate multiple degradation levels and estimate human readability curves.
3. Run at least two generative systems with repeated seeds.
4. Predefine a failure as human-correct/model-wrong under a fixed human-accuracy threshold.
5. Model failure probability as a function of glyph distance, degradation, word frequency, context, language, model, and seed.

Stage A can produce a valid result even if the effect is rare or disappears after controls.

### Stage B — black-box causal characterization

Use factorial context, lexicality, prompt, and sampling interventions. Determine whether errors are systematic lexical substitutions, random instability, scene-conditioned substitutions, or simple failure to improve the input.

### Stage C — white-box localization on one open model

1. Test whether VAE reconstruction alone changes or erases the decisive strokes.
2. Decode intermediate denoising states and identify the transition timestep.
3. Trace representations of text regions and candidate characters/words.
4. Patch correct-condition activations into failing runs and test causal rescue.
5. Intervene on guidance, text conditioning, and image-evidence pathways separately.

### Stage D — cross-model validation

Repeat only the compact behavioral signature set, not full circuit tracing, on other architectures. Report both shared effects and architecture-specific deviations.

## 7. Priority reading order

1. TextSR; GLYPH-SR; TIGER; C3-STISR.
2. Hallucination Score; HalluGen; When Latents Forget Pixels.
3. Siddiqui et al. human psychophysics; Evans et al.; Heilbron et al.
4. Teaching VLMs to Admit Uncertainty in OCR from Lossy Visual Inputs.
5. DAAM; ConceptAttention; cross-attention interventions; cross-attention OV subspaces; DifFRACT.
6. Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers, because it directly warns that similar performance can arise from different internal circuits.

The accompanying BibTeX file contains a 23-item verified shortlist distilled from the 53 Undermind candidates plus primary-source checks. The Undermind deep search titled **Human Readable Text Drift in Generative Restoration** completed on 2026-08-25 with 53 papers and 41 full texts. The shortlist is intentionally smaller than the search result set so that Zotero does not become a dump of only tangentially related papers.
