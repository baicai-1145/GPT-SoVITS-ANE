from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: str
    description: str


@dataclass(frozen=True)
class ExportTargetSpec:
    name: str
    description: str
    implemented: bool
    coreml_ready: bool
    required_artifacts: List[str]
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    notes: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


TARGET_SPECS: Dict[str, ExportTargetSpec] = {
    "g2pw": ExportTargetSpec(
        name="g2pw",
        description="Rebuilt g2pw polyphonic disambiguation model exported through an ONNX->PyTorch bridge.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=[
            "GPT_SoVITS/text/G2PWModel/g2pW.onnx",
            "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json",
        ],
        inputs=[
            TensorSpec("input_ids", "int64", "[batch, token_len]", "Tokenizer ids including [CLS]/[SEP]."),
            TensorSpec("token_type_ids", "int64", "[batch, token_len]", "Token type ids."),
            TensorSpec("attention_mask", "int64", "[batch, token_len]", "Attention mask."),
            TensorSpec("phoneme_mask", "float32", "[batch, label_count]", "Per-query valid phoneme mask."),
            TensorSpec("char_ids", "int64", "[batch]", "Character vocabulary ids for queried polyphonic chars."),
            TensorSpec("position_ids", "int64", "[batch]", "Queried token positions inside the tokenized sentence."),
        ],
        outputs=[
            TensorSpec("probs", "float32", "[batch, label_count]", "Masked phoneme probability distribution."),
        ],
        notes=[
            "The bridge uses onnx2torch to reconstruct an exportable PyTorch graph from the original g2pW ONNX model.",
            "Export uses fixed batch/token capacities and expects host-side padding.",
            "Padding rows must duplicate a valid query row instead of using an all-zero phoneme mask, and callers should ignore outputs beyond the real query count.",
        ],
    ),
    "zh_bert_phone": ExportTargetSpec(
        name="zh_bert_phone",
        description="Chinese BERT wrapper that emits phone-level features matching the current frontend.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"],
        inputs=[
            TensorSpec("input_ids", "int64", "[1, text_tokens]", "Tokenizer ids including special tokens."),
            TensorSpec("attention_mask", "int64", "[1, text_tokens]", "Tokenizer attention mask."),
            TensorSpec("token_type_ids", "int64", "[1, text_tokens]", "Tokenizer token type ids."),
            TensorSpec("word2ph", "int32", "[text_chars]", "Character to phone repeat counts."),
        ],
        outputs=[
            TensorSpec("phone_level_feature", "float32", "[1024, phone_capacity]", "Phone-level BERT feature padded to a fixed phone capacity."),
        ],
        notes=[
            "This preserves the current phone-level feature layout used by TextPreprocessor.",
            "The Core ML path uses a fixed-capacity masked expansion instead of dynamic repeat_interleave.",
            "Export one model per intended phone capacity, e.g. 80 for prompt/reference or 120 for target text.",
        ],
    ),
    "zh_bert_char": ExportTargetSpec(
        name="zh_bert_char",
        description="Chinese BERT wrapper that emits char-level features before word2ph expansion.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"],
        inputs=[
            TensorSpec("input_ids", "int64", "[1, text_tokens]", "Tokenizer ids including special tokens."),
            TensorSpec("attention_mask", "int64", "[1, text_tokens]", "Tokenizer attention mask."),
            TensorSpec("token_type_ids", "int64", "[1, text_tokens]", "Tokenizer token type ids."),
        ],
        outputs=[
            TensorSpec("char_feature", "float32", "[text_chars, 1024]", "Char-level BERT feature before phone expansion."),
        ],
        notes=[
            "Tokenizer, text normalization and word2ph generation remain CPU-side.",
            "Phone-level repeat is expected to stay on CPU / Swift runtime.",
        ],
    ),
    "cnhubert_encoder": ExportTargetSpec(
        name="cnhubert_encoder",
        description="CN-HuBERT neural encoder fed by CPU-preprocessed input_values.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/chinese-hubert-base"],
        inputs=[
            TensorSpec("input_values", "float32", "[1, samples]", "CPU-preprocessed HuBERT input_values."),
        ],
        outputs=[
            TensorSpec("ssl_content", "float32", "[1, channels, frames]", "Transposed SSL content feature."),
        ],
        notes=[
            "Wav2Vec2FeatureExtractor preprocessing stays on CPU side and is not part of the exported Core ML graph.",
        ],
    ),
    "speaker_encoder": ExportTargetSpec(
        name="speaker_encoder",
        description="ERes2Net speaker embedding backbone fed by CPU-precomputed Kaldi fbank features.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"],
        inputs=[
            TensorSpec("fbank_80", "float32", "[1, frames, 80]", "CPU-precomputed 80-bin Kaldi fbank feature."),
        ],
        outputs=[
            TensorSpec("sv_emb", "float32", "[1, speaker_cond_dim]", "Speaker conditioning vector."),
        ],
        notes=[
            "Kaldi fbank extraction stays on CPU side because the current implementation relies on ops unsupported by Core ML conversion.",
            "The VITS bundle export can expose a dynamic frame-count input for fbank_80 so long reference audio is not truncated to the fixed example length used during conversion.",
        ],
    ),
    "ssl_latent_extractor": ExportTargetSpec(
        name="ssl_latent_extractor",
        description="Extract prompt semantic tokens from SSL content using the SoVITS quantizer path.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth or s2Gv2ProPlus.pth"],
        inputs=[
            TensorSpec("ssl_content", "float32", "[1, channels, frames]", "SSL content feature from CN-HuBERT."),
        ],
        outputs=[
            TensorSpec("prompt_semantic", "int64", "[1, prompt_frames]", "Prompt semantic token sequence."),
        ],
        notes=[
            "This is the vq_model.extract_latent path only, not the full decoder.",
        ],
    ),
    "vits_decode_condition": ExportTargetSpec(
        name="vits_decode_condition",
        description="Reference-condition encoder that maps a single reference spectrogram and speaker embedding to ge/ge_text.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth or s2Gv2ProPlus.pth"],
        inputs=[
            TensorSpec("refer", "float32", "[batch, spec_channels, refer_frames]", "Reference spectrogram for style conditioning."),
            TensorSpec("sv_emb", "float32", "[batch, speaker_cond_dim]", "Speaker conditioning vector from speaker_encoder."),
        ],
        outputs=[
            TensorSpec("ge", "float32", "[batch, gin_channels, 1]", "Waveform-conditioning style vector."),
            TensorSpec("ge_text", "float32", "[batch, ge_text_channels, 1]", "Text-prior conditioning style vector."),
        ],
        notes=[
            "This is the single-reference tensor path only; multi-reference averaging remains outside the graph.",
            "The Core ML bundle export supports dynamic refer frame counts so the runtime can keep the original Python spectrogram length instead of right-padding to a fixed frame capacity.",
            "Use this before vits_prior and the downstream VITS wave-generator targets to avoid rebuilding ge/ge_text on the host side.",
        ],
    ),
    "vits_prior": ExportTargetSpec(
        name="vits_prior",
        description="Deterministic SoVITS prior subgraph that maps semantic codes + text + ge_text to prior stats.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth or s2Gv2ProPlus.pth"],
        inputs=[
            TensorSpec("codes", "int64", "[1, batch, semantic_frames]", "Single-codebook semantic token tensor."),
            TensorSpec("text", "int64", "[batch, text_phone_count]", "Target phone id sequence."),
            TensorSpec("ge_text", "float32", "[batch, ge_text_channels, 1]", "Precomputed text-conditioning style vector."),
            TensorSpec("code_lengths", "int64", "[batch]", "Real semantic token counts before zero-padding."),
            TensorSpec("text_lengths", "int64", "[batch]", "Real text phone counts before zero-padding."),
        ],
        outputs=[
            TensorSpec("prior_mean", "float32", "[batch, latent_channels, latent_frames]", "Deterministic prior mean tensor."),
            TensorSpec("prior_log_scale", "float32", "[batch, latent_channels, latent_frames]", "Deterministic prior log-scale tensor."),
            TensorSpec("y_mask", "float32", "[batch, 1, latent_frames]", "Latent validity mask."),
        ],
        notes=[
            "Pair this with vits_decode_condition when you want the full deterministic VITS conditioning path on Core ML.",
            "Feed the outputs into vits_latent_sampler -> vits_flow -> vits_masked_wave_generator for the full split VITS chain.",
        ],
    ),
    "vits_latent_sampler": ExportTargetSpec(
        name="vits_latent_sampler",
        description="Latent sampling wrapper that maps prior stats + explicit standard-normal noise to sampled z_p.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth or s2Gv2ProPlus.pth"],
        inputs=[
            TensorSpec("prior_mean", "float32", "[batch, latent_channels, latent_frames]", "Prior mean tensor from vits_prior."),
            TensorSpec("prior_log_scale", "float32", "[batch, latent_channels, latent_frames]", "Prior log-scale tensor from vits_prior."),
            TensorSpec("noise", "float32", "[batch, latent_channels, latent_frames]", "Explicit standard-normal noise tensor supplied by runtime."),
            TensorSpec("noise_scale", "float32", "[1]", "Runtime noise scale scalar."),
        ],
        outputs=[
            TensorSpec("z_p", "float32", "[batch, latent_channels, latent_frames]", "Sampled prior latent tensor before flow reverse."),
        ],
        notes=[
            "This moves the exp/log-scale arithmetic into Core ML while keeping RNG source explicit at runtime.",
            "Use this between vits_prior and vits_flow when you want the full latent-sampling formula out of host-side tensor code.",
        ],
    ),
    "vits_wave_generator": ExportTargetSpec(
        name="vits_wave_generator",
        description="Final SoVITS waveform generator fed by precomputed latent z and speaker/style condition ge.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth or s2Gv2ProPlus.pth"],
        inputs=[
            TensorSpec("z", "float32", "[batch, latent_channels, latent_frames]", "Precomputed latent tensor after flow(reverse)."),
            TensorSpec("ge", "float32", "[batch, gin_channels, 1]", "Precomputed waveform-conditioning style vector."),
        ],
        outputs=[
            TensorSpec("audio", "float32", "[batch, 1, waveform_samples]", "Decoded waveform."),
        ],
        notes=[
            "This isolates the HiFi-GAN style generator as a standalone Core ML target fed by a precomputed latent z.",
            "Callers should zero invalid latent frames with y_mask before invoking this target.",
            "Waveform sample count depends on latent_frames and the fixed upsample stack.",
        ],
    ),
    "vits_masked_wave_generator": ExportTargetSpec(
        name="vits_masked_wave_generator",
        description="Final SoVITS waveform generator that applies z * y_mask inside the graph before decoding audio.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth or s2Gv2ProPlus.pth"],
        inputs=[
            TensorSpec("z", "float32", "[batch, latent_channels, latent_frames]", "Latent tensor after flow(reverse)."),
            TensorSpec("y_mask", "float32", "[batch, 1, latent_frames]", "Latent validity mask."),
            TensorSpec("ge", "float32", "[batch, gin_channels, 1]", "Precomputed waveform-conditioning style vector."),
        ],
        outputs=[
            TensorSpec("masked_z", "float32", "[batch, latent_channels, latent_frames]", "Masked latent tensor after z * y_mask."),
            TensorSpec("audio", "float32", "[batch, 1, waveform_samples]", "Decoded waveform."),
        ],
        notes=[
            "This removes the post-flow masking step from host-side tensor code.",
            "Prefer this over vits_wave_generator when the bundle/runtime contract provides y_mask directly from vits_prior.",
        ],
    ),
    "vits_flow": ExportTargetSpec(
        name="vits_flow",
        description="Reverse flow subgraph that maps sampled prior latent z_p and condition ge to decoder latent z.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth or s2Gv2ProPlus.pth"],
        inputs=[
            TensorSpec("z_p", "float32", "[batch, latent_channels, latent_frames]", "Sampled prior latent tensor before flow reverse."),
            TensorSpec("y_mask", "float32", "[batch, 1, latent_frames]", "Latent validity mask."),
            TensorSpec("ge", "float32", "[batch, gin_channels, 1]", "Precomputed waveform-conditioning style vector."),
        ],
        outputs=[
            TensorSpec("z", "float32", "[batch, latent_channels, latent_frames]", "Latent tensor after reverse flow."),
        ],
        notes=[
            "This keeps randomness outside Core ML by accepting sampled z_p as an explicit input.",
            "Use this between vits_prior and vits_masked_wave_generator for an ANE-friendly VITS chain.",
        ],
    ),
    "t2s_prefill_prepare": ExportTargetSpec(
        name="t2s_prefill_prepare",
        description="Experimental Text2Semantic prefill front-half that builds prompt embeddings and the stacked attention mask.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/s1v3.ckpt"],
        inputs=[
            TensorSpec("prompts", "int32", "[1, prompt_frames]", "Prompt semantic token sequence."),
            TensorSpec("prompt_length", "int32", "[1]", "Real prompt semantic token count before zero-padding."),
            TensorSpec("ref_seq", "int32", "[1, ref_phone_count]", "Reference phone ids."),
            TensorSpec("ref_seq_length", "int32", "[1]", "Real reference phone count before zero-padding."),
            TensorSpec("text_seq", "int32", "[1, text_phone_count]", "Target phone ids."),
            TensorSpec("text_seq_length", "int32", "[1]", "Real target phone count before zero-padding."),
            TensorSpec("ref_bert", "float32", "[1024, ref_phone_count]", "Reference phone-level BERT feature."),
            TensorSpec("text_bert", "float32", "[1024, text_phone_count]", "Target phone-level BERT feature."),
        ],
        outputs=[
            TensorSpec("xy_pos", "float32", "[1, src_tokens, hidden_dim]", "Concatenated positioned text/audio prompt embeddings."),
            TensorSpec("prompt_attn_mask", "int32", "[1, heads, src_tokens, src_tokens]", "Prompt attention mask encoded as 0/1 for cross-model transport."),
            TensorSpec("active_src_len", "int32", "[1]", "Real packed source length reused as cache_len."),
            TensorSpec("next_position", "int32", "[1]", "Audio-only decode position index seeded from the real prompt semantic length."),
        ],
        notes=[
            "This target is experimental and exists to create a hard artifact boundary before the sensitive transformer prefill core.",
            "Use this ahead of t2s_prefill_core when evaluating mixed-precision split export strategies.",
        ],
    ),
    "t2s_prefill_core": ExportTargetSpec(
        name="t2s_prefill_core",
        description="Experimental Text2Semantic prefill back-half that consumes prepared prompt tensors and emits logits plus KV cache.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/s1v3.ckpt"],
        inputs=[
            TensorSpec("xy_pos", "float32", "[1, src_tokens, hidden_dim]", "Prepared positioned prompt embedding tensor from t2s_prefill_prepare."),
            TensorSpec("prompt_attn_mask", "int32", "[1, heads, src_tokens, src_tokens]", "Prepared 0/1 prompt attention mask from t2s_prefill_prepare."),
            TensorSpec("active_src_len", "int32", "[1]", "Real packed source length reused as cache_len."),
            TensorSpec("position_seed", "int32", "[1]", "Audio-only decode position index seeded from the real prompt semantic length."),
        ],
        outputs=[
            TensorSpec("logits", "float32", "[1, vocab_size]", "Next-token logits after prompt prefill."),
            TensorSpec("sampled_token", "int32", "[1, 1]", "Greedy-decoded semantic token selected inside the Core ML graph."),
            TensorSpec("eos_reached", "int32", "[1]", "Whether sampled_token equals eos_token."),
            TensorSpec("cache_len", "int32", "[1]", "Active KV cache length."),
            TensorSpec("next_position", "int32", "[1]", "Audio-only position index for the next decode step."),
            TensorSpec("k_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Stacked key cache."),
            TensorSpec("v_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Stacked value cache."),
        ],
        notes=[
            "This target is experimental and is intended to stay float32 while earlier prompt preparation stays float16.",
            "Bundle/runtime integration should wait until split-path parity is validated on random samples.",
        ],
    ),
    "t2s_prefill": ExportTargetSpec(
        name="t2s_prefill",
        description="Single-stream Text2Semantic prefill wrapper with stacked KV cache tensors.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/s1v3.ckpt"],
        inputs=[
            TensorSpec("prompts", "int32", "[1, prompt_frames]", "Prompt semantic token sequence."),
            TensorSpec("prompt_length", "int32", "[1]", "Real prompt semantic token count before zero-padding."),
            TensorSpec("ref_seq", "int32", "[1, ref_phone_count]", "Reference phone ids."),
            TensorSpec("ref_seq_length", "int32", "[1]", "Real reference phone count before zero-padding."),
            TensorSpec("text_seq", "int32", "[1, text_phone_count]", "Target phone ids."),
            TensorSpec("text_seq_length", "int32", "[1]", "Real target phone count before zero-padding."),
            TensorSpec("ref_bert", "float32", "[1024, ref_phone_count]", "Reference phone-level BERT feature."),
            TensorSpec("text_bert", "float32", "[1024, text_phone_count]", "Target phone-level BERT feature."),
        ],
        outputs=[
            TensorSpec("logits", "float32", "[1, vocab_size]", "Next-token logits after prompt prefill."),
            TensorSpec("sampled_token", "int32", "[1, 1]", "Greedy-decoded semantic token selected inside the Core ML graph."),
            TensorSpec("eos_reached", "int32", "[1]", "Whether sampled_token equals eos_token."),
            TensorSpec("cache_len", "int32", "[1]", "Active KV cache length."),
            TensorSpec("next_position", "int32", "[1]", "Audio-only position index for the next decode step."),
            TensorSpec("k_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Stacked key cache."),
            TensorSpec("v_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Stacked value cache."),
        ],
        notes=[
            "Current wrapper is exact-safe for single-stream decode and designed for later Core ML stateful export.",
            "Greedy token selection is now emitted directly by the Core ML graph, so the runtime no longer has to argmax logits on the host side.",
        ],
    ),
    "t2s_decode_step": ExportTargetSpec(
        name="t2s_decode_step",
        description="Single-stream Text2Semantic decode-step wrapper that consumes stacked KV cache tensors.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/s1v3.ckpt"],
        inputs=[
            TensorSpec("last_token", "int32", "[1, 1]", "Most recently sampled semantic token."),
            TensorSpec("position_index", "int32", "[1]", "Audio-only position index."),
            TensorSpec("cache_len", "int32", "[1]", "Current active KV cache length."),
            TensorSpec("k_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Stacked key cache."),
            TensorSpec("v_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Stacked value cache."),
        ],
        outputs=[
            TensorSpec("logits", "float32", "[1, vocab_size]", "Next-token logits."),
            TensorSpec("sampled_token", "int32", "[1, 1]", "Greedy-decoded semantic token selected inside the Core ML graph."),
            TensorSpec("eos_reached", "int32", "[1]", "Whether sampled_token equals eos_token."),
            TensorSpec("next_cache_len", "int32", "[1]", "Updated active KV cache length."),
            TensorSpec("next_position", "int32", "[1]", "Position index for the next step."),
            TensorSpec("next_k_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Updated stacked key cache."),
            TensorSpec("next_v_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Updated stacked value cache."),
        ],
        notes=[
            "This wrapper mirrors the current single-step decode path without introducing heuristic changes.",
            "Core ML conversion is landed and validated with chained parity against the PyTorch path.",
            "Greedy token selection is now emitted directly by the Core ML graph, leaving only the decode loop control on the host side.",
        ],
    ),
    "t2s_decode_prepare": ExportTargetSpec(
        name="t2s_decode_prepare",
        description="Experimental Text2Semantic decode front-half that builds the positioned token embedding for a single decode step.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/s1v3.ckpt"],
        inputs=[
            TensorSpec("last_token", "int32", "[1, 1]", "Most recently sampled semantic token."),
            TensorSpec("position_index", "int32", "[1]", "Absolute audio position index."),
        ],
        outputs=[
            TensorSpec("xy_pos", "float32", "[1, 1, hidden_dim]", "Prepared positioned decode embedding tensor."),
        ],
        notes=[
            "This target is experimental and exists to create a hard artifact boundary before the sensitive decode transformer core.",
            "Use this ahead of t2s_decode_core when evaluating mixed-precision split export strategies.",
        ],
    ),
    "t2s_decode_core": ExportTargetSpec(
        name="t2s_decode_core",
        description="Experimental Text2Semantic decode back-half that consumes prepared decode embeddings and KV cache tensors.",
        implemented=True,
        coreml_ready=True,
        required_artifacts=["GPT_SoVITS/pretrained_models/s1v3.ckpt"],
        inputs=[
            TensorSpec("xy_pos", "float32", "[1, 1, hidden_dim]", "Prepared positioned decode embedding tensor from t2s_decode_prepare."),
            TensorSpec("position_index", "int32", "[1]", "Absolute audio position index."),
            TensorSpec("cache_len", "int32", "[1]", "Current active KV cache length."),
            TensorSpec("k_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Stacked key cache."),
            TensorSpec("v_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Stacked value cache."),
        ],
        outputs=[
            TensorSpec("logits", "float32", "[1, vocab_size]", "Next-token logits."),
            TensorSpec("sampled_token", "int32", "[1, 1]", "Greedy-decoded semantic token selected inside the Core ML graph."),
            TensorSpec("eos_reached", "int32", "[1]", "Whether sampled_token equals eos_token."),
            TensorSpec("next_cache_len", "int32", "[1]", "Updated active KV cache length."),
            TensorSpec("next_position", "int32", "[1]", "Position index for the next step."),
            TensorSpec("next_k_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Updated stacked key cache."),
            TensorSpec("next_v_cache", "float32", "[layers, 1, cache_capacity, hidden_dim]", "Updated stacked value cache."),
        ],
        notes=[
            "This target is experimental and is intended to stay float32 while earlier decode preparation stays float16.",
            "Bundle/runtime integration should wait until chained split-decode parity is validated on random samples.",
        ],
    ),
    "vits_decoder": ExportTargetSpec(
        name="vits_decoder",
        description="Legacy placeholder for a monolithic end-to-end VITS decoder export target.",
        implemented=False,
        coreml_ready=False,
        required_artifacts=["GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth or s2Gv2ProPlus.pth"],
        inputs=[],
        outputs=[],
        notes=[
            "The VITS runtime is now Core ML-complete as a split chain: vits_decode_condition -> vits_prior -> vits_latent_sampler -> vits_flow -> vits_masked_wave_generator.",
            "This placeholder remains non-exportable because the runtime contract is bundle-level rather than a single monolithic Core ML model.",
        ],
    ),
}


def list_target_specs() -> List[ExportTargetSpec]:
    return list(TARGET_SPECS.values())


def get_target_spec(name: str) -> ExportTargetSpec:
    return TARGET_SPECS[name]


COREML_READY_TARGETS = tuple(
    spec.name for spec in TARGET_SPECS.values() if spec.implemented and spec.coreml_ready
)
