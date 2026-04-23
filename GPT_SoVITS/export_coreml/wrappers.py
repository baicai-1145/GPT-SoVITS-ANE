from typing import List, Tuple

import torch
from torch import IntTensor, LongTensor, Tensor, nn
from torch.nn import functional as F

from AR.models.t2s_model import scaled_dot_product_attention


def build_phone_level_feature(char_feature: Tensor, word2ph: IntTensor) -> Tensor:
    repeat_counts = word2ph.to(dtype=torch.long, device=char_feature.device)
    phone_level_feature = torch.repeat_interleave(char_feature, repeat_counts, dim=0)
    return phone_level_feature.transpose(0, 1).contiguous()


def build_phone_level_feature_padded(
    char_feature: Tensor,
    word2ph: IntTensor,
    phone_capacity: int,
) -> Tensor:
    repeat_counts = word2ph.to(dtype=torch.long, device=char_feature.device)
    starts = torch.cumsum(repeat_counts, dim=0) - repeat_counts
    ends = starts + repeat_counts
    positions = torch.arange(phone_capacity, device=char_feature.device, dtype=torch.long).view(1, phone_capacity)
    mask = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
    phone_level_feature = torch.matmul(mask.to(dtype=char_feature.dtype).transpose(0, 1), char_feature)
    return phone_level_feature.transpose(0, 1).contiguous()


def _build_decode_invalid_mask(
    next_cache_len: Tensor,
    cache_capacity: int,
    batch_size: int,
    num_heads: int,
    q_len: int,
    device: torch.device,
) -> Tensor:
    positions = torch.arange(cache_capacity, device=device, dtype=torch.long).view(1, 1, 1, cache_capacity)
    invalid = positions >= next_cache_len.to(dtype=torch.long).view(1, 1, 1, 1)
    return invalid.expand(batch_size, num_heads, q_len, cache_capacity)


def _zero_invalid_cache_tail(cache: Tensor, next_cache_len: Tensor) -> Tensor:
    positions = torch.arange(cache.shape[1], device=cache.device, dtype=torch.long).view(1, cache.shape[1], 1)
    valid = positions < next_cache_len.to(dtype=torch.long).view(1, 1, 1)
    return torch.where(valid, cache, torch.zeros_like(cache))


def _update_cache_slot(cache: Tensor, value: Tensor, cache_len: Tensor) -> Tensor:
    cache_index = cache_len.reshape(-1).to(dtype=torch.long, device=cache.device)
    write_mask = F.one_hot(cache_index, num_classes=cache.shape[1]).to(device=cache.device, dtype=torch.bool)
    write_mask = write_mask.view(cache.shape[0], cache.shape[1], 1)
    expanded_value = value.expand(-1, cache.shape[1], -1)
    return torch.where(write_mask, expanded_value, cache)


def _run_t2s_mlp(mlp, x: Tensor) -> Tensor:
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    rows = batch_size * seq_len
    x2d = x.reshape(rows, x.shape[2])
    w1 = mlp.w1_t.transpose(0, 1)
    w2 = mlp.w2_t.transpose(0, 1)
    if rows == 1:
        x2d = F.relu(F.linear(x2d, w1, mlp.b1))
        x2d = F.linear(x2d, w2, mlp.b2)
    else:
        x2d = torch.addmm(mlp.b1, x2d, mlp.w1_t)
        x2d = F.relu(x2d)
        x2d = torch.addmm(mlp.b2, x2d, mlp.w2_t)
    return x2d.view(batch_size, seq_len, mlp.output_dim)


def _decode_next_token_block_tensorized(
    block,
    x: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    cache_len: Tensor,
    torch_sdpa: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_size = x.shape[0]
    q_len = x.shape[1]
    rows = batch_size * q_len
    if rows == 1:
        qkv = F.linear(x.reshape(rows, x.shape[2]), block.qkv_w_t.transpose(0, 1), block.qkv_b)
    else:
        qkv = torch.addmm(block.qkv_b, x.reshape(rows, x.shape[2]), block.qkv_w_t)
    qkv = qkv.view(batch_size, q_len, block.qkv_out_dim)
    q, k, v = qkv.chunk(3, dim=-1)

    next_cache_len = cache_len.to(dtype=torch.int32, device=x.device) + k.shape[1]
    updated_k_cache = _update_cache_slot(k_cache, k, cache_len)
    updated_v_cache = _update_cache_slot(v_cache, v, cache_len)
    safe_k_cache = _zero_invalid_cache_tail(updated_k_cache, next_cache_len)
    safe_v_cache = _zero_invalid_cache_tail(updated_v_cache, next_cache_len)

    q = q.view(batch_size, q_len, block.num_heads, -1).transpose(1, 2)
    k_full = safe_k_cache.view(batch_size, safe_k_cache.shape[1], block.num_heads, -1).transpose(1, 2)
    v_full = safe_v_cache.view(batch_size, safe_v_cache.shape[1], block.num_heads, -1).transpose(1, 2)

    attn_mask = _build_decode_invalid_mask(
        next_cache_len,
        safe_k_cache.shape[1],
        batch_size,
        block.num_heads,
        q_len,
        x.device,
    )
    if torch_sdpa:
        attn = F.scaled_dot_product_attention(q, k_full, v_full, ~attn_mask)
    else:
        attn = scaled_dot_product_attention(q, k_full, v_full, attn_mask)

    attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
    out_rows = batch_size * q_len
    if out_rows == 1:
        attn = F.linear(attn.reshape(out_rows, attn.shape[2]), block.out_w_t.transpose(0, 1), block.out_b)
    else:
        attn = torch.addmm(block.out_b, attn.reshape(out_rows, attn.shape[2]), block.out_w_t)
    attn = attn.view(batch_size, q_len, block.out_out_dim)

    x = x + attn
    x = F.layer_norm(
        x,
        [block.hidden_dim],
        block.norm_w1,
        block.norm_b1,
        block.norm_eps1,
    )
    x = x + _run_t2s_mlp(block.mlp, x)
    x = F.layer_norm(
        x,
        [block.hidden_dim],
        block.norm_w2,
        block.norm_b2,
        block.norm_eps2,
    )
    return x, safe_k_cache, safe_v_cache


def _greedy_token_from_logits(logits: Tensor) -> IntTensor:
    return torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.int32)


def _eos_reached_from_token(sampled_token: Tensor, eos_token: int) -> IntTensor:
    return sampled_token.reshape(-1).eq(int(eos_token)).to(dtype=torch.int32)


class ZhBertPhoneFeatureWrapper(nn.Module):
    def __init__(self, bert_model: nn.Module, phone_capacity: int | None = None):
        super().__init__()
        self.bert_model = bert_model
        self.phone_capacity = int(phone_capacity) if phone_capacity is not None else None

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        word2ph: IntTensor,
    ) -> Tensor:
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        char_feature = torch.cat(outputs["hidden_states"][-3:-2], dim=-1)[0][1:-1]
        if self.phone_capacity is not None:
            return build_phone_level_feature_padded(char_feature, word2ph, self.phone_capacity)
        return build_phone_level_feature(char_feature, word2ph)


class ZhBertCharFeatureWrapper(nn.Module):
    def __init__(self, bert_model: nn.Module):
        super().__init__()
        self.bert_model = bert_model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tensor:
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        return torch.cat(outputs["hidden_states"][-3:-2], dim=-1)[0][1:-1]


class G2PWProbabilityWrapper(nn.Module):
    def __init__(self, g2pw_model: nn.Module):
        super().__init__()
        self.g2pw_model = g2pw_model

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        phoneme_mask: Tensor,
        char_ids: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        output = self.g2pw_model(
            input_ids,
            token_type_ids,
            attention_mask,
            phoneme_mask,
            char_ids,
            position_ids,
        )
        if isinstance(output, (tuple, list)):
            return output[0]
        return output


class CNHubertContentWrapper(nn.Module):
    def __init__(self, cnhubert_model: nn.Module):
        super().__init__()
        self.hubert_model = cnhubert_model.model

    def forward(self, input_values: Tensor) -> Tensor:
        return self.hubert_model(input_values)["last_hidden_state"].transpose(1, 2)


class SpeakerEncoderWrapper(nn.Module):
    def __init__(self, speaker_encoder):
        super().__init__()
        self.embedding_model = speaker_encoder.embedding_model

    def forward(self, fbank_80: Tensor) -> Tensor:
        return self.embedding_model.forward3(fbank_80)


class PromptSemanticExtractorWrapper(nn.Module):
    def __init__(self, vits_model: nn.Module):
        super().__init__()
        self.vq_model = vits_model

    def forward(self, ssl_content: Tensor) -> Tensor:
        codes = self.vq_model.extract_latent(ssl_content)
        return codes[:, 0]


class VITSDecodeConditionWrapper(nn.Module):
    def __init__(self, vits_model: nn.Module):
        super().__init__()
        self.ref_enc = vits_model.ref_enc
        self.version = vits_model.version
        self.is_v2pro = bool(getattr(vits_model, "is_v2pro", False))
        self.sv_emb = getattr(vits_model, "sv_emb", None)
        self.prelu = getattr(vits_model, "prelu", None)
        self.ge_to512 = getattr(vits_model, "ge_to512", None)
        self.refer_channels = int(getattr(vits_model.ref_enc, "in_dim", 704))

    def forward(
        self,
        refer: Tensor,
        sv_emb: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        refer = refer[:, : self.refer_channels, :]
        # For decode_condition the entire refer tensor is valid. With dynamic refer length,
        # expressing this as an explicit all-ones mask introduces symbolic shape ops that
        # Core ML converts but fails to execute reliably at runtime. Passing no mask is
        # mathematically equivalent to an all-valid mask for MelStyleEncoder.
        ge = self.ref_enc(refer, None)
        if self.is_v2pro:
            ge = ge + self.sv_emb(sv_emb).unsqueeze(-1)
            # Core ML rejects rank-3 PReLU with per-channel alpha, so express it with
            # elementwise ops that preserve the exact channel-wise slope.
            pos = F.relu(ge)
            neg = ge - pos
            ge = pos + self.prelu.weight.view(1, -1, 1).to(dtype=ge.dtype, device=ge.device) * neg
            ge_text = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
        else:
            ge_text = ge
        return ge, ge_text


class VITSPriorWrapper(nn.Module):
    def __init__(self, vits_model: nn.Module):
        super().__init__()
        self.quantizer = vits_model.quantizer
        self.enc_p = vits_model.enc_p
        self.semantic_frame_rate = vits_model.semantic_frame_rate

    def forward(
        self,
        codes: LongTensor,
        text: LongTensor,
        ge_text: Tensor,
        code_lengths: LongTensor,
        text_lengths: LongTensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        code_lengths = torch.as_tensor(code_lengths, device=codes.device, dtype=torch.long)
        text_lengths = torch.as_tensor(text_lengths, device=text.device, dtype=torch.long)
        y_lengths = code_lengths * 2

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")

        _, prior_mean, prior_log_scale, y_mask, _, _ = self.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            ge_text,
            1.0,
        )
        return prior_mean, prior_log_scale, y_mask


class VITSWaveGeneratorWrapper(nn.Module):
    def __init__(self, vits_model: nn.Module):
        super().__init__()
        self.dec = vits_model.dec

    def forward(
        self,
        z: Tensor,
        ge: Tensor,
    ) -> Tensor:
        return self.dec(z, g=ge)


class VITSMaskedWaveGeneratorWrapper(nn.Module):
    def __init__(self, vits_model: nn.Module):
        super().__init__()
        self.dec = vits_model.dec

    def forward(
        self,
        z: Tensor,
        y_mask: Tensor,
        ge: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        masked_z = z * y_mask
        audio = self.dec(masked_z, g=ge)
        return masked_z, audio


class VITSFlowWrapper(nn.Module):
    def __init__(self, vits_model: nn.Module):
        super().__init__()
        self.flow = vits_model.flow

    def forward(
        self,
        z_p: Tensor,
        y_mask: Tensor,
        ge: Tensor,
    ) -> Tensor:
        return self.flow(z_p, y_mask, g=ge, reverse=True)


class VITSLatentSamplerWrapper(nn.Module):
    def forward(
        self,
        prior_mean: Tensor,
        prior_log_scale: Tensor,
        noise: Tensor,
        noise_scale: Tensor,
    ) -> Tensor:
        scale = noise_scale.to(dtype=prior_mean.dtype, device=prior_mean.device).view(1, 1, 1)
        return prior_mean + noise * torch.exp(prior_log_scale) * scale


class T2SPrefillWrapper(nn.Module):
    def __init__(self, decoder: nn.Module, max_decode_steps: int = 1500):
        super().__init__()
        self.bert_proj = decoder.bert_proj
        self.ar_text_embedding = decoder.ar_text_embedding
        self.ar_text_position = decoder.ar_text_position
        self.ar_audio_embedding = decoder.ar_audio_embedding
        self.ar_audio_position = decoder.ar_audio_position
        self.t2s_transformer = decoder.t2s_transformer
        self.ar_predict_layer = decoder.ar_predict_layer
        self.num_head = int(decoder.num_head)
        self.max_decode_steps = int(max_decode_steps)
        self.eos_token = int(decoder.EOS)
        self.use_torch_sdpa = False

    def _embed_text_component(
        self,
        seq: LongTensor,
        bert: Tensor,
    ) -> Tensor:
        x = self.ar_text_embedding(seq)
        bert = bert.unsqueeze(0).to(dtype=self.bert_proj.weight.dtype, device=x.device)
        return x + self.bert_proj(bert.transpose(1, 2))

    def _pack_ref_text_embeddings(
        self,
        ref_embed: Tensor,
        text_embed: Tensor,
        ref_seq_length: Tensor,
        text_seq_length: Tensor,
    ) -> Tensor:
        batch_size, ref_capacity, hidden_dim = ref_embed.shape
        text_capacity = int(text_embed.shape[1])
        total_capacity = ref_capacity + text_capacity
        positions = torch.arange(total_capacity, device=ref_embed.device, dtype=torch.long).view(1, total_capacity)
        ref_len = ref_seq_length.to(dtype=torch.long, device=ref_embed.device).view(batch_size, 1)
        text_len = text_seq_length.to(dtype=torch.long, device=text_embed.device).view(batch_size, 1)
        ref_target_positions = positions.view(1, total_capacity, 1)
        ref_source_positions = torch.arange(ref_capacity, device=ref_embed.device, dtype=torch.long).view(1, 1, ref_capacity)
        ref_assignment = (ref_target_positions == ref_source_positions) & (
            ref_target_positions < ref_len.view(batch_size, 1, 1)
        )
        ref_packed = torch.matmul(ref_assignment.to(dtype=ref_embed.dtype), ref_embed)

        text_target_positions = positions.view(1, total_capacity, 1)
        text_source_positions = torch.arange(text_capacity, device=text_embed.device, dtype=torch.long).view(1, 1, text_capacity)
        text_relative_positions = text_target_positions - ref_len.view(batch_size, 1, 1)
        text_assignment = (text_relative_positions == text_source_positions) & (
            text_target_positions >= ref_len.view(batch_size, 1, 1)
        ) & (
            text_target_positions < (ref_len + text_len).view(batch_size, 1, 1)
        )
        text_packed = torch.matmul(text_assignment.to(dtype=text_embed.dtype), text_embed)
        return ref_packed + text_packed

    def _pack_text_audio_embeddings(
        self,
        x: Tensor,
        y_pos: Tensor,
        x_len: Tensor,
        prompt_len: Tensor,
    ) -> Tensor:
        batch_size, x_capacity, hidden_dim = x.shape
        prompt_capacity = int(y_pos.shape[1])
        total_capacity = x_capacity + prompt_capacity
        device = x.device
        positions = torch.arange(total_capacity, device=device, dtype=torch.long).view(1, total_capacity)
        x_len = x_len.to(dtype=torch.long, device=device).view(batch_size, 1)
        prompt_len = prompt_len.to(dtype=torch.long, device=device).view(batch_size, 1)

        x_target_positions = positions.view(1, total_capacity, 1)
        x_source_positions = torch.arange(x_capacity, device=device, dtype=torch.long).view(1, 1, x_capacity)
        x_assignment = (x_target_positions == x_source_positions) & (
            x_target_positions < x_len.view(batch_size, 1, 1)
        )
        x_packed = torch.matmul(x_assignment.to(dtype=x.dtype), x)

        y_target_positions = positions.view(1, total_capacity, 1)
        y_source_positions = torch.arange(prompt_capacity, device=device, dtype=torch.long).view(1, 1, prompt_capacity)
        y_relative_positions = y_target_positions - x_len.view(batch_size, 1, 1)
        y_assignment = (y_relative_positions == y_source_positions) & (
            y_target_positions >= x_len.view(batch_size, 1, 1)
        ) & (
            y_target_positions < (x_len + prompt_len).view(batch_size, 1, 1)
        )
        y_packed = torch.matmul(y_assignment.to(dtype=y_pos.dtype), y_pos)
        return x_packed + y_packed

    def _build_prompt_attn_mask(
        self,
        x_capacity: int,
        prompt_capacity: int,
        x_len: Tensor,
        prompt_len: Tensor,
        device: torch.device,
    ) -> Tensor:
        total_capacity = x_capacity + prompt_capacity
        positions = torch.arange(total_capacity, device=device, dtype=torch.long)
        query_positions = positions.view(1, total_capacity, 1)
        key_positions = positions.view(1, 1, total_capacity)
        x_len = x_len.to(dtype=torch.long, device=device).view(-1, 1, 1)
        prompt_len = prompt_len.to(dtype=torch.long, device=device).view(-1, 1, 1)
        total_valid_len = x_len + prompt_len

        valid_queries = (query_positions < total_valid_len).to(dtype=torch.int32)
        valid_keys = (key_positions < total_valid_len).to(dtype=torch.int32)
        x_queries = (query_positions < x_len).to(dtype=torch.int32)
        y_queries = (query_positions >= x_len).to(dtype=torch.int32) * valid_queries
        y_keys = (key_positions >= x_len).to(dtype=torch.int32) * valid_keys
        future_y_keys = (key_positions > query_positions).to(dtype=torch.int32)

        invalid_score = (1 - valid_queries) + (1 - valid_keys)
        invalid_score = invalid_score + (x_queries * y_keys)
        invalid_score = invalid_score + (y_queries * y_keys * future_y_keys)
        invalid = invalid_score > 0
        return (
            invalid
            .unsqueeze(1)
            .expand(-1, self.num_head, total_capacity, total_capacity)
            .contiguous()
        )

    def _build_prompt_core_inputs(
        self,
        prompts: LongTensor,
        prompt_length: IntTensor,
        ref_seq: LongTensor,
        ref_seq_length: IntTensor,
        text_seq: LongTensor,
        text_seq_length: IntTensor,
        ref_bert: Tensor,
        text_bert: Tensor,
    ) -> Tuple[Tensor, Tensor, IntTensor, IntTensor]:
        ref_embed = self._embed_text_component(ref_seq, ref_bert)
        text_embed = self._embed_text_component(text_seq, text_bert)
        x = self._pack_ref_text_embeddings(ref_embed, text_embed, ref_seq_length, text_seq_length)
        x = self.ar_text_position(x)
        y_emb = self.ar_audio_embedding(prompts)
        prompt_capacity = int(y_emb.shape[1])
        prompt_len = prompt_length.to(dtype=torch.long, device=y_emb.device).reshape(1)
        prompt_positions = torch.arange(prompt_capacity, device=y_emb.device, dtype=torch.long).view(1, prompt_capacity)
        prompt_mask = prompt_positions < prompt_len
        y_emb = y_emb * prompt_mask.unsqueeze(-1).to(dtype=y_emb.dtype)
        y_pos = self.ar_audio_position(y_emb)
        x_len = ref_seq_length.to(dtype=torch.long, device=x.device).reshape(1) + text_seq_length.to(
            dtype=torch.long,
            device=x.device,
        ).reshape(1)
        xy_pos = self._pack_text_audio_embeddings(x, y_pos, x_len, prompt_len)
        prompt_attn_mask = self._build_prompt_attn_mask(
            int(x.shape[1]),
            prompt_capacity,
            x_len,
            prompt_len,
            x.device,
        )
        active_src_len = (x_len + prompt_len).to(dtype=torch.int32, device=x.device).reshape(1)
        next_position = prompt_len.to(dtype=torch.int32, device=x.device).reshape(1)
        return xy_pos, prompt_attn_mask, active_src_len, next_position

    def _run_prefill_core(
        self,
        xy_pos: Tensor,
        prompt_attn_mask: Tensor,
        active_src_len: Tensor,
        position_seed: Tensor,
    ) -> Tuple[Tensor, IntTensor, IntTensor, Tensor, Tensor, Tensor, Tensor]:
        prompt_attn_mask = prompt_attn_mask.to(dtype=torch.bool, device=xy_pos.device)
        cache_len_tensor = active_src_len.to(dtype=torch.int32, device=xy_pos.device).reshape(1)
        next_position_tensor = position_seed.to(dtype=torch.int32, device=xy_pos.device).reshape(1)
        active_src_len_long = active_src_len.to(dtype=torch.long, device=xy_pos.device).reshape(1)
        positions = torch.arange(xy_pos.shape[1], device=xy_pos.device, dtype=torch.long).view(1, xy_pos.shape[1], 1)
        padding_mask = positions >= active_src_len_long.view(1, 1, 1)
        xy_dec, k_cache, v_cache, cache_len = self.t2s_transformer.process_prompt(
            xy_pos,
            prompt_attn_mask,
            self.max_decode_steps,
            padding_mask,
            self.use_torch_sdpa,
        )
        last_index = torch.clamp(active_src_len_long - 1, min=0)
        last_positions = torch.arange(xy_dec.shape[1], device=xy_pos.device, dtype=torch.long).view(1, xy_dec.shape[1], 1)
        last_mask = last_positions == last_index.view(1, 1, 1)
        last_hidden = torch.sum(xy_dec * last_mask.to(dtype=xy_dec.dtype), dim=1)
        logits = self.ar_predict_layer(last_hidden)
        sampled_token = _greedy_token_from_logits(logits)
        eos_reached = _eos_reached_from_token(sampled_token, self.eos_token)
        k_cache_tensor = torch.stack(k_cache, dim=0)
        v_cache_tensor = torch.stack(v_cache, dim=0)
        return (
            logits,
            sampled_token,
            eos_reached,
            cache_len_tensor,
            next_position_tensor,
            k_cache_tensor,
            v_cache_tensor,
        )

    def forward(
        self,
        prompts: LongTensor,
        prompt_length: IntTensor,
        ref_seq: LongTensor,
        ref_seq_length: IntTensor,
        text_seq: LongTensor,
        text_seq_length: IntTensor,
        ref_bert: Tensor,
        text_bert: Tensor,
    ) -> Tuple[Tensor, IntTensor, IntTensor, Tensor, Tensor, Tensor, Tensor]:
        xy_pos, prompt_attn_mask, active_src_len, next_position = self._build_prompt_core_inputs(
            prompts,
            prompt_length,
            ref_seq,
            ref_seq_length,
            text_seq,
            text_seq_length,
            ref_bert,
            text_bert,
        )
        return self._run_prefill_core(xy_pos, prompt_attn_mask, active_src_len, next_position)


class T2SPrefillPrepareWrapper(T2SPrefillWrapper):
    def forward(
        self,
        prompts: LongTensor,
        prompt_length: IntTensor,
        ref_seq: LongTensor,
        ref_seq_length: IntTensor,
        text_seq: LongTensor,
        text_seq_length: IntTensor,
        ref_bert: Tensor,
        text_bert: Tensor,
    ) -> Tuple[Tensor, IntTensor, IntTensor, IntTensor]:
        xy_pos, prompt_attn_mask, active_src_len, next_position = self._build_prompt_core_inputs(
            prompts,
            prompt_length,
            ref_seq,
            ref_seq_length,
            text_seq,
            text_seq_length,
            ref_bert,
            text_bert,
        )
        return xy_pos, prompt_attn_mask.to(dtype=torch.int32), active_src_len, next_position


class T2SPrefillCoreWrapper(T2SPrefillWrapper):
    def forward(
        self,
        xy_pos: Tensor,
        prompt_attn_mask: IntTensor,
        active_src_len: IntTensor,
        position_seed: IntTensor,
    ) -> Tuple[Tensor, IntTensor, IntTensor, Tensor, Tensor, Tensor, Tensor]:
        return self._run_prefill_core(xy_pos, prompt_attn_mask, active_src_len, position_seed)


def _layer_norm_fp32(
    x: Tensor,
    normalized_shape: List[int],
    weight: Tensor,
    bias: Tensor,
    eps: float,
) -> Tensor:
    x_fp32 = x.to(dtype=torch.float32)
    return F.layer_norm(
        x_fp32,
        normalized_shape,
        weight.to(dtype=torch.float32, device=x.device),
        bias.to(dtype=torch.float32, device=x.device),
        eps,
    )


def _process_prompt_block_residual_layernorm_fp32(
    block,
    x: Tensor,
    attn_mask: Tensor,
    max_decode_steps: int,
    torch_sdpa: bool,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    batch_size = x.shape[0]
    q_len = x.shape[1]
    rows = batch_size * q_len
    if rows == 1:
        qkv = F.linear(x.reshape(rows, x.shape[2]), block.qkv_w_t.transpose(0, 1), block.qkv_b)
    else:
        qkv = torch.addmm(block.qkv_b, x.reshape(rows, x.shape[2]), block.qkv_w_t)
    qkv = qkv.view(batch_size, q_len, block.qkv_out_dim)
    q, k, v = qkv.chunk(3, dim=-1)

    kv_len = k.shape[1]
    k_cache = k
    v_cache = v

    q = q.view(batch_size, q_len, block.num_heads, -1).transpose(1, 2)
    k = k_cache.view(batch_size, kv_len, block.num_heads, -1).transpose(1, 2)
    v = v_cache.view(batch_size, kv_len, block.num_heads, -1).transpose(1, 2)

    if torch_sdpa:
        attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)
    else:
        attn = scaled_dot_product_attention(q, k, v, attn_mask)

    attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
    out_rows = batch_size * q_len
    if out_rows == 1:
        attn = F.linear(attn.reshape(out_rows, attn.shape[2]), block.out_w_t.transpose(0, 1), block.out_b)
    else:
        attn = torch.addmm(block.out_b, attn.reshape(out_rows, attn.shape[2]), block.out_w_t)
    attn = attn.view(batch_size, q_len, block.out_out_dim)

    runtime_dtype = x.dtype
    x = _layer_norm_fp32(
        x.to(dtype=torch.float32) + attn.to(dtype=torch.float32),
        [block.hidden_dim],
        block.norm_w1,
        block.norm_b1,
        block.norm_eps1,
    ).to(dtype=runtime_dtype)
    x = _layer_norm_fp32(
        x.to(dtype=torch.float32) + _run_t2s_mlp(block.mlp, x).to(dtype=torch.float32),
        [block.hidden_dim],
        block.norm_w2,
        block.norm_b2,
        block.norm_eps2,
    ).to(dtype=runtime_dtype)

    cache_capacity = kv_len + max_decode_steps
    full_k_cache = k_cache.new_zeros((batch_size, cache_capacity, k_cache.shape[2]))
    full_v_cache = v_cache.new_zeros((batch_size, cache_capacity, v_cache.shape[2]))
    full_k_cache[:, :kv_len] = k_cache
    full_v_cache[:, :kv_len] = v_cache
    return x, full_k_cache, full_v_cache, kv_len


class T2SPrefillResidualLayerNormFP32Wrapper(T2SPrefillWrapper):
    def forward(
        self,
        prompts: LongTensor,
        prompt_length: IntTensor,
        ref_seq: LongTensor,
        ref_seq_length: IntTensor,
        text_seq: LongTensor,
        text_seq_length: IntTensor,
        ref_bert: Tensor,
        text_bert: Tensor,
    ) -> Tuple[Tensor, IntTensor, IntTensor, Tensor, Tensor, Tensor, Tensor]:
        xy_pos, prompt_attn_mask, active_src_len, next_position = self._build_prompt_core_inputs(
            prompts,
            prompt_length,
            ref_seq,
            ref_seq_length,
            text_seq,
            text_seq_length,
            ref_bert,
            text_bert,
        )

        xy_dec = xy_pos
        k_layers: List[Tensor] = []
        v_layers: List[Tensor] = []
        cache_len = 0
        for block in self.t2s_transformer.blocks:
            xy_dec, k_cache, v_cache, cache_len = _process_prompt_block_residual_layernorm_fp32(
                block,
                xy_dec,
                prompt_attn_mask,
                self.max_decode_steps,
                self.use_torch_sdpa,
            )
            k_layers.append(k_cache)
            v_layers.append(v_cache)

        active_src_len_long = active_src_len.to(dtype=torch.long, device=xy_pos.device).reshape(1)
        last_index = torch.clamp(active_src_len_long - 1, min=0)
        last_positions = torch.arange(xy_dec.shape[1], device=xy_pos.device, dtype=torch.long).view(1, xy_dec.shape[1], 1)
        last_mask = last_positions == last_index.view(1, 1, 1)
        last_hidden = torch.sum(xy_dec * last_mask.to(dtype=xy_dec.dtype), dim=1)
        logits = self.ar_predict_layer(last_hidden)
        sampled_token = _greedy_token_from_logits(logits)
        eos_reached = _eos_reached_from_token(sampled_token, self.eos_token)
        cache_len_tensor = active_src_len.to(dtype=torch.int32, device=logits.device).reshape(1)
        return (
            logits,
            sampled_token,
            eos_reached,
            cache_len_tensor,
            next_position,
            torch.stack(k_layers, dim=0),
            torch.stack(v_layers, dim=0),
        )


class T2SDecodeStepWrapper(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.ar_audio_embedding = decoder.ar_audio_embedding
        self.ar_audio_position = decoder.ar_audio_position
        self.ar_predict_layer = decoder.ar_predict_layer
        self.blocks = decoder.t2s_transformer.blocks
        self.eos_token = int(decoder.EOS)
        self.use_torch_sdpa = False

    def _build_decode_xy_pos(
        self,
        last_token: LongTensor,
        position_index: Tensor,
    ) -> Tensor:
        y_emb = self.ar_audio_embedding(last_token)
        pe_table = self.ar_audio_position.pe[0].to(dtype=y_emb.dtype, device=y_emb.device)
        pos = torch.index_select(pe_table, dim=0, index=position_index.reshape(-1).to(dtype=torch.long))
        pos = pos.unsqueeze(0).expand(y_emb.shape[0], -1, -1)
        return y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * pos

    def _run_decode_core(
        self,
        xy_pos: Tensor,
        position_index: Tensor,
        cache_len: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
    ) -> Tuple[Tensor, IntTensor, IntTensor, Tensor, Tensor, Tensor, Tensor]:
        position_index = position_index.to(dtype=torch.int32, device=xy_pos.device).reshape(1)
        cache_len = cache_len.to(dtype=torch.int32, device=xy_pos.device).reshape(1)

        k_layers = [layer.clone() for layer in k_cache.unbind(0)]
        v_layers = [layer.clone() for layer in v_cache.unbind(0)]
        next_k_layers: List[Tensor] = []
        next_v_layers: List[Tensor] = []
        xy_dec = xy_pos
        for block, layer_k_cache, layer_v_cache in zip(self.blocks, k_layers, v_layers):
            xy_dec, next_k_cache, next_v_cache = _decode_next_token_block_tensorized(
                block,
                xy_dec,
                layer_k_cache,
                layer_v_cache,
                cache_len,
                self.use_torch_sdpa,
            )
            next_k_layers.append(next_k_cache)
            next_v_layers.append(next_v_cache)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        sampled_token = _greedy_token_from_logits(logits)
        eos_reached = _eos_reached_from_token(sampled_token, self.eos_token)
        next_cache_len_tensor = cache_len.to(dtype=torch.int32, device=logits.device) + 1
        next_position_tensor = position_index.to(dtype=torch.int32) + 1
        return (
            logits,
            sampled_token,
            eos_reached,
            next_cache_len_tensor,
            next_position_tensor,
            torch.stack(next_k_layers, dim=0),
            torch.stack(next_v_layers, dim=0),
        )

    def forward(
        self,
        last_token: LongTensor,
        position_index: Tensor,
        cache_len: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
    ) -> Tuple[Tensor, IntTensor, IntTensor, Tensor, Tensor, Tensor, Tensor]:
        xy_pos = self._build_decode_xy_pos(last_token, position_index)
        return self._run_decode_core(xy_pos, position_index, cache_len, k_cache, v_cache)


class T2SDecodePrepareWrapper(T2SDecodeStepWrapper):
    def forward(
        self,
        last_token: LongTensor,
        position_index: Tensor,
    ) -> Tensor:
        return self._build_decode_xy_pos(last_token, position_index)


class T2SDecodeCoreWrapper(T2SDecodeStepWrapper):
    def forward(
        self,
        xy_pos: Tensor,
        position_index: Tensor,
        cache_len: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
    ) -> Tuple[Tensor, IntTensor, IntTensor, Tensor, Tensor, Tensor, Tensor]:
        return self._run_decode_core(xy_pos, position_index, cache_len, k_cache, v_cache)
