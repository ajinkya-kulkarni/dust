"""Minimal FWDes implementation: forward-only transformer training with SGD.

No autograd or backward pass is used.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


def attention(q, k, v, window):
    """Causal attention with a left window; tensors have shape [batch, token, head, dim]."""
    length = q.shape[1]
    positions = torch.arange(length, device=q.device)
    distance = positions[:, None] - positions[None, :]
    mask = (distance >= 0) & (distance <= window)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
    )
    return out.transpose(1, 2)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    window_pattern: str = 'SSSL'


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = (x[..., :d], x[..., d:])
    return torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin))
        q, k = (norm(q), norm(k))
        y = attention(q, k, v, window_size[0])
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(padded_vocab, config.n_embd),
            'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3 ** 0.5 * self.config.n_embd ** (-0.5)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        (self.cos, self.sin) = (cos, sin)
        if self.transformer.wte.weight.device.type == 'cuda':
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = (freqs.cos().bfloat16(), freqs.sin().bfloat16())
        return (cos[None, :, None, :], sin[None, :, None, :])

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = (config.sequence_len, config.sequence_len // 2)
        char_to_w = {'L': (long_w, 0), 'S': (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits


@dataclass
class Recipe:
    writers: list[int]
    hidden: list[int]
    hubs: list[int]
    embedding: int
    head: int
    local: dict[str, int]
    nrep: int
    lr: float
    momentum: float
    gamma: float = 0.98  # attention-internal credit decay per token of lag


DEFAULT_CONFIGS = {
    '1M': {
        'steps': 61,
        'eval_interval': 2,
        'optimizer': {
            256: dict(lr=0.25, momentum=0.95),
            1024: dict(lr=0.225, momentum=0.95),
            4096: dict(lr=0.225, momentum=0.98),
            16384: dict(lr=0.225, momentum=0.98),
        },
    },
    '10M': {
        'steps': 610,
        'eval_interval': 10,
        'optimizer': {
            256: dict(lr=0.175, momentum=0.95),
            1024: dict(lr=0.25, momentum=0.95),
            4096: dict(lr=0.25, momentum=0.95),
            16384: dict(lr=0.25, momentum=0.95),
        },
        # At 16,384 draws the estimator constants were tuned with (lr, momentum) at fixed compute:
        # twice the attention-output draws, paid by the writer streams, and credit decay 0.99.
        'allocation': {
            16384: dict(writers=[34, 28, 14, 8, 6, 4, 2, 2], hubs=[12, 10, 8, 4, 4, 4, 4, 4], gamma=0.99),
        },
    },
}

DRAW_CHUNK_SIZES = {256: 2, 1024: 8, 4096: 8, 16384: 16}


def recipe(tokens, population):
    """Build a saved SGD configuration; population counts direct-loss draws only."""
    optimizer = DEFAULT_CONFIGS[tokens]['optimizer'][population]
    nrep = DRAW_CHUNK_SIZES[population]
    scale = population // 256
    override = DEFAULT_CONFIGS[tokens].get('allocation', {}).get(population, {})
    writers = [v * scale for v in override.get('writers', [38, 32, 16, 10, 6, 4, 2, 2])]
    hidden = writers.copy()
    hubs = [v * scale for v in override.get('hubs', [6, 6, 4, 2, 2, 2, 2, 2])]
    embedding, local = 10 * scale, [v * scale for v in [2, 20, 6, 12, 8]]

    return Recipe(
        writers, hidden, hubs, embedding, 4 * population,
        dict(zip(('q', 'v', 'gate', 'k', 've'), local)),
        nrep, **optimizer, gamma=override.get('gamma', 0.98),
    )


def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


class FWDes:
    """Estimate activation errors using forward evaluations, then form local outer products."""

    def __init__(self, model, settings, seed=42, rank=0, world=1):
        self.model, self.settings, self.world = model, settings, world
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * rank)
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.jitter = {}, {}, {}
        self.capturing = False

        counts = settings.writers + settings.hidden + settings.hubs + [settings.embedding]
        counts += list(settings.local.values())
        if any(k % (world * settings.nrep) for k in counts) or 8 % world:
            raise ValueError('Population/chunk size is incompatible with this GPU count.')
        slabs = model.config.vocab_size // 256
        if model.config.vocab_size % 256 or settings.head % (world * settings.nrep * slabs):
            raise ValueError('Head draws must visit each 256-column slab equally on every GPU.')

        self.hooks = [m.register_forward_hook(self.output_hook(n)) for n, m in self.modules.items()]
        for layer in range(model.config.n_layer):
            module = self.modules[f'transformer.h.{layer}.attn.c_proj']
            self.hooks.append(module.register_forward_pre_hook(self.input_hook(f'o@{layer}')))

    def autocast(self):
        return torch.autocast('cuda', dtype=self.dtype) if self.device.type == 'cuda' else nullcontext()

    def output_hook(self, name):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output
            if name in self.jitter:
                return output + self.jitter[name].to(output.dtype)

        return hook

    def input_hook(self, name):
        def hook(module, args):
            if name in self.jitter:
                return (args[0] + self.jitter[name].to(args[0].dtype),)

        return hook

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def noise(self, *shape, dtype=None):
        return torch.randn(*shape, device=self.device, dtype=dtype or self.dtype, generator=self.generator)

    def losses(self, x, y):
        with self.autocast():
            return self.model(x, y, loss_reduction='none').view_as(y).float()

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.capturing = True
        try:
            self.clean_loss = self.losses(x, y)
        finally:
            self.capturing = False

    def direct_errors(self, sites, draws, sigma):
        """One-sided loss drops, centered across each chunk of draws without rescaling."""
        n = self.settings.nrep
        x, y = self.x.repeat(n, 1), self.y.repeat(n, 1)
        width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in sites}

        for _ in range(draws // n):
            noise = {s: self.noise(n * self.batch, self.length, width(s)) for s in sites}
            self.jitter = {s: sigma * a for s, a in noise.items()}
            try:
                perturbed_loss = self.losses(x, y).view(n, self.batch, self.length)
            finally:
                self.jitter = {}

            reward = self.clean_loss.unsqueeze(0) - perturbed_loss
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            for site, a in noise.items():
                errors[site] -= torch.einsum(
                    'nbt,nbtd->btd', reward.to(self.dtype),
                    a.view(n, self.batch, self.length, -1),
                ).float() / sigma

        return {s: error / draws for s, error in errors.items()}

    def head_error(self, draws, sigma=0.05):
        """Perturb one vocabulary slab at a time; recompute CE from its changed logits."""
        n, b, t = self.settings.nrep, self.batch, self.length
        raw = self.outputs['lm_head']
        vocab, slab = self.model.config.vocab_size, 256
        slabs = vocab // slab
        logits = 15 * torch.tanh(raw[..., :vocab].float() / 15)
        slab_sums = logits.exp().view(b, t, slabs, slab).sum(-1)
        total = slab_sums.sum(-1)
        target = self.y.clamp(min=0)
        clean_target = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        target = target.repeat(n, 1)
        error = torch.zeros_like(raw, dtype=torch.float32)

        for chunk in range(draws // n):
            block = chunk % slabs
            lo = block * slab
            a = self.noise(n * b, t, slab)
            changed = raw[..., lo:lo + slab].float().repeat(n, 1, 1) + sigma * a.float()
            changed = 15 * torch.tanh(changed / 15)
            partition = (total - slab_sums[..., block]).repeat(n, 1) + changed.exp().sum(-1)
            in_slab = (target >= lo) & (target < lo + slab)
            target_logit = torch.where(
                in_slab,
                changed.gather(
                    -1, (target - lo).clamp(0, slab - 1).unsqueeze(-1),
                ).squeeze(-1),
                clean_target.repeat(n, 1),
            )
            loss = (partition.log() - target_logit).view(n, b, t)
            reward = torch.where(self.y.unsqueeze(0) != -1, self.clean_loss.unsqueeze(0) - loss, 0)
            if b > 1:
                reward = reward - reward.mean(1, keepdim=True)
            error[..., lo:lo + slab] -= torch.einsum(
                'nbt,nbtd->btd', (reward / sigma).to(self.dtype), a.view(n, b, t, slab),
            ).float()

        return error / (draws // slabs)

    def local_errors(self, kind, draws, targets, sigma=0.05):
        """Score attention-output changes against estimated output errors (no derivatives)."""
        cfg = self.model.config
        n, b, t, h, d = self.settings.nrep, self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        site_suffix = {'q': 'attn.c_q', 'k': 'attn.c_k', 'v': 'attn.c_v', 'gate': 'attn.ve_gate'}
        layers = [i for i in range(cfg.n_layer) if kind not in ('ve', 'gate') or has_ve(i, cfg.n_layer)]
        sites = {
            i: f'value_embeds.{i}' if kind == 've' else f'transformer.h.{i}.{site_suffix[kind]}'
            for i in layers
        }
        width = h if kind == 'gate' else cfg.n_embd
        errors = {i: torch.zeros(b, t, width, device=self.device) for i in layers}
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()

        def processed(raw):
            return norm(apply_rotary_emb(raw, cos, sin)).to(self.dtype)

        def repeat(x):
            return x.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, d)

        clean = {}
        for i in layers:
            prefix = f'transformer.h.{i}.attn.'
            q, k, v = [self.outputs[prefix + 'c_' + name].float().view(b, t, h, d) for name in ('q', 'k', 'v')]
            z = self.outputs[prefix + 've_gate'].float() if has_ve(i, cfg.n_layer) else None
            ve = self.outputs[f'value_embeds.{i}'].float().view(b, t, h, d) if z is not None else None
            values = v if z is None else v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp, kp = processed(q), processed(k)
            window = self.model.window_sizes[i][0]
            baseline = attention(qp, kp, values.to(self.dtype), window)
            positions = torch.arange(t, device=self.device)
            lag = positions[None, :] - positions[:, None]  # [perturbation position, output position]
            credit = ((lag >= 0) & (lag <= window)).float() * self.settings.gamma ** lag.clamp(min=0)
            clean[i] = q, k, z, ve, qp, kp, values, baseline, credit

        for _ in range(draws // n):
            for i in layers:
                a = self.noise(n, b, t, width, dtype=torch.float32)
                q, k, z, ve, qp, kp, values, baseline, credit = clean[i]
                window = self.model.window_sizes[i][0]
                divisor = sigma
                if kind in ('q', 'k'):
                    raw = q if kind == 'q' else k
                    perturbed = processed(
                        (raw.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d)
                    )
                    qa, ka = (perturbed, repeat(kp)) if kind == 'q' else (repeat(qp), perturbed)
                    delta = (
                        attention(qa, ka, repeat(values.to(self.dtype)), window).view(n, b, t, h, d)
                        - baseline
                    )
                else:
                    if kind == 'gate':
                        delta_gate = 2 * torch.sigmoid(z + sigma * a) - 2 * torch.sigmoid(z - sigma * a)
                        delta_value = delta_gate.unsqueeze(-1) * ve
                        divisor = 2 * sigma
                    else:
                        delta_value = sigma * a.view(n, b, t, h, d)
                        if kind == 've':
                            delta_value = delta_value * (2 * torch.sigmoid(z)).unsqueeze(-1)
                    delta = attention(
                        repeat(qp), repeat(kp),
                        delta_value.reshape(n * b, t, h, d).to(self.dtype), window,
                    ).view(n, b, t, h, d)

                score = torch.einsum('bthd,nbthd->nbth', targets[i].view(b, t, h, d).to(self.dtype), delta)
                if kind != 'q':
                    score = torch.einsum('nbsh,ts->nbth', score.float(), credit)
                estimate = torch.einsum('nbth,nbthd->bthd', score.float(), a.view(n, b, t, h, -1))
                errors[i] += estimate.reshape(b, t, width) / divisor

        return {sites[i]: errors[i] / draws for i in layers}

    def reassemble(self, site, error):
        """The exact local linear map: error times input, or scatter-add for an embedding."""
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)

    def scalar_gradients(self, draws, sigma=0.03):
        parameters = [self.model.resid_lambdas, self.model.x0_lambdas]
        original = [p.clone() for p in parameters]
        estimate = torch.zeros(2, self.model.config.n_layer, device=self.device)
        try:
            for _ in range(draws):
                a = self.noise(2, self.model.config.n_layer, dtype=torch.float32)
                losses = []
                for sign in (1, -1):
                    for p, clean, direction in zip(parameters, original, a):
                        p.copy_(clean + sign * sigma * direction)
                    losses.append(self.losses(self.x, self.y))
                difference = (losses[0] - losses[1])[self.y != -1].sum() / (2 * sigma)
                estimate += difference * a / draws
        finally:
            for p, clean in zip(parameters, original):
                p.copy_(clean)
        return dict(zip(('resid_lambdas', 'x0_lambdas'), estimate))

    @torch.no_grad()
    def step(self, x, y, optimizer):
        """One clean pass, direct-loss estimates, local attention estimates, then an SGD update."""
        self.capture(x, y)
        r, world = self.settings, self.world
        gradients = {}
        for layer in range(self.model.config.n_layer):
            prefix = f'transformer.h.{layer}.'
            groups = [
                ([prefix + 'attn.c_proj', prefix + 'mlp.c_proj'], r.writers[layer]),
                ([prefix + 'mlp.c_fc'], r.hidden[layer]),
            ]
            for sites, count in groups:
                for site, error in self.direct_errors(sites, count // world, 0.2).items():
                    gradients[site + '.weight'] = self.reassemble(site, error)

        site = 'transformer.wte'
        error = self.direct_errors([site], r.embedding // world, 0.2)[site]
        gradients[site + '.weight'] = self.reassemble(site, error)

        targets = {}
        for layer in range(self.model.config.n_layer):
            site = f'o@{layer}'
            sigma = 0.2 if layer < self.model.config.n_layer // 2 else 0.4
            targets[layer] = average_ranks(self.direct_errors([site], r.hubs[layer] // world, sigma)[site])

        gradients['lm_head.weight'] = self.reassemble('lm_head', self.head_error(r.head // world))
        for kind, count in r.local.items():
            for site, error in self.local_errors(kind, count // world, targets).items():
                gradients[site + '.weight'] = self.reassemble(site, error)
        gradients.update(self.scalar_gradients(8 // world))

        valid_tokens = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            gradient = average_ranks((gradients[name] / valid_tokens).to(parameter.dtype))
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Non-finite estimate for {name}')
            parameter.grad = gradient
        optimizer.step()
        return self.clean_loss[y != -1].mean().item()


def load_sequences(path, count, sequence_len=2048):
    """Read a token matrix or chunked dataset, excluding padded rows."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        rows = [
            chunk.view(data['batch_size'], data['sequence_size'])[:valid]
            for chunk, valid in zip(data['chunks'], data['valid_counts'])
        ]
        data = torch.cat(rows)
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[1] != sequence_len + 1:
        raise ValueError(f'{path}: expected [sequences, {sequence_len + 1}] token IDs')
    if len(data) < count:
        raise ValueError(f'{path}: need {count} sequences, found {len(data)}')
    if data.is_floating_point() or data.is_complex() or data.dtype == torch.bool:
        raise ValueError(f'{path}: token IDs must have an integer dtype')
    data = data[:count].long()
    if data.min() < 0 or data.max() >= 4096:
        raise ValueError(f'{path}: token IDs must be in [0, 4096)')
    return data


@torch.no_grad()
def evaluate(model, rows, device, batch_size=32):
    total, tokens = 0.0, 0
    for batch in rows.split(batch_size):
        batch = batch.to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with context:
            loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite evaluation loss')
        total += loss.item() * y.numel()
        tokens += y.numel()
    return total / tokens


def make_model(config, device, seed):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
    return model.requires_grad_(False)


def make_optimizer(model, settings):
    groups = {}
    for name, parameter in model.named_parameters():
        lr = settings.lr
        if name == 'transformer.wte.weight':
            lr = 1000.0
        elif 'value_embeds' in name:
            lr = 0.3
        elif 'lambdas' in name:
            lr = 0.03
        groups.setdefault(lr, []).append(parameter)
    return torch.optim.SGD(
        [{'params': ps, 'lr': lr} for lr, ps in groups.items()],
        momentum=settings.momentum,
    )


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', choices=DEFAULT_CONFIGS, default='1M')
    parser.add_argument(
        '--population', type=int, choices=DRAW_CHUNK_SIZES, default=16384,
        help='Total direct-loss draws across GPUs (default: 16384); each value selects its tuned recipe.',
    )
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/default'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--steps', type=int,
        help='Stop early for a smoke test; the training subset stays unchanged.',
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU.')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)

    settings = recipe(args.tokens, args.population)
    defaults = DEFAULT_CONFIGS[args.tokens]
    config = GPTConfig()
    steps = defaults['steps']
    if args.steps is not None and not 1 <= args.steps <= steps:
        parser.error(f'--steps must be between 1 and {steps}')
    paths = dict(
        train=args.data_dir / f'train_{args.tokens}.pt',
        val=args.data_dir / 'val.pt',
        test=args.data_dir / 'test.pt',
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error(
            f"Missing data: {', '.join(missing)}. Run python prepare_data.py "
            f"--tokens {args.tokens} --output {args.data_dir}"
        )
    training = load_sequences(paths['train'], steps * 8)
    validation = load_sequences(paths['val'], 544)
    testing = load_sequences(paths['test'], 544)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]
    steps = args.steps or steps

    # Every rank sees the same batch; only the perturbation population is divided across ranks.
    model = make_model(config, device, args.seed)
    optimizer = make_optimizer(model, settings)
    estimator = FWDes(model, settings, args.seed, rank, world)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        torch.distributed.barrier()

    metadata = dict(
        tokens=args.tokens,
        population=args.population,
        seed=args.seed,
        world=world,
        steps=steps,
        training_tokens=steps * 8 * config.sequence_len,
        validation_sequences=544,
        test_sequences=544,
        model=asdict(config),
        recipe=asdict(settings),
        reward_centering='per-token local draw-chunk mean; skipped at nrep=1; no rescaling',
        parameter_groups=[dict(lr=g['lr'], momentum=g['momentum'],
                               parameters=sum(p.numel() for p in g['params']))
                          for g in optimizer.param_groups],
        source_sha256=file_hash(Path(__file__)),
        torch_version=torch.__version__,
    )
    if rank == 0:
        metadata['data_sha256'] = {k: file_hash(v) for k, v in paths.items()}
        (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    best = evaluate(model, validation, device)

    def save_best():
        if rank == 0:
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                args.output / 'best.pt',
            )

    save_best()
    history = [dict(step=0, val_loss=best)]
    interval = defaults['eval_interval']
    start = time.monotonic()
    if rank == 0:
        print(
            f'{args.tokens}: {steps} steps, population={args.population}, GPUs={world}, val={best:.4f}',
            flush=True,
        )

    for step in range(1, steps + 1):
        batch = training[(step - 1) * 8:step * 8].to(device)
        loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        record = dict(step=step, train_loss=loss, elapsed_seconds=time.monotonic() - start)
        if step == 1 or step % interval == 0 or step == steps:
            value = evaluate(model, validation, device)
            record['val_loss'] = value
            if value < best:
                best = value
                save_best()
        history.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')

    estimator.close()
    if rank == 0:
        model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True))
        result = dict(best_val=best, test_at_best_val=evaluate(model, testing, device), history=history)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f"Best validation: {best:.4f}; test: {result['test_at_best_val']:.4f}", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
