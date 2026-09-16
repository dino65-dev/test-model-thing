"""TMT-v4: P0-P4 recurrent byte model with bounded, owned memory credit."""
from __future__ import annotations
import itertools
import math
import os
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

BYTE_VOCAB, EOS, VOCAB_SIZE, EPS = 256, 256, 257, 1e-6
CHECKPOINT_FORMAT = "tmt-v4"


def logit(x):
    x = min(max(x, 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


def unit(x):
    return x / mx.sqrt(mx.sum(mx.square(x)) + EPS)


class SelectiveAdamW(opt.AdamW):
    """AdamW with bias correction and decay only on chosen dense matrices.

    MLX AdamW applies one decay value to every trainable leaf. Recurrent time
    constants, raw angles, gates, normalization and biases must not receive
    that per-byte shrinkage, so this class applies decoupled decay only to the
    conventional dense matrices listed below.
    """

    _DECAY_SUFFIXES = (
        "weights.weight", "key.weight", "query.weight", "value.weight",
        "out.weight", "decode.weight",
    )

    def __init__(self, learning_rate, dense_weight_decay=1e-4):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=0.0,
            bias_correction=True,
        )
        self.dense_weight_decay = float(dense_weight_decay)

    @classmethod
    def _decays(cls, path):
        path = str(path)
        return path.startswith("predictors.") or path.endswith(cls._DECAY_SUFFIXES)

    def update(self, model, gradients):
        parameters = model.trainable_parameters()
        factor = 1.0 - self.learning_rate * self.dense_weight_decay
        def scaled(tree, prefix=""):
            if isinstance(tree, dict):
                return {key: scaled(value, f"{prefix}.{key}" if prefix else str(key)) for key, value in tree.items()}
            if isinstance(tree, list):
                return [scaled(value, f"{prefix}.{i}" if prefix else str(i)) for i, value in enumerate(tree)]
            return tree * factor if self._decays(prefix) else tree
        # Preserve empty module branches: MLX tree_map expects the exact model
        # tree, whereas tree_flatten/tree_unflatten intentionally drops them.
        model.update(self.apply_gradients(gradients, scaled(parameters)))


class Encoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d)
        self._embed_trace = mx.zeros((VOCAB_SIZE, d // 2, 2, 2))

    def __call__(self, t):
        return self.embed(t)

    def reset_memory(self):
        self._embed_trace = mx.zeros(self._embed_trace.shape)


class Decoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.decode = nn.Linear(d, VOCAB_SIZE)

    def __call__(self, x):
        return self.decode(x)


class Layer(nn.Module):
    """Pairwise trace-compatible recurrent dynamics with raw base angles."""

    def __init__(self, d, min_half, max_half, *, complex_enabled=True, learn_decay=True):
        super().__init__()
        if d % 2:
            raise ValueError("dim must be even")
        self.dim, self.complex_enabled, self.learn_decay = d, complex_enabled, learn_decay
        n = d // 2
        nh = max(1, round(math.sqrt(n)))
        np = math.ceil(n / nh)
        half = [math.exp(math.log(min_half) + (math.log(max_half) - math.log(min_half)) * i / max(nh - 1, 1)) for i in range(nh)]
        canonical = [2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 2048]
        periods = canonical[:np] if np <= len(canonical) else [2 * math.exp(math.log(1024) * i / max(np - 1, 1)) for i in range(np)]
        pairs = [(half[i % nh], periods[(i // nh) % np]) for i in range(n)]
        self.decay = mx.array([logit(2 ** (-1 / h)) for h, _ in pairs])
        self.phase = mx.array([2 * math.pi / period for _, period in pairs])
        self.write_scale, self.write_bias = mx.zeros((d,)), mx.zeros((d,))
        self.phase_scale, self.phase_bias = mx.zeros((n,)), mx.zeros((n,))
        self.norm, self.weights, self.silu = nn.LayerNorm(d), nn.Linear(d, d, bias=False), nn.SiLU()
        self._state = mx.zeros((d,))
        self._decay_trace = mx.zeros((d,))
        self._phase_trace = mx.zeros((d,))
        self._write_scale_trace = mx.zeros((d,))
        self._write_bias_trace = mx.zeros((d,))
        self._phase_scale_trace = mx.zeros((d,))
        self._phase_bias_trace = mx.zeros((d,))
        frozen = []
        if not learn_decay:
            frozen.append("decay")
        if not complex_enabled:
            frozen.extend(("phase", "phase_scale", "phase_bias"))
        if frozen:
            self.freeze(keys=frozen, recurse=False)

    @staticmethod
    def rotate(v, c, s):
        a, b = v[..., 0::2], v[..., 1::2]
        return mx.stack((c * a - s * b, s * a + c * b), axis=-1).reshape(v.shape)

    @staticmethod
    def j(v):
        a, b = v[..., 0::2], v[..., 1::2]
        return mx.stack((-b, a), axis=-1).reshape(v.shape)

    def dynamics(self, x):
        rho = mx.sigmoid(self.decay)
        pair_x = .5 * (x[0::2] + x[1::2])
        if self.complex_enabled:
            z = self.phase_scale * pair_x + self.phase_bias
            theta = self.phase + math.pi * mx.tanh(z)
        else:
            # An architectural real-only ablation: frozen phase parameters are
            # not merely set to zero; no phase/controller path is evaluated.
            z = mx.zeros((self.dim // 2,))
            theta = z
        return rho, mx.cos(theta), mx.sin(theta), mx.sqrt(mx.maximum(1 - mx.square(rho), EPS)), z, pair_x

    def step(self, x, state, dummy=None):
        rho, c, s, scale, z, pair_x = self.dynamics(x)
        pair = lambda q: mx.stack((q, q), axis=-1).reshape((self.dim,))
        gate = mx.sigmoid(self.write_scale * x + self.write_bias)
        rotated = self.rotate(state, c, s)
        nxt = pair(rho) * rotated + pair(scale) * gate * x
        if dummy is not None:
            nxt = nxt + dummy
        return x + self.silu(self.weights(self.norm(nxt))), nxt, (rho, c, s, scale, gate, rotated, z, pair_x)

    def history(self, trace, rho, c, s):
        return mx.stack((rho, rho), axis=-1).reshape((self.dim,)) * self.rotate(trace, c, s)

    def phase_direct(self, rho, c, s, state, theta_derivative):
        pair = lambda q: mx.stack((q, q), axis=-1).reshape((self.dim,))
        return pair(rho * theta_derivative) * self.rotate(self.j(state), c, s)

    def next_decay_trace(self, x, rho, c, s, scale, gate, rotated):
        dr = rho * (1 - rho)
        ds = -rho * dr / mx.maximum(scale, EPS)
        pair = lambda q: mx.stack((q, q), axis=-1).reshape((self.dim,))
        return self.history(self._decay_trace, rho, c, s) + pair(dr) * rotated + pair(ds) * gate * x

    def injection_diagonal(self, x, scale, gate):
        pair = lambda q: mx.stack((q, q), axis=-1).reshape((self.dim,))
        return pair(scale) * (gate + x * gate * (1 - gate) * self.write_scale)

    def embedding_block(self, x, state, rho, c, s, scale, gate, z):
        """Exact local 2-by-2 ds_pair/dx_pair including phase rank-one term."""
        n = self.dim // 2
        pair = lambda q: mx.stack((q, q), axis=-1).reshape((self.dim,))
        diagonal = self.injection_diagonal(x, scale, gate).reshape((n, 2))
        direct = mx.stack((
            mx.stack((diagonal[:, 0], mx.zeros((n,))), axis=-1),
            mx.stack((mx.zeros((n,)), diagonal[:, 1]), axis=-1),
        ), axis=-2)
        if not self.complex_enabled:
            return direct
        sech2 = 1 - mx.square(mx.tanh(z))
        gamma = .5 * math.pi * self.phase_scale * sech2
        rotated = (pair(rho) * self.rotate(self.j(state), c, s)).reshape((n, 2))
        return direct + rotated[:, :, None] * gamma[:, None, None]

    def trace_block_history(self, trace, rho, c, s):
        a, b = trace[..., 0, :], trace[..., 1, :]
        return mx.stack((c[None, :, None] * a - s[None, :, None] * b,
                         s[None, :, None] * a + c[None, :, None] * b), axis=-2) * rho[None, :, None, None]

    def reset_memory(self):
        self._state = mx.zeros((self.dim,))
        self._decay_trace = mx.zeros((self.dim,))
        self._phase_trace = mx.zeros((self.dim,))
        self._write_scale_trace = mx.zeros((self.dim,))
        self._write_bias_trace = mx.zeros((self.dim,))
        self._phase_scale_trace = mx.zeros((self.dim,))
        self._phase_bias_trace = mx.zeros((self.dim,))


class AssociativeMemory(nn.Module):
    """Forget-first delta memory; controller learns only from temporal BPTT."""

    def __init__(self, d, m, half_life=1024):
        super().__init__()
        self.memory_dim = m
        self.key, self.query, self.value, self.out = nn.Linear(d, m, bias=False), nn.Linear(d, m, bias=False), nn.Linear(d, m, bias=False), nn.Linear(m, d, bias=False)
        self.forget, self.rate = nn.Linear(d, 1), nn.Linear(d, 1)
        self.forget.bias = mx.array([logit(2 ** (-1 / half_life))])
        self.rate.bias = mx.array([logit(.4)])
        self._matrix = mx.zeros((m, m))

    def read(self, h, matrix=None):
        return (self._matrix if matrix is None else matrix) @ unit(self.query(h))

    def next_matrix(self, h, matrix=None):
        matrix = self._matrix if matrix is None else matrix
        k, v = unit(self.key(h)), mx.tanh(self.value(h))
        decayed = mx.sigmoid(self.forget(h))[0] * matrix
        eta = .25 * mx.sigmoid(self.rate(h))[0]
        error = v - decayed @ k
        return decayed + eta * error[:, None] * k[None, :]

    def reset_memory(self):
        self._matrix = mx.zeros((self.memory_dim, self.memory_dim))


class Model(nn.Module):
    def __init__(self, dim, layers, temp, lr, memory_dim=None, future_block=16, target_ema=.995,
                 min_half_life=4, max_half_life=65536, latent_weight=.25, variance_weight=.02,
                 *, memory_enabled=True, complex_enabled=True, learn_decay=True,
                 dense_weight_decay=1e-4, memory_bptt_block=32):
        super().__init__()
        if dim % 2:
            raise ValueError("dim must be even")
        self.dim, self.layercount, self.temp = dim, layers, temp
        self.future_block, self.target_ema = future_block, target_ema
        self.horizons = (1, 2, 4, 8, 16)
        self.latent_weight, self.variance_weight = latent_weight, variance_weight
        self.memory_enabled, self.complex_enabled, self.learn_decay = memory_enabled, complex_enabled, learn_decay
        self.memory_bptt_block = memory_bptt_block
        self.encoder, self.decoder = Encoder(dim), Decoder(dim)
        self.layers = [Layer(dim, min_half_life, max_half_life, complex_enabled=complex_enabled, learn_decay=learn_decay) for _ in range(layers)]
        self.predictors = {f"h{h}": nn.Linear(dim, dim) for h in self.horizons}
        self.memory = AssociativeMemory(dim, memory_dim or max(16, min(64, dim // 8)))
        # Memory has exactly one optimizer owner. It remains usable in core(),
        # but normal online gradients do not update its controller parameters.
        self.memory.freeze()
        self.optimizer = SelectiveAdamW(lr, dense_weight_decay)
        self.memory_optimizer = SelectiveAdamW(lr, 0.0)
        self._target_embed = mx.array(self.encoder.embed.weight)
        self._repr_buffer, self._repr_capacity = mx.zeros((0, dim)), 32

    def buffer_parameter_invariant(self):
        names = " ".join(str(k) for k, _ in util.tree_flatten(self.parameters()))
        return not any(x in names for x in ("_state", "_trace", "_matrix", "_target_embed", "_repr_buffer"))

    def reset_memory(self):
        for layer in self.layers:
            layer.reset_memory()
        self.encoder.reset_memory()
        self.memory.reset_memory()
        self._repr_buffer = mx.zeros((0, self.dim))

    def core(self, token, dummies=None, states_override=None, matrix_override=None):
        x = self.encoder(mx.array(token) if isinstance(token, int) else token)
        states, dynamics, inputs = [], [], []
        for i, layer in enumerate(self.layers):
            inputs.append(x)
            previous = layer._state if states_override is None else states_override[i]
            x, state, dynamics_i = layer.step(x, previous, None if dummies is None else dummies[i])
            states.append(state)
            dynamics.append(dynamics_i)
        base = x
        h = base + self.memory.out(self.memory.read(base, matrix_override)) if self.memory_enabled else base
        return self.decoder(h), h, states, dynamics, inputs, base

    def commit(self, states, write):
        for layer, state in zip(self.layers, states):
            layer._state = mx.stop_gradient(state)
        if self.memory_enabled:
            self.memory._matrix = mx.stop_gradient(self.memory.next_matrix(write))
            mx.eval(*[x._state for x in self.layers], self.memory._matrix)
        else:
            mx.eval(*[x._state for x in self.layers])

    def frozen_step(self, token):
        logits, *_ = self.represent(token)
        return logits

    def represent(self, token):
        logits, h, states, dynamics, inputs, base = self.core(token)
        self.commit(states, base)
        return logits, h, states, dynamics, inputs, base

    def stateless_step(self, token):
        """P0 no-state probe preserving every live recurrent and trace buffer."""
        old = [x._state for x in self.layers]
        matrix = self.memory._matrix
        traces = [(x._decay_trace, x._phase_trace, x._write_scale_trace, x._write_bias_trace, x._phase_scale_trace, x._phase_bias_trace) for x in self.layers]
        embed_trace, repr_buffer = self.encoder._embed_trace, self._repr_buffer
        self.reset_memory()
        logits, *_ = self.core(token)
        for layer, state, values in zip(self.layers, old, traces):
            layer._state = state
            layer._decay_trace, layer._phase_trace, layer._write_scale_trace, layer._write_bias_trace, layer._phase_scale_trace, layer._phase_bias_trace = values
        self.memory._matrix, self.encoder._embed_trace, self._repr_buffer = matrix, embed_trace, repr_buffer
        return logits

    def sample(self, logits):
        p = mx.softmax(logits)
        entropy = -mx.sum(p * mx.log(p + 1e-8)) / math.log(VOCAB_SIZE)
        return mx.random.categorical(logits / mx.maximum(.1, self.temp * (1 - .5 * entropy)).item()).item()

    def target(self, future, horizon):
        values = list(future)[:horizon]
        if not values:
            raise ValueError("future block is empty")
        weights = mx.array([math.exp(-(i + 1) / max(horizon, 1)) for i in range(len(values))])
        weights = weights / mx.sum(weights)
        return mx.stop_gradient(mx.sum(self._target_embed[mx.array(values)] * weights[:, None], axis=0))

    def latent_loss(self, h, future):
        usable = [horizon for horizon in self.horizons if len(future) >= horizon]
        return mx.mean(mx.stack([1 - mx.sum(unit(self.predictors[f"h{horizon}"](h)) * unit(self.target(future, horizon))) for horizon in usable]))

    def repr_loss(self, h):
        xs = mx.concatenate((self._repr_buffer, h[None, :]), axis=0)
        if xs.shape[0] < 2:
            return mx.array(0.)
        return mx.mean(mx.maximum(0., 1 - mx.sqrt(mx.var(xs, axis=0) + 1e-4)))

    def train_step(self, current, target, future=None):
        """One online backbone step; delayed memory credit is handled separately."""
        if not 0 <= current < VOCAB_SIZE or not 0 <= target < VOCAB_SIZE:
            raise ValueError("invalid token")
        future = tuple(future or (target,))
        params = self.trainable_parameters()
        dummies = [mx.zeros((self.dim,)) for _ in self.layers]

        def lossfn(updated, ds):
            self.update(updated)
            logits, h, states, dyn, inputs, base = self.core(current, ds)
            byte = nn.losses.cross_entropy(logits[None, :], mx.array([target])).mean()
            latent = self.latent_loss(h, future)
            variance = self.repr_loss(h)
            return byte + self.latent_weight * latent + self.variance_weight * variance, (byte, latent, variance, h, states, dyn, inputs, base)

        (loss, aux), (grads, dldstate) = mx.value_and_grad(lossfn, argnums=(0, 1))(params, dummies)
        byte, latent, variance, h, states, dyn, inputs, base = aux
        rho, c, s, scale, gate, rotated, z, pair_x = dyn[0]
        n = self.dim // 2
        block_hist = self.layers[0].trace_block_history(self.encoder._embed_trace, rho, c, s)
        signal = dldstate[0].reshape((n, 2))
        embed_hist = mx.sum(block_hist * signal[None, :, :, None], axis=2).reshape((VOCAB_SIZE, self.dim))
        grads["encoder"]["embed"]["weight"] = grads["encoder"]["embed"]["weight"] + embed_hist
        block_next = block_hist + (mx.arange(VOCAB_SIZE) == current)[:, None, None, None] * self.layers[0].embedding_block(inputs[0], self.layers[0]._state, rho, c, s, scale, gate, z)[None, :, :, :]
        traces = []
        for i, layer in enumerate(self.layers):
            rho, c, s, scale, gate, rotated, z, pair_x = dyn[i]
            pair = lambda q: mx.stack((q, q), axis=-1).reshape((self.dim,))

            def add_vector(name, trace):
                grads["layers"][i][name] = grads["layers"][i][name] + dldstate[i] * layer.history(trace, rho, c, s)

            def add_pair(name, trace):
                grads["layers"][i][name] = grads["layers"][i][name] + mx.sum((dldstate[i] * layer.history(trace, rho, c, s)).reshape((-1, 2)), axis=1)

            if layer.learn_decay:
                add_pair("decay", layer._decay_trace)
            if layer.complex_enabled:
                add_pair("phase", layer._phase_trace)
                add_pair("phase_scale", layer._phase_scale_trace)
                add_pair("phase_bias", layer._phase_bias_trace)
            add_vector("write_scale", layer._write_scale_trace)
            add_vector("write_bias", layer._write_bias_trace)
            sech2 = 1 - mx.square(mx.tanh(z))
            phase_base = layer.phase_direct(rho, c, s, layer._state, mx.ones(z.shape))
            write_scale_direct = pair(scale) * gate * (1 - gate) * mx.square(inputs[i])
            write_bias_direct = pair(scale) * gate * (1 - gate) * inputs[i]
            traces.append((
                layer.next_decay_trace(inputs[i], rho, c, s, scale, gate, rotated),
                layer.history(layer._phase_trace, rho, c, s) + phase_base if layer.complex_enabled else mx.zeros((self.dim,)),
                layer.history(layer._write_scale_trace, rho, c, s) + write_scale_direct,
                layer.history(layer._write_bias_trace, rho, c, s) + write_bias_direct,
                layer.history(layer._phase_scale_trace, rho, c, s) + layer.phase_direct(rho, c, s, layer._state, math.pi * sech2 * pair_x) if layer.complex_enabled else mx.zeros((self.dim,)),
                layer.history(layer._phase_bias_trace, rho, c, s) + layer.phase_direct(rho, c, s, layer._state, math.pi * sech2) if layer.complex_enabled else mx.zeros((self.dim,)),
            ))
        self.optimizer.update(self, grads)
        for layer, state, values in zip(self.layers, states, traces):
            layer._state = mx.stop_gradient(state)
            layer._decay_trace, layer._phase_trace, layer._write_scale_trace, layer._write_bias_trace, layer._phase_scale_trace, layer._phase_bias_trace = [mx.stop_gradient(x) for x in values]
        self.encoder._embed_trace = mx.stop_gradient(block_next)
        if self.memory_enabled:
            self.memory._matrix = mx.stop_gradient(self.memory.next_matrix(base))
        self._repr_buffer = mx.concatenate((self._repr_buffer, mx.stop_gradient(h)[None, :]), axis=0)[-self._repr_capacity:]
        self._target_embed = mx.stop_gradient(self.target_ema * self._target_embed + (1 - self.target_ema) * self.encoder.embed.weight)
        mx.eval(self.parameters(), self.optimizer.state, self.encoder._embed_trace, self._target_embed, self._repr_buffer, self.memory._matrix, *[x._state for x in self.layers])
        return {"loss": loss.item(), "byte": byte.item(), "latent": latent.item(), "variance": variance.item(), "memory_write": mx.stop_gradient(base)}

    def memory_temporal_step(self, writes, targets, initial_matrix):
        """Memory-only BPTT over future byte CE, without altering live state.

        `writes` are detached backbone states from one bounded chunk. The
        decoder and backbone are constants here; gradients flow only through
        key/query/value/out/forget/rate and the unrolled delta-memory matrices.
        """
        if not self.memory_enabled:
            return 0.0
        if not writes or len(writes) != len(targets):
            raise ValueError("matched nonempty memory writes and targets required")
        params = self.memory.parameters()
        initial = mx.stop_gradient(initial_matrix)

        def lossfn(updated):
            self.memory.update(updated)
            matrix, losses = initial, []
            for write, target in zip(writes, targets):
                write = mx.stop_gradient(write)
                matrix = self.memory.next_matrix(write, matrix)
                h = write + self.memory.out(self.memory.read(write, matrix))
                logits = self.decoder(h)
                losses.append(nn.losses.cross_entropy(logits[None, :], mx.array([target])).mean())
            return mx.mean(mx.stack(losses))

        loss, grads = mx.value_and_grad(lossfn)(params)
        # Frozen prevents the online optimizer from touching memory. Temporarily
        # expose these leaves only to their dedicated temporal optimizer.
        self.memory.unfreeze()
        self.memory_optimizer.update(self.memory, grads)
        self.memory.freeze()
        mx.eval(self.memory.parameters(), self.memory_optimizer.state)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError("non-finite temporal memory loss")
        return value

    def health_metrics(self, write):
        """Cheap stability observables for a live training stream."""
        rho = mx.concatenate([mx.sigmoid(layer.decay) for layer in self.layers])
        half_life = math.log(.5) / mx.log(mx.maximum(rho, 1e-8))
        states = mx.concatenate([layer._state for layer in self.layers])
        traces = mx.concatenate([layer._decay_trace for layer in self.layers] + [layer._phase_trace for layer in self.layers])
        phase = mx.concatenate([layer.phase for layer in self.layers])
        forget = mx.sigmoid(self.memory.forget(write))[0] if self.memory_enabled else mx.array(0.)
        values = {
            "rho_half_min": mx.min(half_life), "rho_half_median": mx.median(half_life), "rho_half_max": mx.max(half_life),
            "phase_mean": mx.mean(phase), "phase_std": mx.sqrt(mx.var(phase)),
            "state_rms": mx.sqrt(mx.mean(mx.square(states))), "trace_rms": mx.sqrt(mx.mean(mx.square(traces))),
            "forget": forget,
        }
        mx.eval(*values.values())
        return {key: value.item() for key, value in values.items()}

    def adaptive_patches(self, data, max_patch=16, entropy_threshold=.35):
        patch, ent = [], []
        for token in data:
            logits = self.frozen_step(token)
            p = mx.softmax(logits)
            entropy = (-mx.sum(p * mx.log(p + 1e-8)) / math.log(VOCAB_SIZE)).item()
            patch.append(token)
            ent.append(entropy)
            if entropy > entropy_threshold or len(patch) >= max_patch:
                yield tuple(patch), sum(ent) / len(ent)
                patch, ent = [], []
        if patch:
            yield tuple(patch), sum(ent) / len(ent)

    def _buffers(self):
        data = {
            "buffer.embed_trace": self.encoder._embed_trace,
            "buffer.target_embed": self._target_embed,
            "buffer.repr": self._repr_buffer,
            "buffer.memory": self.memory._matrix,
        }
        for i, layer in enumerate(self.layers):
            for name in ("state", "decay_trace", "phase_trace", "write_scale_trace", "write_bias_trace", "phase_scale_trace", "phase_bias_trace"):
                data[f"buffer.{name}.{i}"] = getattr(layer, f"_{name}")
        return data

    def _metadata(self):
        return {
            "format": CHECKPOINT_FORMAT,
            "dim": str(self.dim), "layers": str(self.layercount),
            "memory_enabled": str(int(self.memory_enabled)),
            "complex_enabled": str(int(self.complex_enabled)),
            "learn_decay": str(int(self.learn_decay)),
        }

    def save(self, path):
        data = {f"m.{key}": value for key, value in util.tree_flatten(self.parameters())}
        data.update({f"o.{key}": value for key, value in util.tree_flatten(self.optimizer.state)})
        data.update({f"mo.{key}": value for key, value in util.tree_flatten(self.memory_optimizer.state)})
        data.update(self._buffers())
        tmp = path + ".temporary.safetensors"
        mx.save_safetensors(tmp, data, metadata=self._metadata())
        os.replace(tmp, path)

    def load(self, path):
        if not os.path.exists(path):
            return False
        data, metadata = mx.load(path, return_metadata=True)
        if metadata != self._metadata():
            raise ValueError(f"checkpoint metadata is incompatible with {CHECKPOINT_FORMAT}; retrain instead of partially loading")
        model = {key[2:]: value for key, value in data.items() if key.startswith("m.")}
        expected_model = dict(util.tree_flatten(self.parameters()))
        expected_buffers = self._buffers()
        supplied_buffers = {key: value for key, value in data.items() if key.startswith("buffer.")}
        def exact(supplied, expected):
            return set(supplied) == set(expected) and all(supplied[key].shape == expected[key].shape for key in expected)
        buffers_exact = set(supplied_buffers) == set(expected_buffers)
        for key, value in supplied_buffers.items():
            if key == "buffer.repr":
                buffers_exact = buffers_exact and value.ndim == 2 and value.shape[1] == self.dim and 0 <= value.shape[0] <= self._repr_capacity
            elif key in expected_buffers:
                buffers_exact = buffers_exact and value.shape == expected_buffers[key].shape
        if not exact(model, expected_model) or not buffers_exact:
            raise ValueError(f"checkpoint is not an exact {CHECKPOINT_FORMAT} parameter/buffer match")
        unknown = [key for key in data if not (key.startswith("m.") or key.startswith("o.") or key.startswith("mo.") or key.startswith("buffer."))]
        if unknown:
            raise ValueError("checkpoint contains unknown tensor keys")
        self.update(util.tree_unflatten(list(model.items())))
        self.encoder._embed_trace = supplied_buffers["buffer.embed_trace"]
        self._target_embed = supplied_buffers["buffer.target_embed"]
        self._repr_buffer = supplied_buffers["buffer.repr"]
        self.memory._matrix = supplied_buffers["buffer.memory"]
        for i, layer in enumerate(self.layers):
            for name in ("state", "decay_trace", "phase_trace", "write_scale_trace", "write_bias_trace", "phase_scale_trace", "phase_bias_trace"):
                setattr(layer, f"_{name}", supplied_buffers[f"buffer.{name}.{i}"])
        main_state = {key[2:]: value for key, value in data.items() if key.startswith("o.")}
        memory_state = {key[3:]: value for key, value in data.items() if key.startswith("mo.")}
        if main_state:
            self.optimizer.state = util.tree_unflatten(list(main_state.items()))
        if memory_state:
            self.memory_optimizer.state = util.tree_unflatten(list(memory_state.items()))
        return True


def continuous_bytes(files):
    files = [Path(x) for x in files]
    if not files:
        raise FileNotFoundError("no corpus files found")
    while True:
        for file in files:
            with file.open("rb") as f:
                while chunk := f.read(65536):
                    yield from chunk
            yield EOS


def stream_examples(stream, future_block):
    it = iter(stream)
    window = deque(itertools.islice(it, future_block + 1))
    while len(window) >= 2:
        current = window.popleft()
        yield current, window[0], tuple(itertools.islice(window, 0, future_block))
        window.append(next(it))


class Runtime:
    def __init__(self, path, **kwargs):
        self.model, self.path, self.step = Model(**kwargs), path, 0

    def train(self, glob_pattern="wikipedia_clean/**/wiki_*"):
        files = sorted(Path().glob(glob_pattern))
        writes, targets = [], []
        block_start = mx.stop_gradient(self.model.memory._matrix)
        report_start, report_step = time.perf_counter(), self.step
        for current, next_token, future in stream_examples(continuous_bytes(files), self.model.future_block):
            metrics = self.model.train_step(current, next_token, future)
            self.step += 1
            if self.model.memory_enabled:
                writes.append(metrics["memory_write"])
                targets.append(next_token)
                if len(writes) == self.model.memory_bptt_block:
                    memory_ce = self.model.memory_temporal_step(writes, targets, block_start)
                    writes, targets = [], []
                    block_start = mx.stop_gradient(self.model.memory._matrix)
                    print(f"{self.step}: memory_bptt_byte={memory_ce:.4f}")
            if self.step % 100 == 0:
                elapsed = max(time.perf_counter() - report_start, 1e-9)
                speed = (self.step - report_step) / elapsed
                report_start, report_step = time.perf_counter(), self.step
                health = self.model.health_metrics(metrics["memory_write"])
                print(f"{self.step}: loss={metrics['loss']:.4f} byte={metrics['byte']:.4f} latent={metrics['latent']:.4f} bytes_per_s={speed:.1f} health={health}")
            if self.step % 500 == 0:
                self.model.save(self.path)

    def chat(self):
        while True:
            text = input("\nUser >> ")
            logits = None
            for token in (*text.encode("utf-8"), EOS):
                logits = self.model.frozen_step(token)
            print("Model >> ", end="", flush=True)
            while True:
                token = self.model.sample(logits)
                if token == EOS:
                    print()
                    break
                print(bytes((token,)).decode("utf-8", errors="replace"), end="", flush=True)
                logits = self.model.frozen_step(token)

    def __call__(self):
        mode = input("mode [train, chat, math] >> ").strip().lower()
        self.model.load(self.path)
        try:
            if mode == "train": self.train()
            elif mode == "chat": self.chat()
            elif mode == "math":
                from math_diagnostics import main
                main()
            else: raise ValueError("mode must be train, chat, or math")
        finally:
            self.model.save(self.path)


if __name__ == "__main__":
    Runtime(path="tmt-v4.safetensors", dim=512, layers=16, temp=.75, lr=5e-4)()
