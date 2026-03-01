#!/usr/bin/env python3
"""
Vertical Oscilloscope Waveform Generator v15
─────────────────────────────────────────────
• Audio Modulator — uses loaded audio amplitude envelope as AM/FM source
• BPM-synced drift with ratio snap + fine tune
• Timebase control
• 4 synth modulators (AM/FM) with cross-mod
• Modulators exceed base amplitude via Overdrive
• Audio monitoring via pygame
• Offline export to MP4/AVI at 720p, 1080p or 4K (CUDA-accelerated when available)
• Patch save/load (JSON)
"""

import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
import numpy as np
import threading
import multiprocessing
import os
import time
import io
import json
import copy
import pathlib
import wave as wavmod

try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError("Install Pillow:  pip install Pillow")

try:
    import cv2
except ImportError:
    raise ImportError("Install OpenCV:  pip install opencv-python")

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
except AttributeError:
    HAS_CUDA = False


# ─── Tables ──────────────────────────────────────────────────────────────────

RATIO_TABLE = [
    ("÷64", 1/64), ("÷32", 1/32), ("÷16", 1/16),
    ("÷8",  1/8),  ("÷6",  1/6),  ("÷4",  1/4),
    ("÷3",  1/3),  ("÷2",  1/2),  ("÷1.5", 2/3),
    ("×1",  1),    ("×1.5", 3/2), ("×2",  2),
    ("×3",  3),    ("×4",  4),    ("×6",  6),
    ("×8",  8),    ("×16", 16),   ("×32", 32),
    ("×64", 64),
]
RATIO_LABELS = [r[0] for r in RATIO_TABLE]
RATIO_VALUES = {r[0]: r[1] for r in RATIO_TABLE}

DRIFT_TABLE = [
    ("OFF",  0),
    ("÷64", 1/64), ("÷32", 1/32), ("÷16", 1/16),
    ("÷8",  1/8),  ("÷6",  1/6),  ("÷4",  1/4),
    ("÷3",  1/3),  ("÷2",  1/2),  ("÷1.5", 2/3),
    ("×1",  1),    ("×1.5", 3/2), ("×2",  2),
    ("×3",  3),    ("×4",  4),    ("×6",  6),
    ("×8",  8),    ("×16", 16),   ("×32", 32),
    ("×64", 64),
]
DRIFT_LABELS = [d[0] for d in DRIFT_TABLE]
DRIFT_VALUES = {d[0]: d[1] for d in DRIFT_TABLE}

TIMEBASE_TABLE = [
    ("÷16", 1/16), ("÷8", 1/8), ("÷4", 1/4),
    ("÷2",  1/2),  ("÷1.5", 2/3),
    ("×1",  1.0),  ("×1.5", 3/2),
    ("×2",  2.0),  ("×4", 4.0),
]
TIMEBASE_LABELS = [t[0] for t in TIMEBASE_TABLE]
TIMEBASE_VALUES = {t[0]: t[1] for t in TIMEBASE_TABLE}

SHAPES = ["Sine", "Square", "Triangle", "Sawtooth", "Noise"]
MOD_TYPES = ["AM", "FM"]
AUDIO_MOD_TYPES = ["AM", "FM"]
REF_H = 480
FPS = 60
FRAME_DUR = 1.0 / FPS
AUDIO_SR = 44100

DOWNLOADS = str(pathlib.Path.home() / "Downloads")


# ─── Audio Envelope Extraction ───────────────────────────────────────────────

def load_audio_samples(path):
    """Load audio file, return mono float samples normalised to [-1, 1] at 44100 Hz."""
    try:
        if HAS_PYDUB:
            seg = AudioSegment.from_file(path)
            seg = seg.set_frame_rate(AUDIO_SR).set_channels(1).set_sample_width(2)
            raw = np.array(seg.get_array_of_samples(), dtype=np.float64)
            return raw / 32768.0
        else:
            with wavmod.open(path, 'rb') as wf:
                nch = wf.getnchannels()
                sw = wf.getsampwidth()
                sr = wf.getframerate()
                n = wf.getnframes()
                raw = wf.readframes(n)
                if sw == 2:
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
                elif sw == 1:
                    samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
                elif sw == 3:
                    # 24-bit: unpack 3-byte little-endian signed integers.
                    # Values >= 2^23 represent negative numbers (two's complement),
                    # so subtract 2^24 to restore the correct sign.
                    raw_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                    int32 = (raw_bytes[:, 0].astype(np.int32)
                             | (raw_bytes[:, 1].astype(np.int32) << 8)
                             | (raw_bytes[:, 2].astype(np.int32) << 16))
                    int32[int32 >= (1 << 23)] -= (1 << 24)
                    samples = int32.astype(np.float64) / (1 << 23)
                elif sw == 4:
                    samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / (1 << 31)
                else:
                    return None
                if nch > 1:
                    samples = samples.reshape(-1, nch).mean(axis=1)
                try:
                    if sr != AUDIO_SR:
                        old_t = np.linspace(0, 1, len(samples))
                        new_n = int(len(samples) * AUDIO_SR / sr)
                        new_t = np.linspace(0, 1, new_n)
                        samples = np.interp(new_t, old_t, samples)
                except Exception:
                    # Resampling failed; return samples at original rate.
                    # Envelope timing may be slightly off but is still usable.
                    pass
                return samples
    except Exception:
        return None


def compute_audio_envelope(samples, t, smoothing):
    """Get audio envelope value at time t. Returns 0..1."""
    if samples is None or len(samples) == 0:
        return 0.0
    centre = int(t * AUDIO_SR)
    half_win = max(1, int(smoothing * AUDIO_SR * 0.5))
    start = max(0, centre - half_win)
    end = min(len(samples), centre + half_win)
    if start >= len(samples):
        return 0.0
    chunk = samples[start:end]
    if len(chunk) == 0:
        return 0.0
    rms = np.sqrt(np.mean(chunk ** 2))
    return min(1.0, rms * 3.0)

# Cache for pre-computed audio cumsum (keyed by id of samples array)
_audio_cumsum_cache: dict = {}


def _get_audio_cumsum(samples):
    """Return cached cumulative sum of squares for the given samples array."""
    key = id(samples)
    if key not in _audio_cumsum_cache:
        sq = samples.astype(np.float64) ** 2
        cs = np.empty(len(samples) + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(sq, out=cs[1:])
        _audio_cumsum_cache[key] = cs
    return _audio_cumsum_cache[key]


def compute_audio_envelope_array(samples, t, smoothing, n_points):
    """
    Compute envelope for an array of points along the vertical axis using a
    sliding window RMS calculation. Creates a waterfall effect where audio
    ripples down the screen.
    """
    if samples is None or len(samples) == 0:
        return np.zeros(n_points)
    n_samples = len(samples)
    spread_sec = smoothing * 2.0
    times = np.linspace(t, t - spread_sec, n_points)
    times = np.clip(times, 0, n_samples / AUDIO_SR)
    centres = (times * AUDIO_SR).astype(np.int64)
    half_win = max(1, int(smoothing * AUDIO_SR * 0.5))

    starts = np.clip(centres - half_win, 0, n_samples)
    ends = np.clip(centres + half_win, 0, n_samples)
    lengths = ends - starts

    cumsum = _get_audio_cumsum(samples)

    window_sums = cumsum[ends] - cumsum[starts]
    valid = lengths > 0
    envelope = np.zeros(n_points)
    envelope[valid] = np.sqrt(window_sums[valid] / lengths[valid]) * 3.0
    np.clip(envelope, 0.0, 1.0, out=envelope)
    return envelope


# ─── Waveform Math ───────────────────────────────────────────────────────────

def generate_wave(shape, phase):
    if shape == "Sine":
        return np.sin(phase)
    elif shape == "Square":
        return np.sign(np.sin(phase))
    elif shape == "Triangle":
        return 2.0 * np.abs(2.0 * (phase / (2*np.pi) - np.floor(phase / (2*np.pi) + 0.5))) - 1.0
    elif shape == "Sawtooth":
        return 2.0 * (phase / (2*np.pi) - np.floor(phase / (2*np.pi) + 0.5))
    elif shape == "Noise":
        pts = max(64, len(phase) // 4)
        sp = np.linspace(0, phase.max() - phase.min(), pts)
        sv = np.random.RandomState(42).uniform(-1, 1, pts)
        return np.interp(phase - phase.min(), sp, sv)
    return np.sin(phase)


def compute_modulator_outputs(y_norm, t, temporal_hz, drift_hz, modulators, xmod_matrix, max_iterations=4):
    n = len(modulators)
    ns = len(y_norm)
    base_phases = []
    for i, m in enumerate(modulators):
        r = m["ratio"]
        base_phases.append(
            2.0 * np.pi * r * y_norm
            + 2.0 * np.pi * temporal_hz * r * t
            + 2.0 * np.pi * drift_hz * r * 0.37 * t
        )
    outputs = [generate_wave(m["shape"], base_phases[i]) * m["amplitude"]
               for i, m in enumerate(modulators)]
    for _ in range(max_iterations):
        new_outputs = []
        for i, m in enumerate(modulators):
            phase = base_phases[i].copy()
            amp_scale = np.ones(ns)
            for j in range(n):
                if i == j or not xmod_matrix[i][j]:
                    continue
                if m["type"] == "FM":
                    phase += m["depth"] * outputs[j]
                else:
                    amp_scale *= 1.0 + m["depth"] * outputs[j]
            new_outputs.append(generate_wave(m["shape"], phase) * amp_scale * m["amplitude"])
        outputs = new_outputs
    return outputs


def compute_waveform(y_norm, t, params, audio_samples=None):
    beat_hz     = params["beat_hz"]
    timebase    = params["timebase"]
    base_ratio  = params["base_ratio"]
    base_shape  = params["base_shape"]
    base_amp    = params["base_amplitude"]
    overdrive   = params["overdrive"]
    drift_ratio = params["drift_ratio"]
    drift_fine  = params["drift_fine"]

    temporal_hz = beat_hz * timebase
    drift_hz    = beat_hz * drift_ratio * drift_fine

    mod_outputs = compute_modulator_outputs(
        y_norm, t, temporal_hz, drift_hz, params["modulators"], params["xmod"])

    spatial  = 2.0 * np.pi * base_ratio * y_norm
    temporal = 2.0 * np.pi * temporal_hz * t
    drift    = 2.0 * np.pi * drift_hz * t
    phase    = spatial + temporal + drift

    fm_sum = np.zeros_like(y_norm)
    am_env = np.ones_like(y_norm)
    for i, m in enumerate(params["modulators"]):
        sig = mod_outputs[i]
        if m["type"] == "FM":
            fm_sum += m["depth"] * sig
        else:
            am_env *= 1.0 + m["depth"] * sig

    # Audio modulator
    audio_mod = params.get("audio_mod")
    if audio_mod and audio_mod["depth"] > 0 and audio_samples is not None:
        smoothing = audio_mod["smoothing"]
        gain = audio_mod["gain"]
        depth = audio_mod["depth"]
        envelope = compute_audio_envelope_array(audio_samples, t, smoothing, len(y_norm))
        envelope = envelope * gain
        if audio_mod["type"] == "FM":
            fm_sum += depth * envelope
        else:
            am_env *= 1.0 + depth * envelope

    phase += fm_sum
    wave = generate_wave(base_shape, phase) * base_amp * am_env
    ceiling = overdrive
    wave = np.clip(wave, -ceiling, ceiling)
    if ceiling > 0:
        wave /= ceiling

    # Audio direct inject
    audio_inject = params.get("audio_inject")
    if audio_inject and audio_inject["amplitude"] > 0 and audio_samples is not None:
        inject_amp = audio_inject["amplitude"]
        n_points = len(y_norm)
        times = np.linspace(t, t - FRAME_DUR, n_points)
        times = np.clip(times, 0, len(audio_samples) / AUDIO_SR)
        indices = (times * AUDIO_SR).astype(np.int64)
        indices = np.clip(indices, 0, len(audio_samples) - 1)
        raw_audio = audio_samples[indices] * inject_amp
        wave += raw_audio

    return wave


# ─── Rendering ────────────────────────────────────────────────────────────────

def compute_param_modulation(t, params, audio_samples=None):
    """
    Compute per-frame scalar modulation offsets for rendering parameters.

    For each target parameter, checks which modulators (M1-M4, Audio) are
    routed to it via param_mod_routing, samples those modulators at y=0.5,
    and returns additive offsets.
    """
    routing = params.get("param_mod_routing")
    if not routing:
        return {}

    any_enabled = any(any(flags) for flags in routing.values())
    if not any_enabled:
        return {}

    beat_hz = params["beat_hz"]
    timebase = params["timebase"]
    drift_ratio = params["drift_ratio"]
    drift_fine = params["drift_fine"]
    temporal_hz = beat_hz * timebase
    drift_hz = beat_hz * drift_ratio * drift_fine

    y_sample = np.array([0.5])

    mod_outputs = compute_modulator_outputs(
        y_sample, t, temporal_hz, drift_hz, params["modulators"], params["xmod"])
    mod_scalars = [float(mod_outputs[i][0]) for i in range(len(params["modulators"]))]

    audio_scalar = 0.0
    audio_mod = params.get("audio_mod")
    if audio_mod and audio_mod["depth"] > 0 and audio_samples is not None:
        envelope_val = compute_audio_envelope(audio_samples, t, audio_mod["smoothing"])
        envelope_val *= audio_mod["gain"]
        audio_scalar = audio_mod["depth"] * min(1.0, envelope_val)

    offsets = {}
    for param_key, flags in routing.items():
        total = 0.0
        # Determine if this is a mod self-modulation case
        self_mod_idx = -1
        if param_key.startswith("mod") and "_" in param_key:
            try:
                self_mod_idx = int(param_key[3:param_key.index("_")])
            except ValueError:
                pass
        for i in range(min(4, len(flags))):
            if flags[i] and i < len(mod_scalars) and i != self_mod_idx:
                m = params["modulators"][i]
                total += m["depth"] * mod_scalars[i]
        if len(flags) > 4 and flags[4]:
            total += audio_scalar
        if total != 0.0:
            offsets[param_key] = total

    return offsets


def render_frame(width, height, t, params, audio_samples=None, trail_buffers=None, output_bgr=False, precomputed=None):
    fg = params["fg_color"]
    bg = params["bg_color"]

    param_mods = compute_param_modulation(t, params, audio_samples)

    line_thickness_mult = params.get("line_thickness", 1.0)
    if "line_thickness" in param_mods:
        line_thickness_mult = max(0.5, line_thickness_mult + param_mods["line_thickness"])

    glow_intensity = params.get("line_glow", 1.0)
    if "line_glow" in param_mods:
        glow_intensity = max(0.0, glow_intensity + param_mods["line_glow"])

    bloom_amount = params.get("line_bloom", 0.0)
    if "line_bloom" in param_mods:
        bloom_amount = max(0.0, bloom_amount + param_mods["line_bloom"])

    trail_amount = params.get("line_trail", 0.0)
    if "line_trail" in param_mods:
        trail_amount = float(np.clip(trail_amount + param_mods["line_trail"], 0.0, 0.95))  # 0.95: avoid infinite persistence

    render_params = params
    if "overdrive" in param_mods or "amplitude" in param_mods:
        render_params = dict(params)
        if "overdrive" in param_mods:
            render_params["overdrive"] = max(0.1, params["overdrive"] + param_mods["overdrive"])  # 0.1: avoid div-by-zero
        if "amplitude" in param_mods:
            render_params["base_amplitude"] = max(0.0, min(1.0, params["base_amplitude"] + param_mods["amplitude"]))
    if any(f"mod{i}_depth" in param_mods or f"mod{i}_amp" in param_mods for i in range(4)):
        if render_params is params:
            render_params = dict(params)
        render_params["modulators"] = [dict(m) for m in params["modulators"]]
        for i in range(min(4, len(render_params["modulators"]))):
            if f"mod{i}_depth" in param_mods:
                render_params["modulators"][i]["depth"] = max(0.0,
                    params["modulators"][i]["depth"] + param_mods[f"mod{i}_depth"])
            if f"mod{i}_amp" in param_mods:
                render_params["modulators"][i]["amplitude"] = max(0.0,
                    params["modulators"][i]["amplitude"] + param_mods[f"mod{i}_amp"])
    if precomputed is not None:
        y_norm = precomputed["y_norm"]
    else:
        y_norm = np.linspace(0, 1, height)
    wave = compute_waveform(y_norm, t, render_params, audio_samples)
    margin = width * 0.04
    centre = width / 2.0
    usable = (width - 2 * margin) / 2.0
    xs = centre + wave * usable
    img = np.full((height, width, 3), (bg[2], bg[1], bg[0]), dtype=np.uint8)
    scale = height / REF_H
    thickness = max(1, round(scale * line_thickness_mult))
    if precomputed is not None:
        y_arange = precomputed["y_arange"]
    else:
        y_arange = np.arange(height, dtype=np.float32)
    pts = np.stack([xs.astype(np.float32), y_arange], axis=1)
    pts = pts.reshape((-1, 1, 2))
    pts_fixed = np.round(pts * 16).astype(np.int32)
    fg_bgr = (int(fg[2]), int(fg[1]), int(fg[0]))

    # Glow layers (controlled by glow_intensity)
    cv2.polylines(img, [pts_fixed], False, fg_bgr, thickness, cv2.LINE_AA, shift=4)
    if thickness >= 2 and glow_intensity > 0:
        glow = np.zeros_like(img)
        cv2.polylines(glow, [pts_fixed], False, fg_bgr, thickness * 3, cv2.LINE_AA, shift=4)
        cv2.addWeighted(img, 1.0, glow, 0.3 * glow_intensity, 0, img)
        glow[:] = 0
        cv2.polylines(glow, [pts_fixed], False, fg_bgr, thickness * 2, cv2.LINE_AA, shift=4)
        cv2.addWeighted(img, 1.0, glow, 0.25 * glow_intensity, 0, img)
    cv2.polylines(img, [pts_fixed], False, fg_bgr, thickness, cv2.LINE_AA, shift=4)
    if thickness >= 3:
        core_bgr = (min(255, int(fg[2])+120), min(255, int(fg[1])+120), min(255, int(fg[0])+120))
        cv2.polylines(img, [pts_fixed], False, core_bgr, max(1, thickness//3), cv2.LINE_AA, shift=4)

    # Bloom (Gaussian blur glow)
    if bloom_amount > 0:
        ksize = max(3, int(scale * bloom_amount * 30) | 1)  # ensure odd
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        cv2.addWeighted(img, 1.0, blurred, bloom_amount * 0.4, 0, img)

    # Afterglow / Trail (phosphor persistence)
    if trail_amount > 0 and trail_buffers is not None:
        key = (width, height)
        if key in trail_buffers:
            prev = trail_buffers[key]
            img = cv2.addWeighted(img, 1.0, prev, trail_amount, 0)
        trail_buffers[key] = img.copy()

    if not output_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def render_frame_gpu(width, height, t, params, audio_samples=None, trail_buffers=None, output_bgr=False):
    """GPU-accelerated version of render_frame using cv2.cuda when HAS_CUDA is True.

    The waveform/numpy computation stays on CPU; all OpenCV post-processing
    (glow blend, bloom, trail, colour conversion) runs on the GPU.
    Falls back to render_frame() transparently if CUDA is unavailable.
    """
    if not HAS_CUDA:
        return render_frame(width, height, t, params, audio_samples, trail_buffers, output_bgr)

    fg = params["fg_color"]
    bg = params["bg_color"]

    param_mods = compute_param_modulation(t, params, audio_samples)

    line_thickness_mult = params.get("line_thickness", 1.0)
    if "line_thickness" in param_mods:
        line_thickness_mult = max(0.5, line_thickness_mult + param_mods["line_thickness"])

    glow_intensity = params.get("line_glow", 1.0)
    if "line_glow" in param_mods:
        glow_intensity = max(0.0, glow_intensity + param_mods["line_glow"])

    bloom_amount = params.get("line_bloom", 0.0)
    if "line_bloom" in param_mods:
        bloom_amount = max(0.0, bloom_amount + param_mods["line_bloom"])

    trail_amount = params.get("line_trail", 0.0)
    if "line_trail" in param_mods:
        trail_amount = float(np.clip(trail_amount + param_mods["line_trail"], 0.0, 0.95))

    render_params = params
    if "overdrive" in param_mods or "amplitude" in param_mods:
        render_params = dict(params)
        if "overdrive" in param_mods:
            render_params["overdrive"] = max(0.1, params["overdrive"] + param_mods["overdrive"])
        if "amplitude" in param_mods:
            render_params["base_amplitude"] = max(0.0, min(1.0, params["base_amplitude"] + param_mods["amplitude"]))
    if any(f"mod{i}_depth" in param_mods or f"mod{i}_amp" in param_mods for i in range(4)):
        if render_params is params:
            render_params = dict(params)
        render_params["modulators"] = [dict(m) for m in params["modulators"]]
        for i in range(min(4, len(render_params["modulators"]))):
            if f"mod{i}_depth" in param_mods:
                render_params["modulators"][i]["depth"] = max(0.0,
                    params["modulators"][i]["depth"] + param_mods[f"mod{i}_depth"])
            if f"mod{i}_amp" in param_mods:
                render_params["modulators"][i]["amplitude"] = max(0.0,
                    params["modulators"][i]["amplitude"] + param_mods[f"mod{i}_amp"])

    y_norm = np.linspace(0, 1, height)
    wave = compute_waveform(y_norm, t, render_params, audio_samples)
    margin = width * 0.04
    centre = width / 2.0
    usable = (width - 2 * margin) / 2.0
    xs = centre + wave * usable
    img = np.full((height, width, 3), (bg[2], bg[1], bg[0]), dtype=np.uint8)
    scale = height / REF_H
    thickness = max(1, round(scale * line_thickness_mult))
    pts = np.stack([xs.astype(np.float32), np.arange(height, dtype=np.float32)], axis=1)
    pts = pts.reshape((-1, 1, 2))
    pts_fixed = np.round(pts * 16).astype(np.int32)
    fg_bgr = (int(fg[2]), int(fg[1]), int(fg[0]))

    # CPU: polylines (no CUDA equivalent)
    cv2.polylines(img, [pts_fixed], False, fg_bgr, thickness, cv2.LINE_AA, shift=4)
    if thickness >= 2 and glow_intensity > 0:
        glow = np.zeros_like(img)
        cv2.polylines(glow, [pts_fixed], False, fg_bgr, thickness * 3, cv2.LINE_AA, shift=4)
        # Upload to GPU for blending
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_glow = cv2.cuda_GpuMat()
        gpu_glow.upload(glow)
        cv2.cuda.addWeighted(gpu_img, 1.0, gpu_glow, 0.3 * glow_intensity, 0, gpu_img)
        glow[:] = 0
        cv2.polylines(glow, [pts_fixed], False, fg_bgr, thickness * 2, cv2.LINE_AA, shift=4)
        gpu_glow.upload(glow)
        cv2.cuda.addWeighted(gpu_img, 1.0, gpu_glow, 0.25 * glow_intensity, 0, gpu_img)
        img = gpu_img.download()
    cv2.polylines(img, [pts_fixed], False, fg_bgr, thickness, cv2.LINE_AA, shift=4)
    if thickness >= 3:
        core_bgr = (min(255, int(fg[2])+120), min(255, int(fg[1])+120), min(255, int(fg[0])+120))
        cv2.polylines(img, [pts_fixed], False, core_bgr, max(1, thickness//3), cv2.LINE_AA, shift=4)

    # GPU: bloom
    if bloom_amount > 0:
        ksize = max(3, int(scale * bloom_amount * 30) | 1)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gauss_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, (ksize, ksize), 0)
        gpu_blurred = gauss_filter.apply(gpu_img)
        cv2.cuda.addWeighted(gpu_img, 1.0, gpu_blurred, bloom_amount * 0.4, 0, gpu_img)
        img = gpu_img.download()

    # GPU: trail
    if trail_amount > 0 and trail_buffers is not None:
        key = (width, height)
        if key in trail_buffers:
            prev = trail_buffers[key]
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_prev = cv2.cuda_GpuMat()
            gpu_prev.upload(prev)
            cv2.cuda.addWeighted(gpu_img, 1.0, gpu_prev, trail_amount, 0, gpu_img)
            img = gpu_img.download()
        trail_buffers[key] = img.copy()

    if not output_bgr:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_img = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2RGB)
        img = gpu_img.download()
    return img


# ─── Export Helpers ──────────────────────────────────────────────────────────

# Per-worker globals set by the pool initializer
_worker_samples = None
_worker_precomputed = None


def _init_export_worker(samples, precomputed):
    """Pool initializer: store audio samples and pre-computed arrays once per worker process."""
    global _worker_samples, _worker_precomputed
    _worker_samples = samples
    _worker_precomputed = precomputed


def _render_frame_worker(args):
    """Top-level picklable worker: render one frame without trail (trail is composited sequentially)."""
    width, height, t, params = args
    return render_frame(width, height, t, params, _worker_samples,
                        trail_buffers=None, output_bgr=True,
                        precomputed=_worker_precomputed)


def _shallow_copy_params(params):
    """Fast copy of params dict — avoids copy.deepcopy overhead."""
    p = dict(params)
    p["modulators"] = [dict(m) for m in params["modulators"]]
    p["xmod"] = [list(row) for row in params["xmod"]]
    if "audio_mod" in params:
        p["audio_mod"] = dict(params["audio_mod"])
    if "audio_inject" in params:
        p["audio_inject"] = dict(params["audio_inject"])
    if "param_mod_routing" in params:
        p["param_mod_routing"] = {k: list(v) for k, v in params["param_mod_routing"].items()}
    return p


# ─── BPM Entry ───────────────────────────────────────────────────────────────

class BPMEntry(tk.Entry):
    def __init__(self, master, linked_var, **kwargs):
        self._linked = linked_var
        super().__init__(master, **kwargs)
        self.insert(0, f"{linked_var.get():.1f}")
        self.bind("<Return>", self._commit)
        self.bind("<KP_Enter>", self._commit)
        self.bind("<FocusOut>", self._commit)
        self.bind("<Escape>", self._revert)
        self._linked.trace_add("write", self._on_slider)
        self._editing = False
        self.bind("<Key>", lambda _: setattr(self, '_editing', True))

    def _commit(self, _e=None):
        self._editing = False
        try:
            v = max(20.0, min(300.0, float(self.get().strip())))
            self._linked.set(v)
            self._set(f"{v:.1f}")
        except ValueError:
            self._set(f"{self._linked.get():.1f}")
        return "break"

    def _revert(self, _e=None):
        self._editing = False
        self._set(f"{self._linked.get():.1f}")
        self.master.focus_set()
        return "break"

    def _on_slider(self, *_):
        if not self._editing:
            self._set(f"{self._linked.get():.1f}")

    def _set(self, txt):
        c = self.index(tk.INSERT)
        self.delete(0, tk.END)
        self.insert(0, txt)
        try:
            self.icursor(min(c, len(txt)))
        except Exception:
            pass


# ─── GUI ──────────────────────────────────────────────────────────────────────

PREVIEW_W, PREVIEW_H = 300, 480


class WaveformApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vertical Oscilloscope v15  ·  9:16")
        self.resizable(False, False)
        self.configure(bg="#111111")

        self.fg_color = (0, 255, 140)
        self.bg_color = (4, 4, 12)
        self.playing = False
        self._t0 = 0.0
        self._elapsed = 0.0
        self._export_cancel = False
        self._audio_path = None
        self._audio_loaded = False
        self._audio_samples = None
        self.xmod_vars: list[list[tk.BooleanVar]] = []
        self.audio_mod_enabled = tk.BooleanVar(value=False)
        self.audio_inject_enabled = tk.BooleanVar(value=False)

        self._automation_data = {}
        self._auto_mode = "off"
        self._overdub_buffer = {}
        self._overdub_start_t = 0.0

        self._trail_buffers = {}

        self._snapshots: list[dict | None] = [None] * 8
        self._active_snapshot = tk.IntVar(value=-1)
        self._snapshots_locked = tk.BooleanVar(value=False)

        self.bar_var = tk.IntVar(value=1)
        self.bar_fine_var = tk.DoubleVar(value=0.0)
        self._bar_tracking = False
        self._audio_dur = 0.0

        self.param_mod_routing = {
            "amplitude":      [tk.BooleanVar(value=False) for _ in range(5)],
            "overdrive":      [tk.BooleanVar(value=False) for _ in range(5)],
            "line_thickness": [tk.BooleanVar(value=False) for _ in range(5)],
            "line_glow":      [tk.BooleanVar(value=False) for _ in range(5)],
            "line_bloom":     [tk.BooleanVar(value=False) for _ in range(5)],
            "line_trail":     [tk.BooleanVar(value=False) for _ in range(5)],
        }
        for i in range(4):
            self.param_mod_routing[f"mod{i}_depth"] = [tk.BooleanVar(value=False) for _ in range(5)]
            self.param_mod_routing[f"mod{i}_amp"]   = [tk.BooleanVar(value=False) for _ in range(5)]

        self._build_ui()
        self._active_snapshot.trace_add("write", self._on_snapshot_select)
        self._tick()

    def _build_ui(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        for w in ("TLabel", "TFrame", "TLabelframe", "TScale", "TCheckbutton"):
            s.configure(w, background="#111111")
        s.configure("TLabel", foreground="#cccccc", font=("Consolas", 9))
        s.configure("TCheckbutton", foreground="#cccccc", font=("Consolas", 8))
        s.configure("Route.TCheckbutton", foreground="#888888", font=("Consolas", 7), background="#111111")
        s.configure("Header.TLabel", foreground="#ffffff", font=("Consolas", 11, "bold"), background="#111111")
        s.configure("Hz.TLabel", foreground="#66bbff", font=("Consolas", 8), background="#111111")
        s.configure("OD.TLabel", foreground="#ff5555", font=("Consolas", 9, "bold"), background="#111111")
        s.configure("TB.TLabel", foreground="#cc88ff", font=("Consolas", 9, "bold"), background="#111111")
        s.configure("Drift.TLabel", foreground="#ffaa44", font=("Consolas", 9, "bold"), background="#111111")
        s.configure("Audio.TLabel", foreground="#44ff88", font=("Consolas", 8), background="#111111")
        s.configure("AudioMod.TLabel", foreground="#ff88ff", font=("Consolas", 9, "bold"), background="#111111")
        s.configure("Time.TLabel", foreground="#ffcc00", font=("Consolas", 12, "bold"), background="#111111")
        s.configure("Rec.TLabel", foreground="#ff3333", font=("Consolas", 10, "bold"), background="#111111")
        s.configure("Meter.TLabel", foreground="#44ff88", font=("Consolas", 9), background="#111111")
        s.configure("TLabelframe.Label", foreground="#888888", font=("Consolas", 9, "bold"), background="#111111")
        s.configure("TButton", font=("Consolas", 9))
        s.configure("Bar.TLabel", foreground="#88ffcc", font=("Consolas", 9, "bold"), background="#111111")
        s.map("TCheckbutton", background=[("active", "#222222")])
        s.configure("Snap.TRadiobutton", background="#111111", foreground="#cccccc", font=("Consolas", 8))

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=6, pady=6)

        ctrl = ttk.Frame(root)
        ctrl.pack(side="left", fill="both", expand=True, padx=(0, 6))

        col_left = ttk.Frame(ctrl)
        col_left.pack(side="left", fill="y", padx=(0, 4))
        col_right = ttk.Frame(ctrl)
        col_right.pack(side="left", fill="y", padx=(4, 0))

        # ═══ LEFT COLUMN ═════════════════════════════════════════════════

        # ── Transport ────────────────────────────────────────────────────
        tp = ttk.LabelFrame(col_left, text="TRANSPORT")
        tp.pack(fill="x", pady=(0, 4))

        br = ttk.Frame(tp)
        br.pack(fill="x", padx=4, pady=3)
        self.play_btn = ttk.Button(br, text="▶  PLAY", command=self._play, width=10)
        self.play_btn.pack(side="left", padx=(0, 4))
        ttk.Button(br, text="⏹  STOP", command=self._stop, width=10).pack(side="left", padx=(0, 4))
        self.time_lbl = ttk.Label(br, text="00:00.000", style="Time.TLabel")
        self.time_lbl.pack(side="left", padx=4)

        bar_row = ttk.Frame(tp)
        bar_row.pack(fill="x", padx=4, pady=(0, 2))
        ttk.Label(bar_row, text="Bar").pack(side="left", padx=(0, 4))
        self.bar_slider = ttk.Scale(bar_row, from_=1, to=999, variable=self.bar_var,
                                    orient="horizontal", length=120,
                                    command=self._on_bar_seek)
        self.bar_slider.pack(side="left", padx=(0, 4))
        self.bar_lbl = ttk.Label(bar_row, text="Bar 1", style="Bar.TLabel", width=7)
        self.bar_lbl.pack(side="left", padx=(0, 6))
        self.beat_lbl = ttk.Label(bar_row, text="Beat 1.0", style="Hz.TLabel", width=9)
        self.beat_lbl.pack(side="left", padx=(0, 6))
        ttk.Label(bar_row, text="Fine").pack(side="left", padx=(0, 2))
        self.bar_fine_slider = ttk.Scale(bar_row, from_=0.0, to=1.0,
                                         variable=self.bar_fine_var,
                                         orient="horizontal", length=60,
                                         command=self._on_bar_seek)
        self.bar_fine_slider.pack(side="left", padx=(0, 2))
        self.bar_fine_lbl = ttk.Label(bar_row, text="0.00", width=5)
        self.bar_fine_lbl.pack(side="left")

        tbr = ttk.Frame(tp)
        tbr.pack(fill="x", padx=4, pady=(0, 3))
        ttk.Label(tbr, text="Timebase").pack(side="left", padx=(0, 4))
        self.tb_preset_var = tk.StringVar(value="×1")
        tbc = ttk.Combobox(tbr, textvariable=self.tb_preset_var, values=TIMEBASE_LABELS,
                           state="readonly", width=5)
        tbc.pack(side="left", padx=(0, 4))
        tbc.bind("<<ComboboxSelected>>", self._on_tb_preset)
        self.timebase_var = tk.DoubleVar(value=1.0)
        ttk.Scale(tbr, from_=0.01, to=4.0, variable=self.timebase_var,
                  orient="horizontal", length=100).pack(side="left", padx=(0, 4))
        self.tb_lbl = ttk.Label(tbr, text="1.00×", style="TB.TLabel", width=6)
        self.tb_lbl.pack(side="left")
        self.tb_hz_lbl = ttk.Label(tp, text="", style="Hz.TLabel")
        self.tb_hz_lbl.pack(fill="x", padx=4, pady=(0, 2))

        # ── Audio ────────────────────────────────────────────────────────
        af = ttk.LabelFrame(col_left, text="AUDIO")
        af.pack(fill="x", pady=(0, 4))

        ar = ttk.Frame(af)
        ar.pack(fill="x", padx=4, pady=3)
        ttk.Button(ar, text="Load Audio…", command=self._load_audio).pack(side="left", padx=(0, 4))
        ttk.Button(ar, text="Clear", command=self._clear_audio).pack(side="left")
        self.audio_lbl = ttk.Label(af,
            text="No audio loaded" if HAS_PYGAME else "pip install pygame",
            style="Audio.TLabel")
        self.audio_lbl.pack(fill="x", padx=4, pady=(0, 2))
        self.audio_meter_lbl = ttk.Label(af, text="Level: ────────────────────",
                                         style="Meter.TLabel")
        self.audio_meter_lbl.pack(fill="x", padx=4, pady=(0, 3))

        # ── BPM ──────────────────────────────────────────────────────────
        bpm_f = ttk.LabelFrame(col_left, text="MASTER CLOCK")
        bpm_f.pack(fill="x", pady=(0, 4))

        bpmr = ttk.Frame(bpm_f)
        bpmr.pack(fill="x", padx=4, pady=2)
        ttk.Label(bpmr, text="BPM").pack(side="left", padx=(0, 4))
        self.bpm_var = tk.DoubleVar(value=120.0)
        ttk.Scale(bpmr, from_=20, to=300, variable=self.bpm_var,
                  orient="horizontal", length=120).pack(side="left", padx=(0, 4))
        self.bpm_entry = BPMEntry(bpmr, self.bpm_var, width=7,
                                  font=("Consolas", 12, "bold"), justify="center",
                                  relief="sunken", bd=2)
        self.bpm_entry.pack(side="left", padx=(0, 4))
        self._tap_times: list[float] = []
        ttk.Button(bpmr, text="⏎ Tap", command=self._tap_tempo).pack(side="left")
        self.bpm_hz_lbl = ttk.Label(bpm_f, text="", style="Hz.TLabel")
        self.bpm_hz_lbl.pack(fill="x", padx=4, pady=(0, 2))

        # ── Base Waveform ────────────────────────────────────────────────
        bf = ttk.LabelFrame(col_left, text="BASE WAVEFORM")
        bf.pack(fill="x", pady=(0, 4))

        bwr = ttk.Frame(bf)
        bwr.pack(fill="x", padx=4, pady=2)
        ttk.Label(bwr, text="Shape").pack(side="left", padx=(0, 4))
        self.base_shape = tk.StringVar(value="Sine")
        ttk.Combobox(bwr, textvariable=self.base_shape, values=SHAPES,
                     state="readonly", width=9).pack(side="left", padx=(0, 8))
        ttk.Label(bwr, text="Ratio").pack(side="left", padx=(0, 4))
        self.base_ratio_var = tk.StringVar(value="×4")
        ttk.Combobox(bwr, textvariable=self.base_ratio_var, values=RATIO_LABELS,
                     state="readonly", width=7).pack(side="left", padx=(0, 4))
        self.base_hz_lbl = ttk.Label(bwr, text="", style="Hz.TLabel")
        self.base_hz_lbl.pack(side="left")

        self._add_slider(bf, "Amplitude", 0.0, 1.0, 0.4, "amplitude", mod_route_key="amplitude")
        self._add_slider(bf, "Mod OD", 1.0, 5.0, 3.0, "overdrive", mod_route_key="overdrive")

        # Drift
        dr = ttk.Frame(bf)
        dr.pack(fill="x", padx=4, pady=2)
        ttk.Label(dr, text="Drift").pack(side="left", padx=(0, 4))
        self.drift_ratio_var = tk.StringVar(value="÷4")
        ttk.Combobox(dr, textvariable=self.drift_ratio_var, values=DRIFT_LABELS,
                     state="readonly", width=5).pack(side="left", padx=(0, 4))
        ttk.Label(dr, text="Fine").pack(side="left", padx=(0, 4))
        self.drift_fine_var = tk.DoubleVar(value=1.0)
        ttk.Scale(dr, from_=0.1, to=3.0, variable=self.drift_fine_var,
                  orient="horizontal", length=100).pack(side="left", padx=(0, 4))
        self.drift_fine_lbl = ttk.Label(dr, text="1.00×", width=5)
        self.drift_fine_lbl.pack(side="left")
        self.drift_info_lbl = ttk.Label(bf, text="", style="Drift.TLabel")
        self.drift_info_lbl.pack(fill="x", padx=4, pady=(0, 2))

        # ── Colors ───────────────────────────────────────────────────────
        cf = ttk.LabelFrame(col_left, text="COLORS")
        cf.pack(fill="x", pady=(0, 4))
        cr = ttk.Frame(cf)
        cr.pack(fill="x", padx=4, pady=3)
        self.fg_btn = tk.Button(cr, text="  FG  ", bg=self._hex(self.fg_color),
                                command=self._pick_fg, relief="flat", font=("Consolas", 9))
        self.fg_btn.pack(side="left", padx=(0, 8))
        self.bg_btn = tk.Button(cr, text="  BG  ", bg=self._hex(self.bg_color),
                                command=self._pick_bg, relief="flat", font=("Consolas", 9))
        self.bg_btn.pack(side="left")

        # ── Line Style ───────────────────────────────────────────────────
        lsf = ttk.LabelFrame(col_left, text="LINE STYLE")
        lsf.pack(fill="x", pady=(0, 4))
        self._add_slider(lsf, "Thickness", 0.5, 6.0, 1.0, "line_thickness", mod_route_key="line_thickness")
        self._add_slider(lsf, "Glow",      0.0, 2.0, 1.0, "line_glow", mod_route_key="line_glow")
        self._add_slider(lsf, "Bloom",     0.0, 3.0, 0.0, "line_bloom", mod_route_key="line_bloom")
        self._add_slider(lsf, "Trail",     0.0, 0.95, 0.0, "line_trail", mod_route_key="line_trail")

        # ── Patch ────────────────────────────────────────────────────────
        pf = ttk.LabelFrame(col_left, text="PATCH")
        pf.pack(fill="x", pady=(0, 4))
        pr = ttk.Frame(pf)
        pr.pack(fill="x", padx=4, pady=3)
        ttk.Button(pr, text="💾 Save…", command=self._save_patch).pack(side="left", padx=(0, 8))
        ttk.Button(pr, text="📂 Load…", command=self._load_patch).pack(side="left")

        # ── Snapshots ─────────────────────────────────────────────────────
        sf = ttk.LabelFrame(col_left, text="SNAPSHOTS")
        sf.pack(fill="x", pady=(0, 4))

        # Lock row
        lock_row = ttk.Frame(sf)
        lock_row.pack(fill="x", padx=4, pady=(2, 0))
        self._snap_lock_btn = tk.Button(
            lock_row, text="🔓 UNLOCKED",
            bg="#333333", fg="#cccccc", activebackground="#333333",
            relief="raised", bd=1, padx=4, pady=1,
            command=self._toggle_snap_lock, font=("Consolas", 8))
        self._snap_lock_btn.pack(side="left")

        self._snapshot_btns: list[ttk.Button] = []
        self._snapshot_radios: list[ttk.Radiobutton] = []

        for row_start in (0, 4):
            btn_row = ttk.Frame(sf)
            btn_row.pack(fill="x", padx=4, pady=(2, 0))
            radio_row = ttk.Frame(sf)
            radio_row.pack(fill="x", padx=4, pady=(0, 2))

            for i in range(row_start, row_start + 4):
                idx = i
                btn = ttk.Button(btn_row, text=str(i + 1), width=4,
                                 command=lambda ii=idx: self._snapshot_store(ii))
                btn.pack(side="left", padx=2, expand=True)
                self._snapshot_btns.append(btn)

                rb = ttk.Radiobutton(radio_row, text="", variable=self._active_snapshot,
                                     value=i, style="Snap.TRadiobutton")
                rb.pack(side="left", padx=2, expand=True)
                self._snapshot_radios.append(rb)

        # ═══ RIGHT COLUMN ════════════════════════════════════════════════

        # ── Modulators 1–4 ───────────────────────────────────────────────
        self.mod_vars: list[dict] = []
        self.xmod_vars = []
        for i in range(4):
            mf = ttk.LabelFrame(col_right, text=f"MOD {i+1}")
            mf.pack(fill="x", pady=(0, 3))
            mv: dict = {}

            r0 = ttk.Frame(mf)
            r0.pack(fill="x", padx=2, pady=1)
            mv["type"] = tk.StringVar(value="FM" if i % 2 == 0 else "AM")
            ttk.Combobox(r0, textvariable=mv["type"], values=MOD_TYPES,
                         state="readonly", width=4).pack(side="left", padx=2)
            mv["shape"] = tk.StringVar(value="Sine")
            ttk.Combobox(r0, textvariable=mv["shape"], values=SHAPES,
                         state="readonly", width=7).pack(side="left", padx=2)
            ttk.Label(r0, text="Ratio").pack(side="left", padx=(6, 2))
            mv["ratio"] = tk.StringVar(value="×1")
            ttk.Combobox(r0, textvariable=mv["ratio"], values=RATIO_LABELS,
                         state="readonly", width=5).pack(side="left", padx=2)
            mv["hz_lbl"] = ttk.Label(r0, text="", style="Hz.TLabel")
            mv["hz_lbl"].pack(side="left", padx=4)

            self._mod_slider(mf, "Dpt", 0.0, 5.0, 0.0, mv, "depth",
                             mod_route_key=f"mod{i}_depth", disable_index=i)
            self._mod_slider(mf, "Amp", 0.0, 3.0, 1.0, mv, "amp",
                             mod_route_key=f"mod{i}_amp", disable_index=i)

            xrow = ttk.Frame(mf)
            xrow.pack(fill="x", padx=2, pady=(0, 2))
            ttk.Label(xrow, text="XMod ←", font=("Consolas", 8)).pack(side="left", padx=(0, 4))
            row_vars = []
            for j in range(4):
                bv = tk.BooleanVar(value=False)
                ttk.Checkbutton(xrow, text=f"M{j+1}", variable=bv,
                                state="disabled" if j == i else "!disabled").pack(side="left", padx=2)
                row_vars.append(bv)
            self.xmod_vars.append(row_vars)

            self.mod_vars.append(mv)

        # ── Audio Modulator ──────────────────────────────────────────────
        amf = ttk.LabelFrame(col_right, text="AUDIO MOD (from loaded audio)")
        amf.pack(fill="x", pady=(0, 4))

        am_r0 = ttk.Frame(amf)
        am_r0.pack(fill="x", padx=4, pady=2)
        ttk.Label(am_r0, text="Type").pack(side="left", padx=(0, 4))
        self.audio_mod_type = tk.StringVar(value="AM")
        ttk.Combobox(am_r0, textvariable=self.audio_mod_type, values=AUDIO_MOD_TYPES,
                     state="readonly", width=4).pack(side="left", padx=(0, 8))
        self.audio_mod_btn = tk.Button(
            am_r0, text="⊘ CONNECT",
            bg="#333333", fg="#cccccc", activebackground="#333333",
            relief="raised", bd=1, padx=4, pady=1,
            command=self._toggle_audio_mod)
        self.audio_mod_btn.pack(side="left", padx=(0, 6))
        self.audio_mod_info = ttk.Label(am_r0, text="", style="AudioMod.TLabel")
        self.audio_mod_info.pack(side="left", padx=4)

        self._audio_mod_slider(amf, "Depth", 0.0, 5.0, 1.0, "audio_mod_depth")
        self._audio_mod_slider(amf, "Gain", 0.1, 5.0, 1.0, "audio_mod_gain")
        self._audio_mod_slider(amf, "Smooth", 0.005, 0.5, 0.05, "audio_mod_smoothing")

        ttk.Label(amf, text="Smooth = envelope speed · Gain = input boost",
                  font=("Consolas", 7), foreground="#666666",
                  background="#111111").pack(fill="x", padx=4, pady=(0, 2))

        # Audio inject row
        ai_r0 = ttk.Frame(amf)
        ai_r0.pack(fill="x", padx=4, pady=2)
        self.audio_inject_btn = tk.Button(
            ai_r0, text="⊘ INJECT",
            bg="#333333", fg="#cccccc", activebackground="#333333",
            relief="raised", bd=1, padx=4, pady=1,
            command=self._toggle_audio_inject)
        self.audio_inject_btn.pack(side="left", padx=(0, 6))
        self.audio_inject_info = ttk.Label(ai_r0, text="", style="AudioMod.TLabel")
        self.audio_inject_info.pack(side="left", padx=4)

        self._audio_mod_slider(amf, "Inject Amp", 0.0, 1.0, 0.0, "audio_inject_amp")

        # ── Automation ──────────────────────────────────────────────────
        af = ttk.LabelFrame(col_right, text="AUTOMATION")
        af.pack(fill="x", pady=(0, 4))

        ar0 = ttk.Frame(af)
        ar0.pack(fill="x", padx=4, pady=3)

        self.auto_write_btn = tk.Button(
            ar0, text="✎ WRITE", bg="#333333", fg="#cccccc",
            activebackground="#333333", relief="raised", bd=1,
            padx=4, pady=1, command=self._auto_write, width=8)
        self.auto_write_btn.pack(side="left", padx=(0, 4))

        self.auto_overdub_btn = tk.Button(
            ar0, text="⊕ OVERDUB", bg="#333333", fg="#cccccc",
            activebackground="#333333", relief="raised", bd=1,
            padx=4, pady=1, command=self._auto_overdub, width=9)
        self.auto_overdub_btn.pack(side="left", padx=(0, 4))

        self.auto_reset_btn = ttk.Button(
            ar0, text="↺ RESET", command=self._auto_reset, width=8)
        self.auto_reset_btn.pack(side="left")

        self.auto_status_lbl = ttk.Label(af, text="No automation",
            style="Rec.TLabel")
        self.auto_status_lbl.pack(fill="x", padx=4, pady=(0, 3))

        # ── Offline Export ───────────────────────────────────────────────
        ef = ttk.LabelFrame(col_right, text="OFFLINE EXPORT (video only)")
        ef.pack(fill="x", pady=(0, 4))

        er0 = ttk.Frame(ef)
        er0.pack(fill="x", padx=4, pady=2)
        ttk.Label(er0, text="Sec").pack(side="left", padx=(0, 4))
        self.duration_var = tk.IntVar(value=5)
        ttk.Spinbox(er0, from_=1, to=500, textvariable=self.duration_var,
                    width=4).pack(side="left", padx=(0, 8))
        ttk.Label(er0, text="Fmt").pack(side="left", padx=(0, 4))
        self.format_var = tk.StringVar(value="MP4")
        ttk.Combobox(er0, textvariable=self.format_var, values=["MP4", "AVI"],
                     state="readonly", width=5).pack(side="left", padx=(0, 8))
        ttk.Label(er0, text="Res").pack(side="left", padx=(0, 4))
        self.res_var = tk.StringVar(value="1080×1920")
        ttk.Combobox(er0, textvariable=self.res_var,
                     values=["720×1280", "1080×1920", "4K (2160×3840)"],
                     state="readonly", width=14).pack(side="left")

        er1 = ttk.Frame(ef)
        er1.pack(fill="x", padx=4, pady=3)
        self.export_btn = ttk.Button(er1, text="Export…", command=self._export)
        self.export_btn.pack(side="left", padx=(0, 8))
        self.cancel_btn = ttk.Button(er1, text="Cancel", command=self._cancel_export,
                                     state="disabled")
        self.cancel_btn.pack(side="left")
        gpu_text = "🟢 CUDA" if HAS_CUDA else "⚪ CPU"
        ttk.Label(er1, text=gpu_text, font=("Consolas", 8),
                  foreground="#888888", background="#111111").pack(side="left", padx=(10, 0))

        self.progress = tk.DoubleVar(value=0)
        ttk.Progressbar(ef, variable=self.progress, maximum=100,
                        length=340).pack(fill="x", padx=4, pady=2)
        self.progress_lbl = ttk.Label(ef, text="", font=("Consolas", 8),
                                       foreground="#888888", background="#111111")
        self.progress_lbl.pack(fill="x", padx=4, pady=(0, 2))

        # ── Preview ──────────────────────────────────────────────────────
        pvf = ttk.Frame(root)
        pvf.pack(side="right", fill="both")
        ttk.Label(pvf, text="PREVIEW", style="Header.TLabel").pack(pady=(0, 2))
        self.canvas = tk.Canvas(pvf, width=PREVIEW_W, height=PREVIEW_H,
                                bg="black", highlightthickness=0)
        self.canvas.pack()

    # ── Slider helpers ───────────────────────────────────────────────────

    def _add_slider(self, parent, label, lo, hi, default, attr, mod_route_key=None):
        r = ttk.Frame(parent)
        r.pack(fill="x", padx=4, pady=1)
        ttk.Label(r, text=label, width=9).pack(side="left")
        var = tk.DoubleVar(value=default)
        setattr(self, attr, var)
        ttk.Scale(r, from_=lo, to=hi, variable=var,
                  orient="horizontal", length=120).pack(side="left", padx=(0, 4))
        lbl = ttk.Label(r, text=f"{default:.2f}", width=5)
        lbl.pack(side="left")
        setattr(self, f"_{attr}_lbl", lbl)

        if mod_route_key and mod_route_key in self.param_mod_routing:
            ttk.Label(r, text="│", font=("Consolas", 7), foreground="#444444",
                      background="#111111").pack(side="left", padx=(2, 2))
            route_vars = self.param_mod_routing[mod_route_key]
            for idx, name in enumerate(["M1", "M2", "M3", "M4", "Au"]):
                ttk.Checkbutton(r, text=name, variable=route_vars[idx],
                                style="Route.TCheckbutton").pack(side="left", padx=1)

    def _mod_slider(self, parent, label, lo, hi, default, store, key, mod_route_key=None, disable_index=None):
        r = ttk.Frame(parent)
        r.pack(fill="x", padx=2, pady=1)
        ttk.Label(r, text=label, width=4).pack(side="left")
        var = tk.DoubleVar(value=default)
        store[key] = var
        ttk.Scale(r, from_=lo, to=hi, variable=var,
                  orient="horizontal", length=140).pack(side="left", padx=(0, 4))
        lbl = ttk.Label(r, text=f"{default:.2f}", width=5)
        lbl.pack(side="left")
        store[f"{key}_lbl"] = lbl

        if mod_route_key and mod_route_key in self.param_mod_routing:
            ttk.Label(r, text="│", font=("Consolas", 7), foreground="#444444",
                      background="#111111").pack(side="left", padx=(2, 2))
            route_vars = self.param_mod_routing[mod_route_key]
            for idx, name in enumerate(["M1", "M2", "M3", "M4", "Au"]):
                state = "disabled" if idx == disable_index else "!disabled"
                ttk.Checkbutton(r, text=name, variable=route_vars[idx],
                                style="Route.TCheckbutton", state=state).pack(side="left", padx=1)

    def _audio_mod_slider(self, parent, label, lo, hi, default, attr):
        r = ttk.Frame(parent)
        r.pack(fill="x", padx=4, pady=1)
        ttk.Label(r, text=label, width=7).pack(side="left")
        var = tk.DoubleVar(value=default)
        setattr(self, attr, var)
        ttk.Scale(r, from_=lo, to=hi, variable=var,
                  orient="horizontal", length=160).pack(side="left", padx=(0, 4))
        lbl = ttk.Label(r, text=f"{default:.3f}", width=6)
        lbl.pack(side="left")
        setattr(self, f"_{attr}_lbl", lbl)

    def _toggle_audio_mod(self):
        self.audio_mod_enabled.set(not self.audio_mod_enabled.get())
        self._update_audio_mod_btn()

    def _update_audio_mod_btn(self):
        if self.audio_mod_enabled.get():
            self.audio_mod_btn.configure(
                text="● CONNECTED", bg="#226622", fg="#aaffaa",
                activebackground="#226622")
        else:
            self.audio_mod_btn.configure(
                text="⊘ CONNECT", bg="#333333", fg="#cccccc",
                activebackground="#333333")

    def _toggle_audio_inject(self):
        self.audio_inject_enabled.set(not self.audio_inject_enabled.get())
        self._update_audio_inject_btn()

    def _update_audio_inject_btn(self):
        if self.audio_inject_enabled.get():
            self.audio_inject_btn.configure(
                text="● INJECTED", bg="#226622", fg="#aaffaa",
                activebackground="#226622")
        else:
            self.audio_inject_btn.configure(
                text="⊘ INJECT", bg="#333333", fg="#cccccc",
                activebackground="#333333")

    # ── Color helpers ────────────────────────────────────────────────────

    @staticmethod
    def _hex(c):
        return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

    def _pick_fg(self):
        c = colorchooser.askcolor(initialcolor=self._hex(self.fg_color), title="FG")
        if c and c[0]:
            self.fg_color = tuple(int(v) for v in c[0])
            self.fg_btn.configure(bg=self._hex(self.fg_color))

    def _pick_bg(self):
        c = colorchooser.askcolor(initialcolor=self._hex(self.bg_color), title="BG")
        if c and c[0]:
            self.bg_color = tuple(int(v) for v in c[0])
            self.bg_btn.configure(bg=self._hex(self.bg_color))

    # ── BPM / Tap ────────────────────────────────────────────────────────

    def _tap_tempo(self):
        now = time.time()
        if self._tap_times and (now - self._tap_times[-1]) > 3.0:
            self._tap_times.clear()
        self._tap_times.append(now)
        if len(self._tap_times) > 8:
            self._tap_times = self._tap_times[-8:]
        if len(self._tap_times) >= 2:
            intervals = [self._tap_times[k+1] - self._tap_times[k]
                         for k in range(len(self._tap_times) - 1)]
            self.bpm_var.set(max(20.0, min(300.0, 60.0 / (sum(intervals) / len(intervals)))))

    def _on_tb_preset(self, _e=None):
        self.timebase_var.set(TIMEBASE_VALUES.get(self.tb_preset_var.get(), 1.0))

    def _on_bar_seek(self, _val=None):
        if self._bar_tracking:
            return
        bpm = self.bpm_var.get()
        bar_duration = 4 * 60.0 / bpm  # 4 beats per bar (4/4 time)
        bar_num = self.bar_var.get()
        fine = self.bar_fine_var.get()
        target_time = (bar_num - 1 + fine) * bar_duration
        self._elapsed = target_time
        if self.playing:
            self._t0 = time.time()
            if HAS_PYGAME and self._audio_loaded:
                try:
                    pygame.mixer.music.play(0, start=target_time)
                except TypeError:
                    pygame.mixer.music.play(0)

    # ── Audio ────────────────────────────────────────────────────────────

    def _load_audio(self, path=None):
        if not HAS_PYGAME:
            messagebox.showwarning("Audio", "pip install pygame")
            return
        if path is None:
            path = filedialog.askopenfilename(
                filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac"), ("All", "*.*")])
            if not path:
                return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext != ".wav" and HAS_PYDUB:
                buf = io.BytesIO()
                AudioSegment.from_file(path).export(buf, format="wav")
                buf.seek(0)
                pygame.mixer.music.load(buf)
            else:
                pygame.mixer.music.load(path)
            self._audio_path = path
            self._audio_loaded = True
            # Pre-load samples for envelope extraction
            samples = load_audio_samples(path)
            if samples is not None:
                self._audio_samples = samples
                dur = len(samples) / AUDIO_SR
                self._audio_dur = dur
                self.audio_lbl.config(text=f"✓ {os.path.basename(path)} ({dur:.1f}s)")
                # Update bar slider range based on audio duration
                bpm = self.bpm_var.get()
                bar_duration = 4 * 60.0 / bpm
                max_bars = max(1, int(dur / bar_duration) + 1)
                self.bar_slider.config(to=max_bars)
            else:
                self._audio_samples = None
                self.audio_lbl.config(
                    text="✓ playback OK · ⚠ envelope extraction failed")
        except Exception as e:
            self._audio_loaded = False
            self._audio_path = None
            self._audio_samples = None
            messagebox.showerror("Audio Error", str(e))
            self.audio_lbl.config(text="Load failed")

    def _clear_audio(self):
        if HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        self._audio_loaded = False
        self._audio_path = None
        self._audio_samples = None
        self._audio_dur = 0.0
        self.bar_slider.config(to=999)
        self.audio_lbl.config(text="No audio loaded")
        self.audio_meter_lbl.config(text="Level: ────────────────────")

    def _stop_audio(self):
        if HAS_PYGAME:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    # ── Transport ────────────────────────────────────────────────────────

    def _wall_time(self):
        if self.playing:
            return self._elapsed + (time.time() - self._t0)
        return self._elapsed

    def _play(self):
        if self.playing:
            self.playing = False
            self._elapsed += time.time() - self._t0
            self._stop_audio()
            self.play_btn.config(text="▶  PLAY")
        else:
            self._t0 = time.time()
            self.playing = True
            self.play_btn.config(text="⏸  PAUSE")
            if HAS_PYGAME and self._audio_loaded:
                try:
                    pygame.mixer.music.play(0, start=self._elapsed)
                except TypeError:
                    pygame.mixer.music.play(0)

    def _stop(self):
        self.playing = False
        self._elapsed = 0.0
        self._stop_audio()
        self.play_btn.config(text="▶  PLAY")
        self.bar_var.set(1)
        self.bar_fine_var.set(0.0)
        if self._auto_mode == "write":
            self._auto_stop_write()
        elif self._auto_mode == "overdub":
            self._auto_stop_overdub()

    # ── Params ───────────────────────────────────────────────────────────

    def _build_params(self):
        beat_hz = self.bpm_var.get() / 60.0
        drift_ratio = DRIFT_VALUES.get(self.drift_ratio_var.get(), 0.0)
        return {
            "beat_hz": beat_hz,
            "timebase": self.timebase_var.get(),
            "base_ratio": RATIO_VALUES.get(self.base_ratio_var.get(), 1.0),
            "base_shape": self.base_shape.get(),
            "base_amplitude": self.amplitude.get(),
            "overdrive": self.overdrive.get(),
            "drift_ratio": drift_ratio,
            "drift_fine": self.drift_fine_var.get(),
            "modulators": [
                {
                    "type": m["type"].get(),
                    "shape": m["shape"].get(),
                    "ratio": RATIO_VALUES.get(m["ratio"].get(), 1.0),
                    "depth": m["depth"].get(),
                    "amplitude": m["amp"].get(),
                }
                for m in self.mod_vars
            ],
            "xmod": [[self.xmod_vars[i][j].get() for j in range(4)] for i in range(4)],
            "audio_mod": {
                "type": self.audio_mod_type.get(),
                "depth": self.audio_mod_depth.get() if self.audio_mod_enabled.get() else 0.0,
                "gain": self.audio_mod_gain.get(),
                "smoothing": self.audio_mod_smoothing.get(),
            },
            "audio_inject": {
                "enabled": self.audio_inject_enabled.get(),
                "amplitude": self.audio_inject_amp.get() if self.audio_inject_enabled.get() else 0.0,
            },
            "fg_color": self.fg_color,
            "bg_color": self.bg_color,
            "line_thickness": self.line_thickness.get(),
            "line_glow": self.line_glow.get(),
            "line_bloom": self.line_bloom.get(),
            "line_trail": self.line_trail.get(),
            "param_mod_routing": {
                key: [v.get() for v in vars_list]
                for key, vars_list in self.param_mod_routing.items()
            },
        }

    # ── Automation ───────────────────────────────────────────────────────

    def _get_automatable_params(self):
        """Return dict mapping automation key -> DoubleVar."""
        d = {
            "amplitude": self.amplitude,
            "overdrive": self.overdrive,
            "drift_fine": self.drift_fine_var,
            "line_thickness": self.line_thickness,
            "line_glow": self.line_glow,
            "line_bloom": self.line_bloom,
            "line_trail": self.line_trail,
            "audio_mod_depth": self.audio_mod_depth,
            "audio_mod_gain": self.audio_mod_gain,
            "audio_mod_smoothing": self.audio_mod_smoothing,
            "audio_inject_amp": self.audio_inject_amp,
            "timebase": self.timebase_var,
            "active_snapshot": self._active_snapshot,
        }
        for i in range(4):
            d[f"mod{i}_depth"] = self.mod_vars[i]["depth"]
            d[f"mod{i}_amp"] = self.mod_vars[i]["amp"]
        return d

    def _auto_record_tick(self):
        """Called from _tick() when recording automation."""
        t = self._wall_time()
        params = self._get_automatable_params()
        snap_val = params["active_snapshot"].get()
        snap_lane = self._automation_data.get("active_snapshot", [])
        snap_is_changing = bool(snap_lane) and abs(snap_lane[-1]["v"] - snap_val) > 1e-6
        if snap_is_changing:
            # Snapshot switched this tick: only record active_snapshot.
            # The snapshot recall sets all other params, so individual lanes
            # would create redundant and conflicting keyframe data.
            snap_lane.append({"t": t, "v": snap_val})
            return
        for key, var in params.items():
            val = var.get()
            lane = self._automation_data.setdefault(key, [])
            if not lane or abs(lane[-1]["v"] - val) > 1e-6:
                lane.append({"t": t, "v": val})

    def _auto_overdub_tick(self):
        """Called from _tick() when overdubbing automation."""
        t = self._wall_time()
        params = self._get_automatable_params()
        for key, var in params.items():
            val = var.get()
            lane = self._overdub_buffer.setdefault(key, [])
            if not lane or abs(lane[-1]["v"] - val) > 1e-6:
                lane.append({"t": t, "v": val})

    def _auto_playback_tick(self):
        """Called from _tick() when playing back automation."""
        t = self._wall_time()
        params = self._get_automatable_params()
        snap_lane = self._automation_data.get("active_snapshot", [])
        if snap_lane:
            # Snapshot automation is active: apply it (step/hold) and skip all
            # individual parameter lanes.  The snapshot recall triggered by
            # _on_snapshot_select already sets every parameter value, so playing
            # back the individual lanes would only cause linear-interpolation
            # artifacts that fight the instant snapshot switch.
            val = self._step_lane(snap_lane, t)
            if val is not None:
                params["active_snapshot"].set(int(round(val)))
            return
        for key, var in params.items():
            lane = self._automation_data.get(key, [])
            if not lane:
                continue
            if key == "active_snapshot":
                val = self._step_lane(lane, t)
                if val is not None:
                    var.set(int(round(val)))
            else:
                val = self._interpolate_lane(lane, t)
                if val is not None:
                    var.set(val)

    def _interpolate_lane(self, lane, t):
        """Linear interpolation between keyframes."""
        if not lane:
            return None
        if t <= lane[0]["t"]:
            return lane[0]["v"]
        if t >= lane[-1]["t"]:
            return lane[-1]["v"]
        lo, hi = 0, len(lane) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if lane[mid]["t"] <= t:
                lo = mid
            else:
                hi = mid
        t0, v0 = lane[lo]["t"], lane[lo]["v"]
        t1, v1 = lane[hi]["t"], lane[hi]["v"]
        if t1 == t0:
            return v0
        frac = (t - t0) / (t1 - t0)
        return v0 + (v1 - v0) * frac

    def _step_lane(self, lane, t):
        """Step/hold lookup — return the value of the most recent keyframe at or before time t.
        Unlike _interpolate_lane, this does NOT interpolate between keyframes.
        Use for discrete parameters like snapshot index."""
        if not lane:
            return None
        if t < lane[0]["t"]:
            return lane[0]["v"]
        # Binary search for the last keyframe at or before t
        lo, hi = 0, len(lane) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if lane[mid]["t"] <= t:
                lo = mid
            else:
                hi = mid - 1
        return lane[lo]["v"]

    def _apply_automation_to_params(self, params, t):
        """Override params dict values with automation data at time t."""
        for key, lane in self._automation_data.items():
            if not lane:
                continue
            if key == "active_snapshot":
                val = self._step_lane(lane, t)
                if val is not None:
                    self._active_snapshot.set(int(round(val)))
                continue
            val = self._interpolate_lane(lane, t)
            if val is None:
                continue
            if key == "amplitude":
                params["base_amplitude"] = val
            elif key == "overdrive":
                params["overdrive"] = val
            elif key == "drift_fine":
                params["drift_fine"] = val
            elif key == "timebase":
                params["timebase"] = val
            elif key == "line_thickness":
                params["line_thickness"] = val
            elif key == "line_glow":
                params["line_glow"] = val
            elif key == "line_bloom":
                params["line_bloom"] = val
            elif key == "line_trail":
                params["line_trail"] = val
            elif key == "audio_mod_depth":
                params["audio_mod"]["depth"] = val
            elif key == "audio_mod_gain":
                params["audio_mod"]["gain"] = val
            elif key == "audio_mod_smoothing":
                params["audio_mod"]["smoothing"] = val
            elif key == "audio_inject_amp":
                params["audio_inject"]["amplitude"] = val
            elif key.startswith("mod") and "_" in key:
                parts = key.split("_", 1)
                idx = int(parts[0][3:])
                field = parts[1]
                if field == "depth":
                    params["modulators"][idx]["depth"] = val
                elif field == "amp":
                    params["modulators"][idx]["amplitude"] = val
        return params

    def _auto_write(self):
        if self._auto_mode == "write":
            self._auto_stop_write()
        else:
            self._auto_mode = "write"
            self._automation_data = {}
            self._elapsed = 0.0
            self._t0 = time.time()
            self.playing = True
            self.play_btn.config(text="⏸  PAUSE")
            if HAS_PYGAME and self._audio_loaded:
                try:
                    pygame.mixer.music.play(0)
                except Exception:
                    pass
            self.auto_write_btn.config(
                text="■ STOP WRITE", bg="#662222", fg="#ffaaaa",
                activebackground="#662222")

    def _auto_stop_write(self):
        self._auto_mode = "off"
        self.auto_write_btn.config(
            text="✎ WRITE", bg="#333333", fg="#cccccc",
            activebackground="#333333")

    def _auto_overdub(self):
        if self._auto_mode == "overdub":
            self._auto_stop_overdub()
        else:
            self._auto_mode = "overdub"
            self._overdub_buffer = {}
            self._overdub_start_t = self._wall_time()
            self._elapsed = 0.0
            self._t0 = time.time()
            self.playing = True
            self.play_btn.config(text="⏸  PAUSE")
            if HAS_PYGAME and self._audio_loaded:
                try:
                    pygame.mixer.music.play(0)
                except Exception:
                    pass
            self.auto_overdub_btn.config(
                text="■ STOP ODUB", bg="#664422", fg="#ffddaa",
                activebackground="#664422")

    def _auto_stop_overdub(self):
        self._auto_mode = "off"
        # Splice overdub buffer into main automation data
        t_start = self._overdub_start_t
        t_end = self._wall_time()
        for key, new_lane in self._overdub_buffer.items():
            if not new_lane:
                continue
            existing = self._automation_data.get(key, [])
            # Remove keyframes in the overdubbed time range
            kept = [kf for kf in existing if not (t_start <= kf["t"] <= t_end)]
            # Insert new keyframes and sort
            merged = kept + new_lane
            merged.sort(key=lambda kf: kf["t"])
            self._automation_data[key] = merged
        self._overdub_buffer = {}
        self.auto_overdub_btn.config(
            text="⊕ OVERDUB", bg="#333333", fg="#cccccc",
            activebackground="#333333")

    def _auto_reset(self):
        from tkinter import messagebox as _mb
        if _mb.askyesno("Reset", "Clear all automation?"):
            self._automation_data = {}
            self._overdub_buffer = {}
            self._auto_mode = "off"
            self.auto_write_btn.config(
                text="✎ WRITE", bg="#333333", fg="#cccccc",
                activebackground="#333333")
            self.auto_overdub_btn.config(
                text="⊕ OVERDUB", bg="#333333", fg="#cccccc",
                activebackground="#333333")
            self.auto_status_lbl.config(text="No automation")

    # ── Preview Loop ─────────────────────────────────────────────────────

    def _tick(self):
        bpm = self.bpm_var.get()
        beat_hz = bpm / 60.0
        tb = self.timebase_var.get()
        temporal_hz = beat_hz * tb

        drift_ratio = DRIFT_VALUES.get(self.drift_ratio_var.get(), 0.0)
        drift_fine = self.drift_fine_var.get()
        drift_hz = beat_hz * drift_ratio * drift_fine

        self.bpm_hz_lbl.config(text=f"= {beat_hz:.3f} Hz (beat rate)")
        self.tb_lbl.config(text=f"{tb:.2f}×")
        self.tb_hz_lbl.config(text=f"Animation: {temporal_hz:.3f} Hz")
        self.drift_fine_lbl.config(text=f"{drift_fine:.2f}×")

        if drift_ratio == 0:
            self.drift_info_lbl.config(text="Drift OFF")
        else:
            period = 1.0 / drift_hz if drift_hz > 0 else float('inf')
            bpc = 1.0 / (drift_ratio * drift_fine) if drift_ratio > 0 else float('inf')
            self.drift_info_lbl.config(
                text=f"Drift: {drift_hz:.3f} Hz · {period:.2f}s · {bpc:.1f} beats/cycle")

        base_ratio = RATIO_VALUES.get(self.base_ratio_var.get(), 1.0)
        self.base_hz_lbl.config(text=f"{base_ratio:.1f} wiggles")
        self._amplitude_lbl.config(text=f"{self.amplitude.get():.2f}")
        self._overdrive_lbl.config(text=f"{self.overdrive.get():.1f}×")

        # Audio mod labels
        self._audio_mod_depth_lbl.config(text=f"{self.audio_mod_depth.get():.2f}")
        self._audio_mod_gain_lbl.config(text=f"{self.audio_mod_gain.get():.2f}")
        self._audio_mod_smoothing_lbl.config(text=f"{self.audio_mod_smoothing.get():.3f}")
        self._audio_inject_amp_lbl.config(text=f"{self.audio_inject_amp.get():.2f}")

        # Line style labels
        self._line_thickness_lbl.config(text=f"{self.line_thickness.get():.1f}×")
        self._line_glow_lbl.config(text=f"{self.line_glow.get():.2f}")
        self._line_bloom_lbl.config(text=f"{self.line_bloom.get():.2f}")
        self._line_trail_lbl.config(text=f"{self.line_trail.get():.2f}")

        for mv in self.mod_vars:
            mv["depth_lbl"].config(text=f"{mv['depth'].get():.2f}")
            mv["amp_lbl"].config(text=f"{mv['amp'].get():.2f}")
            r = RATIO_VALUES.get(mv["ratio"].get(), 1.0)
            mv["hz_lbl"].config(text=f"{r:.1f}× · {temporal_hz * r:.3f} Hz")

        # Automation
        if self._auto_mode == "write" and self.playing:
            self._auto_record_tick()
            n_keys = sum(len(v) for v in self._automation_data.values())
            self.auto_status_lbl.config(text=f"✎ WRITING · {n_keys} keyframes")
        elif self._auto_mode == "overdub" and self.playing:
            self._auto_overdub_tick()
            n_keys = sum(len(v) for v in self._overdub_buffer.values())
            self.auto_status_lbl.config(text=f"⊕ OVERDUB · {n_keys} new keyframes")
        elif self._automation_data and self.playing:
            self._auto_playback_tick()
            n_total = sum(len(v) for v in self._automation_data.values())
            all_t = [kf["t"] for lane in self._automation_data.values() for kf in lane]
            dur = max(all_t) if all_t else 0
            self.auto_status_lbl.config(text=f"▶ {n_total} keyframes · {dur:.1f}s")
        elif self._automation_data:
            n_total = sum(len(v) for v in self._automation_data.values())
            all_t = [kf["t"] for lane in self._automation_data.values() for kf in lane]
            dur = max(all_t) if all_t else 0
            self.auto_status_lbl.config(text=f"⏸ {n_total} keyframes · {dur:.1f}s")
        else:
            self.auto_status_lbl.config(text="No automation")

        wall = self._wall_time()
        m = int(wall) // 60
        sec = wall - m * 60
        self.time_lbl.config(text=f"{m:02d}:{sec:06.3f}")

        # Bar / beat tracking
        bpm = self.bpm_var.get()
        bar_duration = 4 * 60.0 / bpm  # 4 beats per bar (4/4 time)
        current_bar = int(wall / bar_duration) + 1
        beat_in_bar = (wall % bar_duration) / (60.0 / bpm) + 1
        self.bar_lbl.config(text=f"Bar {current_bar}")
        self.beat_lbl.config(text=f"Beat {beat_in_bar:.1f}")
        self.bar_fine_lbl.config(text=f"{self.bar_fine_var.get():.2f}")
        if self.playing:
            self._bar_tracking = True
            self.bar_var.set(current_bar)
            self._bar_tracking = False

        # Audio level meter
        if not self.audio_mod_enabled.get():
            self.audio_meter_lbl.config(text="Level: ────────────────────")
            self.audio_mod_info.config(text="☐ disconnected")
        elif self._audio_samples is not None and self.playing:
            level = compute_audio_envelope(
                self._audio_samples, wall, self.audio_mod_smoothing.get())
            level *= self.audio_mod_gain.get()
            level = min(1.0, level)
            bars = int(level * 24)
            meter = "█" * bars + "░" * (24 - bars)
            self.audio_meter_lbl.config(text=f"Level: {meter} {level:.2f}")
            depth = self.audio_mod_depth.get()
            if depth > 0:
                self.audio_mod_info.config(
                    text=f"→ {self.audio_mod_type.get()} {level * depth:.2f}")
            else:
                self.audio_mod_info.config(text="(depth = 0)")
        elif self._audio_samples is not None:
            self.audio_meter_lbl.config(text="Level: (paused)")
            depth = self.audio_mod_depth.get()
            if depth > 0:
                self.audio_mod_info.config(
                    text=f"→ {self.audio_mod_type.get()} depth={depth:.2f} (paused)")
            else:
                self.audio_mod_info.config(text="(paused)")
        elif self._audio_loaded:
            self.audio_mod_info.config(text="⚠ no envelope data")
        else:
            self.audio_mod_info.config(text="⚠ no audio loaded")

        # Audio inject status
        if not self.audio_inject_enabled.get():
            self.audio_inject_info.config(text="☐ off")
        elif self._audio_samples is not None:
            amp = self.audio_inject_amp.get()
            self.audio_inject_info.config(text=f"● amp={amp:.2f}")
        else:
            self.audio_inject_info.config(text="⚠ no audio loaded")

        params = self._build_params()
        pf = render_frame(PREVIEW_W, PREVIEW_H, wall, params, self._audio_samples,
                          trail_buffers=self._trail_buffers)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(pf))
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        self.after(16, self._tick)

    # ── Patch Save / Load ────────────────────────────────────────────────
    def _gather_patch(self):
        return {
            "version": 15,
            "bpm": self.bpm_var.get(),
            "timebase": self.timebase_var.get(),
            "timebase_preset": self.tb_preset_var.get(),
            "base_shape": self.base_shape.get(),
            "base_ratio": self.base_ratio_var.get(),
            "amplitude": self.amplitude.get(),
            "overdrive": self.overdrive.get(),
            "drift_ratio": self.drift_ratio_var.get(),
            "drift_fine": self.drift_fine_var.get(),
            "fg_color": list(self.fg_color),
            "bg_color": list(self.bg_color),
            "modulators": [
                {
                    "type": mv["type"].get(),
                    "shape": mv["shape"].get(),
                    "ratio": mv["ratio"].get(),
                    "depth": mv["depth"].get(),
                    "amp": mv["amp"].get(),
                    "xmod": [self.xmod_vars[i][j].get() for j in range(4)],
                }
                for i, mv in enumerate(self.mod_vars)
            ],
            "audio_mod": {
                "type": self.audio_mod_type.get(),
                "depth": self.audio_mod_depth.get(),
                "gain": self.audio_mod_gain.get(),
                "smoothing": self.audio_mod_smoothing.get(),
                "enabled": self.audio_mod_enabled.get(),
            },
            "audio_inject": {
                "enabled": self.audio_inject_enabled.get(),
                "amplitude": self.audio_inject_amp.get(),
            },
            "audio_path": self._audio_path,
            "export_duration": self.duration_var.get(),
            "export_format": self.format_var.get(),
            "export_res": self.res_var.get(),
            "automation": self._automation_data,
            "line_thickness": self.line_thickness.get(),
            "line_glow": self.line_glow.get(),
            "line_bloom": self.line_bloom.get(),
            "line_trail": self.line_trail.get(),
            "param_mod_routing": {
                key: [v.get() for v in vars_list]
                for key, vars_list in self.param_mod_routing.items()
            },
            "snapshots": list(self._snapshots),
            "active_snapshot": self._active_snapshot.get(),
            "snapshots_locked": self._snapshots_locked.get(),
            "start_bar": self.bar_var.get(),
            "start_bar_fine": self.bar_fine_var.get(),
        }

    def _apply_patch(self, p):
        self.bpm_var.set(p.get("bpm", 120.0))
        self.timebase_var.set(p.get("timebase", 1.0))
        self.tb_preset_var.set(p.get("timebase_preset", "×1"))
        self.base_shape.set(p.get("base_shape", "Sine"))
        self.base_ratio_var.set(p.get("base_ratio", "×4"))
        self.amplitude.set(p.get("amplitude", 0.4))
        self.overdrive.set(p.get("overdrive", 3.0))
        self.drift_ratio_var.set(p.get("drift_ratio", "÷4"))
        self.drift_fine_var.set(p.get("drift_fine", 1.0))
        self.fg_color = tuple(p.get("fg_color", [0, 255, 140]))
        self.fg_btn.configure(bg=self._hex(self.fg_color))
        self.bg_color = tuple(p.get("bg_color", [4, 4, 12]))
        self.bg_btn.configure(bg=self._hex(self.bg_color))
        for i, mv in enumerate(self.mod_vars):
            mods = p.get("modulators", [])
            md = mods[i] if i < len(mods) else {}
            mv["type"].set(md.get("type", "FM"))
            mv["shape"].set(md.get("shape", "Sine"))
            mv["ratio"].set(md.get("ratio", "×1"))
            mv["depth"].set(md.get("depth", 0.0))
            mv["amp"].set(md.get("amp", 1.0))
            xm = md.get("xmod", [False, False, False, False])
            for j in range(4):
                if j != i:
                    self.xmod_vars[i][j].set(xm[j] if j < len(xm) else False)
        am = p.get("audio_mod", {})
        self.audio_mod_type.set(am.get("type", "AM"))
        self.audio_mod_depth.set(am.get("depth", 0.0))
        self.audio_mod_gain.set(am.get("gain", 1.0))
        self.audio_mod_smoothing.set(am.get("smoothing", 0.05))
        self.audio_mod_enabled.set(am.get("enabled", False))
        self._update_audio_mod_btn()
        ai = p.get("audio_inject", {})
        self.audio_inject_enabled.set(ai.get("enabled", False))
        self.audio_inject_amp.set(ai.get("amplitude", 0.0))
        self._update_audio_inject_btn()
        self.duration_var.set(p.get("export_duration", 5))
        self.format_var.set(p.get("export_format", "MP4"))
        self.res_var.set(p.get("export_res", "1080×1920"))
        self._automation_data = p.get("automation", {})
        self.line_thickness.set(p.get("line_thickness", 1.0))
        self.line_glow.set(p.get("line_glow", 1.0))
        self.line_bloom.set(p.get("line_bloom", 0.0))
        self.line_trail.set(p.get("line_trail", 0.0))
        routing = p.get("param_mod_routing", {})
        for key, vars_list in self.param_mod_routing.items():
            saved = routing.get(key, [False] * 5)
            for i, var in enumerate(vars_list):
                var.set(saved[i] if i < len(saved) else False)
        ap = p.get("audio_path")
        if ap and os.path.isfile(ap):
            self._load_audio(ap)
        elif ap:
            self.audio_lbl.config(text=f"⚠ {os.path.basename(ap)}")
            self._audio_path = ap
            self._audio_loaded = False
        saved_snaps = p.get("snapshots", [None] * 8)
        for i in range(8):
            if i < len(saved_snaps) and saved_snaps[i] is not None:
                self._snapshots[i] = saved_snaps[i]
                self._snapshot_btns[i].config(text=f"●{i+1}")
            else:
                self._snapshots[i] = None
                self._snapshot_btns[i].config(text=str(i+1))
        self._active_snapshot.set(p.get("active_snapshot", -1))
        self._snapshots_locked.set(p.get("snapshots_locked", False))
        # Update lock button appearance
        if self._snapshots_locked.get():
            self._snap_lock_btn.configure(
                text="🔒 LOCKED", bg="#662222", fg="#ffaaaa",
                activebackground="#662222")
        else:
            self._snap_lock_btn.configure(
                text="🔓 UNLOCKED", bg="#333333", fg="#cccccc",
                activebackground="#333333")
        bar = p.get("start_bar", 1)
        fine = p.get("start_bar_fine", 0.0)
        validated_bar = max(1, bar)
        self.bar_var.set(validated_bar)
        self.bar_fine_var.set(fine)
        bpm = self.bpm_var.get()
        bar_duration = 4 * 60.0 / bpm  # 4 beats per bar (4/4 time)
        self._elapsed = (validated_bar - 1 + fine) * bar_duration

    # ── Snapshot methods ─────────────────────────────────────────────────

    def _toggle_snap_lock(self):
        locked = not self._snapshots_locked.get()
        self._snapshots_locked.set(locked)
        if locked:
            self._snap_lock_btn.configure(
                text="🔒 LOCKED", bg="#662222", fg="#ffaaaa",
                activebackground="#662222")
        else:
            self._snap_lock_btn.configure(
                text="🔓 UNLOCKED", bg="#333333", fg="#cccccc",
                activebackground="#333333")

    def _snapshot_store(self, index):
        """Capture current state into snapshot slot."""
        if self._snapshots_locked.get():
            return  # Do nothing when locked
        patch = self._gather_patch()
        for key in ("audio_path", "export_duration", "export_format", "export_res",
                    "automation", "version", "snapshots", "active_snapshot"):
            patch.pop(key, None)
        self._snapshots[index] = patch
        self._snapshot_btns[index].config(text=f"●{index+1}")

    def _on_snapshot_select(self, *_):
        idx = self._active_snapshot.get()
        if idx < 0 or idx >= 8:
            return
        snap = self._snapshots[idx]
        if snap is None:
            return
        self._apply_snapshot(snap)

    def _apply_snapshot(self, snap):
        """Apply a snapshot instantly — same as _apply_patch but skipping
        audio_path, export settings, and automation."""
        self.bpm_var.set(snap.get("bpm", 120.0))
        self.timebase_var.set(snap.get("timebase", 1.0))
        self.tb_preset_var.set(snap.get("timebase_preset", "×1"))
        self.base_shape.set(snap.get("base_shape", "Sine"))
        self.base_ratio_var.set(snap.get("base_ratio", "×4"))
        self.amplitude.set(snap.get("amplitude", 0.4))
        self.overdrive.set(snap.get("overdrive", 3.0))
        self.drift_ratio_var.set(snap.get("drift_ratio", "÷4"))
        self.drift_fine_var.set(snap.get("drift_fine", 1.0))
        self.fg_color = tuple(snap.get("fg_color", [0, 255, 140]))
        self.fg_btn.configure(bg=self._hex(self.fg_color))
        self.bg_color = tuple(snap.get("bg_color", [4, 4, 12]))
        self.bg_btn.configure(bg=self._hex(self.bg_color))
        for i, mv in enumerate(self.mod_vars):
            mods = snap.get("modulators", [])
            md = mods[i] if i < len(mods) else {}
            mv["type"].set(md.get("type", "FM"))
            mv["shape"].set(md.get("shape", "Sine"))
            mv["ratio"].set(md.get("ratio", "×1"))
            mv["depth"].set(md.get("depth", 0.0))
            mv["amp"].set(md.get("amp", 1.0))
            xm = md.get("xmod", [False, False, False, False])
            for j in range(4):
                if j != i:
                    self.xmod_vars[i][j].set(xm[j] if j < len(xm) else False)
        am = snap.get("audio_mod", {})
        self.audio_mod_type.set(am.get("type", "AM"))
        self.audio_mod_depth.set(am.get("depth", 0.0))
        self.audio_mod_gain.set(am.get("gain", 1.0))
        self.audio_mod_smoothing.set(am.get("smoothing", 0.05))
        self.audio_mod_enabled.set(am.get("enabled", False))
        self._update_audio_mod_btn()
        ai = snap.get("audio_inject", {})
        self.audio_inject_enabled.set(ai.get("enabled", False))
        self.audio_inject_amp.set(ai.get("amplitude", 0.0))
        self._update_audio_inject_btn()
        self.line_thickness.set(snap.get("line_thickness", 1.0))
        self.line_glow.set(snap.get("line_glow", 1.0))
        self.line_bloom.set(snap.get("line_bloom", 0.0))
        self.line_trail.set(snap.get("line_trail", 0.0))
        routing = snap.get("param_mod_routing", {})
        for key, vars_list in self.param_mod_routing.items():
            saved = routing.get(key, [False] * 5)
            for i, var in enumerate(vars_list):
                var.set(saved[i] if i < len(saved) else False)

    def _save_patch(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Patch", "*.json"), ("All", "*.*")],
            title="Save Patch")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._gather_patch(), f, indent=2)
            messagebox.showinfo("Patch", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _load_patch(self):
        path = filedialog.askopenfilename(
            filetypes=[("Patch", "*.json"), ("All", "*.*")],
            title="Load Patch")
        if not path:
            return
        try:
            with open(path) as f:
                self._apply_patch(json.load(f))
            messagebox.showinfo("Patch", f"Loaded:\n{os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Offline Export ───────────────────────────────────────────────────

    def _export(self):
        dur = self.duration_var.get()
        if not 1 <= dur <= 500:
            messagebox.showerror("Error", "Duration must be 1–30 s")
            return
        fmt = self.format_var.get()
        ext = ".mp4" if fmt == "MP4" else ".avi"
        ftypes = [(fmt, f"*{ext}")]
        path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=ftypes)
        if not path:
            return
        self.export_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self._export_cancel = False
        start_t = self._elapsed
        threading.Thread(target=self._export_worker, args=(path, dur, fmt, start_t), daemon=True).start()

    def _cancel_export(self):
        self._export_cancel = True

    def _export_worker(self, path, dur, fmt, start_t=0.0):
        total = dur * FPS
        res = self.res_var.get()
        if "4K" in res:
            ew, eh = 2160, 3840
        elif "720" in res:
            ew, eh = 720, 1280
        else:
            ew, eh = 1080, 1920

        # Try hardware-accelerated encoder first (NVENC via H264), fall back to software
        writer = None
        if fmt == "MP4":
            for cc in ("H264", "h264", "avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*cc)
                w = cv2.VideoWriter(path, fourcc, FPS, (ew, eh))
                if w.isOpened():
                    writer = w
                    break
        if writer is None:
            cc = "MJPG" if fmt == "AVI" else "mp4v"
            fourcc = cv2.VideoWriter_fourcc(*cc)
            writer = cv2.VideoWriter(path, fourcc, FPS, (ew, eh))
        if not writer.isOpened():
            self._done("Failed to open video writer.")
            return

        # Build base params ONCE (thread-safe snapshot)
        base_params = self._build_params()
        has_automation = bool(self._automation_data)
        samples = self._audio_samples
        trail_buffers = {}
        trail_amount = base_params.get("line_trail", 0.0)

        # Pre-compute reusable arrays sent once to each worker via the initializer
        precomputed = {
            "y_norm": np.linspace(0, 1, eh),
            "y_arange": np.arange(eh, dtype=np.float32),
        }

        n_workers = os.cpu_count() or 1
        # 2× worker count keeps the pool fed without holding too many frames in memory at once
        batch_size = n_workers * 2
        ts = time.time()

        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_init_export_worker,
            initargs=(samples, precomputed),
        ) as pool:
            i = 0
            while i < total:
                if self._export_cancel:
                    pool.terminate()
                    writer.release()
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    self._done("Export cancelled.")
                    return

                # Build per-frame params for this batch in the main thread
                batch_end = min(i + batch_size, total)
                batch_args = []
                for j in range(i, batch_end):
                    t_frame = start_t + j * FRAME_DUR
                    if has_automation:
                        frame_params = _shallow_copy_params(base_params)
                        self._apply_automation_to_params(frame_params, t_frame)
                    else:
                        frame_params = base_params
                    batch_args.append((ew, eh, t_frame, frame_params))

                # Render batch in parallel (no trail)
                frames = pool.map(_render_frame_worker, batch_args)

                # Sequential trail compositing and write
                for frame_bgr in frames:
                    if trail_amount > 0:
                        key = (ew, eh)
                        if key in trail_buffers:
                            frame_bgr = cv2.addWeighted(frame_bgr, 1.0, trail_buffers[key], trail_amount, 0)
                        trail_buffers[key] = frame_bgr.copy()
                    writer.write(frame_bgr)

                i = batch_end
                pct = i / total * 100
                self.progress.set(pct)
                el = time.time() - ts
                if i > 0:
                    eta = el / i * (total - i)
                    self.after(0, lambda p=pct, e=eta: self.progress_lbl.config(
                        text=f"{p:.0f}% · ETA {e:.0f}s · {ew}×{eh}"))

        writer.release()
        fsize = os.path.getsize(path) / (1024 * 1024)
        self._done(f"Saved: {path}\n{ew}×{eh} · {dur}s · {fmt} · {fsize:.1f} MB")

    def _done(self, msg):
        self.after(0, lambda: self.export_btn.config(state="normal"))
        self.after(0, lambda: self.cancel_btn.config(state="disabled"))
        self.after(0, lambda: self.progress.set(0))
        self.after(0, lambda: self.progress_lbl.config(text=""))
        self.after(0, lambda: messagebox.showinfo("Export", msg))


if __name__ == "__main__":
    app = WaveformApp()
    app.mainloop()
