"""
DSP Core - Image to Audio and Audio to Image
============================================
Encoding strategy:
  - Image resized to max MAX_IMAGE_DIM
  - Header: sync tone + 4 dimension tones (W high, W low, H high, H low) + end marker
  - Each pixel (R, G, B) → 3 mixed sine tones:
      R -> BASE_FREQ_R amplitude = R/255
      G -> BASE_FREQ_G amplitude = G/255
      B -> BASE_FREQ_B amplitude = B/255
  - Each pixel segment = PIXEL_DURATION seconds
  - Amplitude is detected via Goertzel-like DFT at decoding
"""

import numpy as np
from PIL import Image
import io
import wave
from scipy.io import wavfile

SAMPLE_RATE    = 44100
PIXEL_DURATION = 0.001       # 10ms per pixel for better frequency resolution
BASE_FREQ_R    = 440.0      # A4 - nice musical note
BASE_FREQ_G    = 880.0      # A5
BASE_FREQ_B    = 1760.0     # A6
SYNC_FREQ      = 220.0      # A3 sync marker
SYNC_DURATION  = 0.1        # 100ms per header tone
END_DURATION   = 0.05       # 50ms end marker
MAX_IMAGE_DIM  = 512         # 48x48 = 2304 pixels = ~23s audio

# Pre-calculated header samples
_sync_s = int(SAMPLE_RATE * SYNC_DURATION)
_end_s  = int(SAMPLE_RATE * END_DURATION)
HEADER_SAMPLES = _sync_s + 4 * _sync_s + _end_s  # = 5*4410 + 2205 = 24255
SAMPLES_PER_PIXEL = int(SAMPLE_RATE * PIXEL_DURATION)


def resize_image(img: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_dim / w, max_dim / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _make_tone(freq: float, amplitude: float, n_samples: int) -> np.ndarray:
    t = np.arange(n_samples) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * freq * t)


def encode_image_to_audio(image_bytes: bytes) -> tuple:
    img = Image.open(io.BytesIO(image_bytes))
    img = resize_image(img)
    width, height = img.size
    pixels = np.array(img)

    segments = []

    # --- Header ---
    # 1. Sync tone (full volume)
    segments.append(_make_tone(SYNC_FREQ, 1.0, _sync_s))

    # 2. Dimension tones (4 bytes: W_high, W_low, H_high, H_low)
    w_high = (width >> 8) & 0xFF
    w_low  = width & 0xFF
    h_high = (height >> 8) & 0xFF
    h_low  = height & 0xFF

    for val, freq in [(w_high, 150.0), (w_low, 250.0), (h_high, 350.0), (h_low, 450.0)]:
        amp = val / 255.0
        segments.append(_make_tone(freq, amp, _sync_s))

    # 3. End-of-header marker (half-amplitude sync)
    segments.append(_make_tone(SYNC_FREQ, 0.5, _end_s))

    # --- Pixel data ---
    spp = SAMPLES_PER_PIXEL
    t   = np.arange(spp) / SAMPLE_RATE
    sin_r = np.sin(2 * np.pi * BASE_FREQ_R * t)
    sin_g = np.sin(2 * np.pi * BASE_FREQ_G * t)
    sin_b = np.sin(2 * np.pi * BASE_FREQ_B * t)

    for row in pixels:
        for pixel in row:
            r_amp = float(pixel[0]) / 255.0
            g_amp = float(pixel[1]) / 255.0
            b_amp = float(pixel[2]) / 255.0
            # Mix and normalize by 3 channels
            seg = (r_amp * sin_r + g_amp * sin_g + b_amp * sin_b) / 3.0
            segments.append(seg)

    audio = np.concatenate(segments)

    audio_i16 = np.int16(audio * 32767)

    wav_buf = io.BytesIO()
    wavfile.write(wav_buf, SAMPLE_RATE, audio_i16)

    metadata = {
        'width': width,
        'height': height,
        'total_pixels': width * height,
        'audio_duration_sec': len(audio) / SAMPLE_RATE,
    }
    return wav_buf.getvalue(), metadata


def _decode_amp(audio: np.ndarray, start: int, freq: float) -> float:
    """Estimate amplitude of freq in audio[start:start+SAMPLES_PER_PIXEL]."""
    spp = SAMPLES_PER_PIXEL
    seg = audio[start: start + spp]
    if len(seg) < spp:
        return 0.0
    t = np.arange(spp) / float(SAMPLE_RATE)
    s = np.sin(2 * np.pi * freq * t)
    c = np.cos(2 * np.pi * freq * t)
    a = 2.0 * np.dot(seg, s) / spp
    b = 2.0 * np.dot(seg, c) / spp
    return float(np.sqrt(a*a + b*b))


def _decode_header_amp(audio: np.ndarray, start: int, freq: float) -> float:
    """Estimate amplitude of freq over a sync-duration segment."""
    seg = audio[start: start + _sync_s]
    if len(seg) < _sync_s:
        return 0.0
    t = np.arange(_sync_s) / float(SAMPLE_RATE)
    s = np.sin(2 * np.pi * freq * t)
    c = np.cos(2 * np.pi * freq * t)
    a = 2.0 * np.dot(seg, s) / _sync_s
    b = 2.0 * np.dot(seg, c) / _sync_s
    return float(np.sqrt(a*a + b*b))


def decode_audio_to_image(wav_bytes: bytes) -> tuple:
    wav_buf = io.BytesIO(wav_bytes)
    with wave.open(wav_buf, 'rb') as wf:
        n_ch  = wf.getnchannels()
        sw    = wf.getsampwidth()
        fr    = wf.getframerate()
        raw   = wf.readframes(wf.getnframes())

    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0

    if n_ch > 1:
        audio = audio[::n_ch]

    # Resample if needed
    if fr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, fr)
        audio = resample_poly(audio, SAMPLE_RATE // g, fr // g)

    # --- Decode header dimensions ---
    offset = _sync_s  # skip sync tone
    w_high_amp = _decode_header_amp(audio, offset,              150.0)
    w_low_amp  = _decode_header_amp(audio, offset + _sync_s,    250.0)
    h_high_amp = _decode_header_amp(audio, offset + 2*_sync_s,  350.0)
    h_low_amp  = _decode_header_amp(audio, offset + 3*_sync_s,  450.0)

    w_high = int(round(w_high_amp * 255))
    w_low  = int(round(w_low_amp  * 255))
    h_high = int(round(h_high_amp * 255))
    h_low  = int(round(h_low_amp  * 255))

    width  = max(1, min((w_high << 8) | w_low,  512))
    height = max(1, min((h_high << 8) | h_low,  512))

    # --- Decode pixels ---
    pixel_start  = HEADER_SAMPLES
    total_pixels = width * height
    pixels       = []

    for i in range(total_pixels):
        start = pixel_start + i * SAMPLES_PER_PIXEL
        if start + SAMPLES_PER_PIXEL > len(audio):
            pixels.append((0, 0, 0))
            continue
        r_amp = _decode_amp(audio, start, BASE_FREQ_R) * 3.0
        g_amp = _decode_amp(audio, start, BASE_FREQ_G) * 3.0
        b_amp = _decode_amp(audio, start, BASE_FREQ_B) * 3.0
        r = int(np.clip(round(r_amp * 255), 0, 255))
        g = int(np.clip(round(g_amp * 255), 0, 255))
        b = int(np.clip(round(b_amp * 255), 0, 255))
        pixels.append((r, g, b))

    img_arr = np.array(pixels, dtype=np.uint8).reshape(height, width, 3)
    img     = Image.fromarray(img_arr, 'RGB')

    # Scale up for display
    scale   = max(1, 256 // max(width, height))
    img_big = img.resize((width * scale, height * scale), Image.NEAREST)

    out = io.BytesIO()
    img_big.save(out, format='PNG')

    return out.getvalue(), {'width': width, 'height': height, 'decoded_pixels': len(pixels)}