"""
DSP Core - Image to Audio and Audio to Image
============================================
Encoding strategy (Stereo / Dual-Pixel):
  - Image resized to max MAX_IMAGE_DIM
  - Header: sync tone + 4 dimension tones (W high, W low, H high, H low) + end marker
      → identical waveform written to BOTH Left and Right channels
  - Each pixel pair (Pixel1, Pixel2) is encoded into one audio frame:
      Left  channel  ← Pixel1 (R,G,B) → 3 mixed sine tones
      Right channel  ← Pixel2 (R,G,B) → 3 mixed sine tones
  - Each pixel segment = PIXEL_DURATION seconds (audio duration halved vs mono)
  - Amplitude is detected via Goertzel-like DFT at decoding
  - If the total pixel count is odd the last Left slot has a real pixel and
    the last Right slot is padded with silence (black pixel).
"""

import numpy as np
from PIL import Image
import io
import wave
from scipy.io import wavfile

SAMPLE_RATE    = 22050
PIXEL_DURATION = 0.02       # Color info
BASE_FREQ_R    = 440.0      # A4
BASE_FREQ_G    = 880.0      # A5
BASE_FREQ_B    = 1760.0     # A6
SYNC_FREQ      = 220.0      # A3 sync marker
SYNC_DURATION  = 0.1        # 100 ms per header tone
END_DURATION   = 0.05       # 50 ms end marker
MAX_IMAGE_DIM  = 512        # Max width/height

# Pre-calculated header samples
_sync_s = int(SAMPLE_RATE * SYNC_DURATION)
_end_s  = int(SAMPLE_RATE * END_DURATION)
HEADER_SAMPLES    = _sync_s + 4 * _sync_s + _end_s
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


def _pixel_wave(pixel, sin_r: np.ndarray, sin_g: np.ndarray, sin_b: np.ndarray) -> np.ndarray:
    """Return the FDM wave for a single (R,G,B) pixel tuple."""
    r_amp = float(pixel[0]) / 255.0
    g_amp = float(pixel[1]) / 255.0
    b_amp = float(pixel[2]) / 255.0
    return (r_amp * sin_r + g_amp * sin_g + b_amp * sin_b) / 3.0


def _build_header_mono() -> np.ndarray:
    """Build the mono header waveform (sync + 4 dim tones + end marker)."""
    parts = [_make_tone(SYNC_FREQ, 1.0, _sync_s)]
    # Dimension values are embedded at encode time; return a zero-filled
    # placeholder that callers replace with the real values.
    # This helper is not called directly — see encode_image_to_audio.
    return np.concatenate(parts)


def encode_image_to_audio(image_bytes: bytes) -> tuple:
    img = Image.open(io.BytesIO(image_bytes))
    img = resize_image(img)
    width, height = img.size
    pixels = np.array(img)          # shape (H, W, 3)
    flat   = pixels.reshape(-1, 3)  # shape (total_pixels, 3)
    total_pixels = len(flat)

    # ------------------------------------------------------------------ #
    #  Build mono header (identical for both channels)                     #
    # ------------------------------------------------------------------ #
    hdr_parts = [_make_tone(SYNC_FREQ, 1.0, _sync_s)]

    w_high = (width  >> 8) & 0xFF
    w_low  =  width        & 0xFF
    h_high = (height >> 8) & 0xFF
    h_low  =  height       & 0xFF

    for val, freq in [(w_high, 150.0), (w_low, 250.0),
                      (h_high, 350.0), (h_low, 450.0)]:
        hdr_parts.append(_make_tone(freq, val / 255.0, _sync_s))

    hdr_parts.append(_make_tone(SYNC_FREQ, 0.5, _end_s))
    header_mono = np.concatenate(hdr_parts)          # length = HEADER_SAMPLES

    # ------------------------------------------------------------------ #
    #  Pre-compute carrier sinusoids                                       #
    # ------------------------------------------------------------------ #
    spp = SAMPLES_PER_PIXEL
    t   = np.arange(spp) / SAMPLE_RATE
    sin_r = np.sin(2 * np.pi * BASE_FREQ_R * t)
    sin_g = np.sin(2 * np.pi * BASE_FREQ_G * t)
    sin_b = np.sin(2 * np.pi * BASE_FREQ_B * t)
    silence = np.zeros(spp)

    # ------------------------------------------------------------------ #
    #  Iterate in chunks of 2 → one audio frame per pair                  #
    # ------------------------------------------------------------------ #
    left_segs  = [header_mono]
    right_segs = [header_mono.copy()]   # identical copy for right channel

    for i in range(0, total_pixels, 2):
        px1 = flat[i]
        left_segs.append(_pixel_wave(px1, sin_r, sin_g, sin_b))

        if i + 1 < total_pixels:
            px2 = flat[i + 1]
            right_segs.append(_pixel_wave(px2, sin_r, sin_g, sin_b))
        else:
            # Odd pixel count — pad right with silence
            right_segs.append(silence)

    left_audio  = np.concatenate(left_segs)
    right_audio = np.concatenate(right_segs)

    # ------------------------------------------------------------------ #
    #  Stack into (N, 2) stereo array and convert to 16-bit PCM           #
    # ------------------------------------------------------------------ #
    stereo = np.column_stack((left_audio, right_audio))  # shape (N, 2)
    stereo_i16 = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)

    wav_buf = io.BytesIO()
    wavfile.write(wav_buf, SAMPLE_RATE, stereo_i16)

    # Number of audio frames = header frames + ceil(total_pixels / 2) pixel frames
    n_frames = len(left_audio)
    metadata = {
        'width':             width,
        'height':            height,
        'total_pixels':      total_pixels,
        'audio_duration_sec': n_frames / SAMPLE_RATE,
    }
    return wav_buf.getvalue(), metadata


# ------------------------------------------------------------------ #
#  Goertzel-style amplitude estimators                                 #
# ------------------------------------------------------------------ #

def _decode_amp(channel: np.ndarray, start: int, freq: float) -> float:
    """Estimate amplitude of `freq` in channel[start : start+SAMPLES_PER_PIXEL]."""
    spp = SAMPLES_PER_PIXEL
    seg = channel[start: start + spp]
    if len(seg) < spp:
        return 0.0
    t = np.arange(spp) / float(SAMPLE_RATE)
    s = np.sin(2 * np.pi * freq * t)
    c = np.cos(2 * np.pi * freq * t)
    a = 2.0 * np.dot(seg, s) / spp
    b = 2.0 * np.dot(seg, c) / spp
    return float(np.sqrt(a * a + b * b))


def _decode_header_amp(channel: np.ndarray, start: int, freq: float) -> float:
    """Estimate amplitude of `freq` over a sync-duration segment."""
    seg = channel[start: start + _sync_s]
    if len(seg) < _sync_s:
        return 0.0
    t = np.arange(_sync_s) / float(SAMPLE_RATE)
    s = np.sin(2 * np.pi * freq * t)
    c = np.cos(2 * np.pi * freq * t)
    a = 2.0 * np.dot(seg, s) / _sync_s
    b = 2.0 * np.dot(seg, c) / _sync_s
    return float(np.sqrt(a * a + b * b))


def _load_stereo_channels(wav_bytes: bytes):
    """
    Load a WAV file and return (left, right, sample_rate).

    For legacy mono files the right channel is a copy of the left so that
    the dual-pixel decoder degrades gracefully (right pixels will be black).
    """
    wav_buf = io.BytesIO(wav_bytes)
    with wave.open(wav_buf, 'rb') as wf:
        n_ch = wf.getnchannels()
        sw   = wf.getsampwidth()
        fr   = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())

    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    else:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0

    if n_ch >= 2:
        # De-interleave: even samples → left, odd samples → right
        left  = pcm[0::2]
        right = pcm[1::2]
    else:
        # Mono legacy file — duplicate channel
        left  = pcm
        right = pcm.copy()

    # Resample if needed
    if fr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g     = gcd(SAMPLE_RATE, fr)
        up    = SAMPLE_RATE // g
        down  = fr // g
        left  = resample_poly(left,  up, down)
        right = resample_poly(right, up, down)

    return left, right


def decode_audio_to_image(wav_bytes: bytes) -> tuple:
    left, right = _load_stereo_channels(wav_bytes)

    # --- Decode header from left channel (both channels are identical) ---
    offset     = _sync_s
    w_high_amp = _decode_header_amp(left, offset,             150.0)
    w_low_amp  = _decode_header_amp(left, offset + _sync_s,   250.0)
    h_high_amp = _decode_header_amp(left, offset + 2*_sync_s, 350.0)
    h_low_amp  = _decode_header_amp(left, offset + 3*_sync_s, 450.0)

    w_high = int(round(w_high_amp * 255))
    w_low  = int(round(w_low_amp  * 255))
    h_high = int(round(h_high_amp * 255))
    h_low  = int(round(h_low_amp  * 255))

    width  = max(1, min((w_high << 8) | w_low,  512))
    height = max(1, min((h_high << 8) | h_low,  512))

    # --- Decode pixels (dual-pixel: left=px[2i], right=px[2i+1]) ---
    total_pixels = width * height
    pixels       = [None] * total_pixels
    spp          = SAMPLES_PER_PIXEL

    frame_idx = 0
    i = 0
    while i < total_pixels:
        frame_start = HEADER_SAMPLES + frame_idx * spp

        # Left channel → pixel i
        if frame_start + spp <= len(left):
            r = int(np.clip(round(_decode_amp(left, frame_start, BASE_FREQ_R) * 3.0 * 255), 0, 255))
            g = int(np.clip(round(_decode_amp(left, frame_start, BASE_FREQ_G) * 3.0 * 255), 0, 255))
            b = int(np.clip(round(_decode_amp(left, frame_start, BASE_FREQ_B) * 3.0 * 255), 0, 255))
            pixels[i] = (r, g, b)
        else:
            pixels[i] = (0, 0, 0)
        i += 1

        # Right channel → pixel i+1
        if i < total_pixels:
            if frame_start + spp <= len(right):
                r = int(np.clip(round(_decode_amp(right, frame_start, BASE_FREQ_R) * 3.0 * 255), 0, 255))
                g = int(np.clip(round(_decode_amp(right, frame_start, BASE_FREQ_G) * 3.0 * 255), 0, 255))
                b = int(np.clip(round(_decode_amp(right, frame_start, BASE_FREQ_B) * 3.0 * 255), 0, 255))
                pixels[i] = (r, g, b)
            else:
                pixels[i] = (0, 0, 0)
            i += 1

        frame_idx += 1

    img_arr = np.array(pixels, dtype=np.uint8).reshape(height, width, 3)
    img     = Image.fromarray(img_arr, 'RGB')

    scale   = max(1, 256 // max(width, height))
    img_big = img.resize((width * scale, height * scale), Image.NEAREST)

    out = io.BytesIO()
    img_big.save(out, format='PNG')

    return out.getvalue(), {'width': width, 'height': height, 'decoded_pixels': len(pixels)}