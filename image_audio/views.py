import json
import time
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render
from .dsp_core import (
    encode_image_to_audio, decode_audio_to_image,
    HEADER_SAMPLES, SAMPLES_PER_PIXEL, SAMPLE_RATE,
    BASE_FREQ_R, BASE_FREQ_G, BASE_FREQ_B,
    _sync_s, _decode_header_amp,
)


def index(request):
    return render(request, 'image_audio/index.html')


@csrf_exempt
@require_http_methods(["POST"])
def encode_image(request):
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image file provided'}, status=400)
    image_file = request.FILES['image']
    if image_file.content_type not in ['image/jpeg','image/png','image/gif','image/bmp','image/webp']:
        return JsonResponse({'error': f'Unsupported type: {image_file.content_type}'}, status=400)
    if image_file.size > 10 * 1024 * 1024:
        return JsonResponse({'error': 'Image too large (max 10MB)'}, status=400)
    try:
        wav_bytes, metadata = encode_image_to_audio(image_file.read())
        response = HttpResponse(wav_bytes, content_type='audio/wav')
        response['Content-Disposition'] = f'attachment; filename="dsp_image_{int(time.time())}.wav"'
        response['X-Image-Width']    = str(metadata['width'])
        response['X-Image-Height']   = str(metadata['height'])
        response['X-Total-Pixels']   = str(metadata['total_pixels'])
        response['X-Audio-Duration'] = f"{metadata['audio_duration_sec']:.2f}"
        response['Access-Control-Expose-Headers'] = 'X-Image-Width,X-Image-Height,X-Total-Pixels,X-Audio-Duration'
        return response
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def decode_audio(request):
    if 'audio' not in request.FILES:
        return JsonResponse({'error': 'No audio file provided'}, status=400)
    audio_file = request.FILES['audio']
    if audio_file.size > 100 * 1024 * 1024:
        return JsonResponse({'error': 'Audio too large (max 100MB)'}, status=400)
    try:
        wav_bytes = audio_file.read()
        png_bytes, metadata = decode_audio_to_image(wav_bytes)
        response = HttpResponse(png_bytes, content_type='image/png')
        response['Content-Disposition'] = 'inline; filename="decoded_image.png"'
        response['X-Decoded-Width']  = str(metadata['width'])
        response['X-Decoded-Height'] = str(metadata['height'])
        response['Access-Control-Expose-Headers'] = 'X-Decoded-Width,X-Decoded-Height'
        return response
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def decode_audio_stream(request):
    if 'audio' not in request.FILES:
        return JsonResponse({'error': 'No audio file provided'}, status=400)
    audio_file = request.FILES['audio']
    wav_bytes = audio_file.read()

    def pixel_generator(wav_bytes):
        import wave, io as _io
        import numpy as np

        try:
            with wave.open(_io.BytesIO(wav_bytes), 'rb') as wf:
                fr  = wf.getframerate()
                nch = wf.getnchannels()
                sw  = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        if sw == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        else:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
        if nch > 1:
            audio = audio[::nch]

        if fr != SAMPLE_RATE:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(SAMPLE_RATE, fr)
            audio = resample_poly(audio, SAMPLE_RATE // g, fr // g)

        spp = SAMPLES_PER_PIXEL

        def dec_hdr(start, freq):
            seg = audio[start: start + _sync_s]
            if len(seg) < _sync_s:
                return 0.0
            t = np.arange(_sync_s) / float(SAMPLE_RATE)
            s = np.sin(2*np.pi*freq*t); c = np.cos(2*np.pi*freq*t)
            a = 2.0*np.dot(seg,s)/_sync_s; b = 2.0*np.dot(seg,c)/_sync_s
            return float(np.sqrt(a*a+b*b))

        def dec_px(start, freq):
            seg = audio[start: start + spp]
            if len(seg) < spp:
                return 0.0
            t = np.arange(spp) / float(SAMPLE_RATE)
            s = np.sin(2*np.pi*freq*t); c = np.cos(2*np.pi*freq*t)
            a = 2.0*np.dot(seg,s)/spp; b = 2.0*np.dot(seg,c)/spp
            return float(np.sqrt(a*a+b*b))

        offset = _sync_s
        w_high = int(round(dec_hdr(offset,              150.0) * 255))
        w_low  = int(round(dec_hdr(offset + _sync_s,    250.0) * 255))
        h_high = int(round(dec_hdr(offset + 2*_sync_s,  350.0) * 255))
        h_low  = int(round(dec_hdr(offset + 3*_sync_s,  450.0) * 255))

        width  = max(1, min((w_high << 8) | w_low, 512))
        height = max(1, min((h_high << 8) | h_low, 512))

        yield f"data: {json.dumps({'type':'dimensions','width':width,'height':height})}\n\n"

        total = width * height
        batch = []
        BATCH = 32

        for i in range(total):
            start = HEADER_SAMPLES + i * spp
            if start + spp > len(audio):
                batch.append([0, 0, 0])
            else:
                r = int(np.clip(round(dec_px(start, BASE_FREQ_R) * 3.0 * 255), 0, 255))
                g = int(np.clip(round(dec_px(start, BASE_FREQ_G) * 3.0 * 255), 0, 255))
                b = int(np.clip(round(dec_px(start, BASE_FREQ_B) * 3.0 * 255), 0, 255))
                batch.append([r, g, b])
            if len(batch) >= BATCH:
                yield f"data: {json.dumps({'type':'pixels','start':i-BATCH+1,'pixels':batch})}\n\n"
                batch = []

        if batch:
            yield f"data: {json.dumps({'type':'pixels','start':total-len(batch),'pixels':batch})}\n\n"

        yield f"data: {json.dumps({'type':'done','total':total})}\n\n"

    response = StreamingHttpResponse(pixel_generator(wav_bytes), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response