# pip install sounddevice soundfile torch pyperclip keyboard pillow accelerate huggingface_hub sentencepiece protobuf librosa
# pip install git+https://github.com/huggingface/transformers
# pip install --upgrade "mistral-common[audio]"


# Cuda CUDA 11.8 Support
# pip install torch torchvision torchaudio - -index-url https: // download.pytorch.org/whl/cu118

# Für CUDA 12.1 Support (empfohlen)
# pip install torch torchvision torchaudio - -index-url https: // download.pytorch.org/whl/cu121

r"""
Background recorder + transcription with Voxtral (local) + clipboard copy
-----------------------------------------------------------
• Ctrl+R   Start / Stop recording
• Overlay  Red pulsing dot while recording, blue dot while transcribing
• After transcription a short Windows beep notifies that text is in clipboard
• Last chosen language is persisted in %USERPROFILE%\\.voxtral_config.json

Works with Python 3.10‒3.11. 3.13 currently unsupported by PyTorch.
"""

import threading
import queue
import tempfile
import json
import os
import sys
import winsound
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, List, Union, Optional, Callable

import sounddevice as sd
import soundfile as sf
import torch
import pyperclip
import keyboard
from transformers import VoxtralForConditionalGeneration, AutoProcessor

# ---------- Einstellbare Parameter ----------
MODEL_NAME = "mistralai/Voxtral-Mini-3B-2507"
RATE, CHANNELS = 16_000, 1
LANGS = ("de", "en", "fr", "es", "it", "pt", "nl")
CFG_PATH = os.path.join(os.path.expanduser("~"), ".voxtral_config.json")

# Standard-Einstellungen (können angepasst werden)
DEFAULT_LANG = "de"  # Änderbar: Standardsprache
# Änderbar: Sampling-Stärke (0.0 = deterministisch, höher = variabler)
DEFAULT_TEMPERATURE = 0.0

# ---------- Persistenz für Einstellungen ----------
last_lang = DEFAULT_LANG
temperature = DEFAULT_TEMPERATURE

if os.path.exists(CFG_PATH):
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as fh:
            config = json.load(fh)
            last_lang = config.get("language", DEFAULT_LANG)
            temperature = config.get("temperature", DEFAULT_TEMPERATURE)
    except Exception as e:
        print(f"Config load error: {e}", file=sys.stderr)


def save_config(lang: str, temp: float) -> None:
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as fh:
            json.dump({"language": lang, "temperature": temp}, fh)
    except Exception as e:
        print(f"Config save error: {e}", file=sys.stderr)


# ---------- Tk‑GUI basics ----------
root, status = tk.Tk(), tk.StringVar(value="Idle")
root.withdraw()
sel_lang = tk.StringVar(value=last_lang)
sel_temperature = tk.DoubleVar(value=temperature)
overlay: Optional[tk.Toplevel] = None

# ---------- Recording thread ----------
audio_q: queue.Queue = queue.Queue()
stop_ev = threading.Event()
tmp_wav: Optional[str] = None
recording = False
transcribing = False  # Neuer Status für Transkription


def cb(indata, frames, time_info, status_info):
    """Collect microphone chunks into queue."""
    audio_q.put(indata.copy())


def recorder() -> None:
    with sf.SoundFile(tmp_wav, "w", samplerate=RATE, channels=CHANNELS) as f, \
            sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=cb):
        while not stop_ev.is_set():
            try:
                f.write(audio_q.get(timeout=0.25))
            except queue.Empty:
                pass


# ---------- Model loading ----------
load_evt = threading.Event()
model = None
processor = None


def load_model() -> None:
    global model, processor
    status.set("Loading Voxtral …")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU-Speicher leeren falls nötig
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU (CUDA not available)")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = VoxtralForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        status.set("Model ready")
        print(f"Model loaded on {device}")
    except Exception as e:
        status.set("Model load error")
        messagebox.showerror(
            "Error", f"Voxtral could not be loaded:\n{e}")
        print(f"Model load error: {e}", file=sys.stderr)
    finally:
        load_evt.set()

# ---------- Beep ----------


def beep():
    try:
        winsound.MessageBeep(winsound.MB_OK)
    except:
        print("Beep!")  # Fallback if winsound fails

# ---------- Transcription ----------


def transcribe(lang: str) -> None:
    global transcribing, overlay
    transcribing = True

    # Overlay für Transkription umschalten wenn es existiert
    if overlay and overlay.winfo_exists():
        switch_to_transcribe_mode()

    try:
        if not load_evt.is_set():
            load_evt.wait(120)
        if model is None or processor is None:
            status.set("Model not loaded")
            return

        lg = lang if lang in LANGS else "auto"

        status.set("Transcribing …")

        # Use the proper Voxtral transcription method
        if lg == "auto":
            # Auto-detect language
            inputs = processor.apply_transcrition_request(
                audio=tmp_wav,
                model_id=MODEL_NAME
            )
        else:
            # Specific language
            inputs = processor.apply_transcrition_request(
                language=lg,
                audio=tmp_wav,
                model_id=MODEL_NAME
            )

        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device, dtype=torch.bfloat16)

        # Generate transcription with proper sampling parameters
        temp = sel_temperature.get()
        generation_kwargs = {
            **inputs,
            "max_new_tokens": 500,
            "do_sample": temp > 0.0,  # Sampling nur wenn Temperatur > 0
            "top_p": 0.9 if temp > 0.0 else None,  # Nucleus sampling
            "repetition_penalty": 1.1,  # Verhindert Wiederholungen
            "pad_token_id": processor.tokenizer.eos_token_id
        }

        # Nur bei deterministischer Generierung (temp=0) diese Parameter setzen
        if temp == 0.0:
            generation_kwargs.update({
                "num_beams": 1,  # Greedy decoding
                "early_stopping": True
            })

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        # Decode the output
        decoded_outputs = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        text_result = decoded_outputs[0].strip() if decoded_outputs else ""

        if text_result:
            pyperclip.copy(text_result)
            status.set("In clipboard")
            beep()
            save_config(lg, sel_temperature.get())
            print(f"Transcribed: {text_result}")
        else:
            status.set("No text detected")

    except Exception as e:
        status.set("Transcription error")
        messagebox.showerror(
            "Error", f"Transcription failed:\n{e}")
        print(f"Transcription error: {e}", file=sys.stderr)

    finally:
        transcribing = False
        # Overlay nach Abschluss der Transkription schließen
        if overlay and overlay.winfo_exists():
            overlay.destroy()
            overlay = None

# ---------- Overlay ----------


def switch_to_transcribe_mode():
    """Schaltet das Overlay von Aufnahme- in Transkriptionsmodus."""
    if not overlay or not overlay.winfo_exists():
        return

    # Canvas finden und Farbe ändern
    for widget in overlay.winfo_children():
        if isinstance(widget, tk.Canvas):
            # Hintergrund und Punkt zu blau ändern
            widget.config(bg="#000080")  # Dunkelblau
            # Punkt zu blau ändern
            for item in widget.find_all():
                if widget.type(item) == "oval":
                    widget.itemconfig(item, fill="#0080ff")  # Hellblau
            break

    # Hintergrund des Overlays ändern
    overlay.config(bg="#000080")


def make_overlay() -> tk.Toplevel:
    ov = tk.Toplevel(bg="#7f0000")
    ov.title("Recording")
    ov.attributes("-topmost", True)
    ov.geometry("+120+120")
    ov.resizable(False, False)

    cv = tk.Canvas(ov, width=50, height=50, bg="#7f0000", highlightthickness=0)
    dot = cv.create_oval(5, 5, 45, 45, fill="#ff0000")
    cv.grid(row=0, column=0, rowspan=3, padx=10, pady=10)

    def pulse():
        if (recording or transcribing) and ov.winfo_exists():
            try:
                current_fill = cv.itemcget(dot, "fill")

                if recording:
                    # Rot pulsieren während Aufnahme
                    new_fill = "#aa0000" if current_fill == "#ff0000" else "#ff0000"
                elif transcribing:
                    # Blau pulsieren während Transkription
                    new_fill = "#0060cc" if current_fill == "#0080ff" else "#0080ff"

                cv.itemconfig(dot, fill=new_fill)
                ov.after(600, pulse)
            except tk.TclError:
                # Window was destroyed
                pass

    pulse()

    # Sprache
    ttk.Label(ov, text="Language:").grid(row=0, column=1, sticky="w")
    lang_combo = ttk.Combobox(
        ov, values=LANGS, textvariable=sel_lang, width=6, state="readonly")
    lang_combo.grid(row=0, column=2, padx=(5, 10))

    # Temperatur
    ttk.Label(ov, text="Sampling:").grid(row=1, column=1, sticky="w")
    temp_scale = tk.Scale(ov, from_=0.0, to=1.0, resolution=0.1,
                          variable=sel_temperature, orient="horizontal", length=100)
    temp_scale.grid(row=1, column=2, padx=(5, 10))

    # Stop-Button
    stop_btn = ttk.Button(ov, text="Stop", command=hotkey)
    stop_btn.grid(row=0, column=3, rowspan=2, padx=10, sticky="ns")

    # Status
    ttk.Label(ov, textvariable=status).grid(row=2, column=1,
                                            columnspan=3, sticky="w", padx=5, pady=(0, 8))

    def on_close():
        # Nur schließen wenn nicht gerade transkribiert wird
        if not transcribing:
            hotkey()

    ov.protocol("WM_DELETE_WINDOW", on_close)
    return ov

# ---------- Hotkey ----------


def hotkey() -> None:
    global recording, overlay, tmp_wav, transcribing

    # Verhindern, dass während Transkription gestoppt wird
    if transcribing:
        return

    if not recording:
        status.set("Recording …")
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        stop_ev.clear()
        threading.Thread(target=recorder, daemon=True).start()
        overlay = make_overlay()
        recording = True
        if not load_evt.is_set():
            threading.Thread(target=load_model, daemon=True).start()
    else:
        stop_ev.set()
        recording = False
        # Overlay NICHT schließen, sondern für Transkription beibehalten
        threading.Thread(target=transcribe, args=(
            sel_lang.get(),), daemon=True).start()

# ---------- Start ----------


if __name__ == "__main__":
    if sys.version_info >= (3, 13):
        print("Warning: Python 3.13 is not fully supported by PyTorch/Transformers yet. Use 3.10/3.11.")

    print("Ctrl+R starts/stops recording")
    print("Overlay shows language and sampling settings")
    print("Sampling: 0.0 = precise/deterministic, 1.0 = variable/creative")
    print("Overlay stays visible during transcription (blue = transcribing)")
    keyboard.add_hotkey("ctrl+r", hotkey)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except:
                pass
