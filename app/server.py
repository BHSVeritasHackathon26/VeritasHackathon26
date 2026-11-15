"""
Thought-to-Text demo server.

Run this on the computer that has the trained model:

    pip install -r requirements.txt
    python server.py

Then open the printed URL in a browser (any device on the same network).
Connect the model by editing model_adapter.py.
"""

import base64
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import traceback
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path

import h5py
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import model_adapter

HERE = Path(__file__).parent
PORT = int(os.environ.get("PORT", "8000"))

STATE = {"model": None, "demo": True, "error": None, "load_seconds": 0.0}
PREDICT_LOCK = threading.Lock()  # one trial at a time through the model


@asynccontextmanager
async def lifespan(_app):
    t0 = time.perf_counter()
    try:
        STATE["model"] = model_adapter.load_model()
        STATE["demo"] = STATE["model"] is None
    except Exception:
        STATE["error"] = traceback.format_exc()
        print("\n!! load_model() failed -- running in demo mode:\n" + STATE["error"])
    STATE["load_seconds"] = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print("  DEMO MODE (no model connected yet)" if STATE["demo"]
          else f"  Model loaded in {STATE['load_seconds']:.1f}s")
    print(f"  On this computer:   http://localhost:{PORT}")
    ip = _lan_ip()
    if ip:
        print(f"  On the same Wi-Fi:  http://{ip}:{PORT}")
    print("=" * 60 + "\n")
    yield


app = FastAPI(lifespan=lifespan)


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the LAN interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def _to_str(v):
    """HDF5 strings arrive as str, bytes, numpy scalars or char-code arrays."""
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").strip("\x00").strip()
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "iu":  # array of character codes, zero padded
            return bytes(int(c) for c in v.ravel() if 0 < c < 256).decode("utf-8", "replace").strip()
        if v.size == 1:
            return _to_str(v.ravel()[0])
        return " ".join(_to_str(x) for x in v.ravel())
    return str(v).strip()


def _to_int(v):
    try:
        return int(np.asarray(v).ravel()[0])
    except Exception:
        return None


def find_trials(f):
    """Every group holding an `input_features` dataset is one trial."""
    if "input_features" in f:
        return [("trial", f)]
    out = []
    for key in sorted(f.keys()):
        g = f[key]
        if isinstance(g, h5py.Group) and "input_features" in g:
            out.append((key, g))
    return out


def make_preview(x, rows=32, cols=120):
    """Small 8-bit heat-map of the neural activity, for the UI."""
    try:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.size == 0:
            return None
        T, C = x.shape
        t_edges = np.linspace(0, T, min(cols, T) + 1).astype(int)
        c_edges = np.linspace(0, C, min(rows, C) + 1).astype(int)
        xt = np.add.reduceat(x, t_edges[:-1], axis=0) / np.diff(t_edges)[:, None]
        xc = np.add.reduceat(xt, c_edges[:-1], axis=1) / np.diff(c_edges)[None, :]
        lo, hi = np.percentile(xc, [2, 98])
        img = np.clip((xc - lo) / (hi - lo + 1e-6), 0, 1).T  # rows = channels
        return {"w": int(img.shape[1]), "h": int(img.shape[0]),
                "data": base64.b64encode((img * 255).astype(np.uint8).tobytes()).decode()}
    except Exception:
        return None


DEMO_SENTENCES = ["This is a placeholder sentence", "The model is not connected yet",
                  "Real predictions will appear here"]


def run_model(features, info, index):
    """Inference only. This server never trains: there is no optimizer, no loss
    and no label reading anywhere in it, and gradients are switched off below."""
    if STATE["demo"]:
        time.sleep(0.12)  # so the demo feels like real decoding
        return {"text": DEMO_SENTENCES[index % len(DEMO_SENTENCES)]}
    with PREDICT_LOCK, _no_grad():
        out = model_adapter.predict(STATE["model"], features, info)
    if isinstance(out, dict):
        return {"text": str(out.get("text", "")).strip(), "phonemes": out.get("phonemes")}
    return {"text": str(out).strip()}


def _no_grad():
    """torch.inference_mode() when PyTorch is installed: no gradients can be computed."""
    try:
        import torch
        return torch.inference_mode()
    except ImportError:
        return nullcontext()


def _line(obj):
    return json.dumps(obj) + "\n"


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status():
    return {"ok": True, "demo": STATE["demo"], "model_name": model_adapter.MODEL_NAME,
            "load_error": STATE["error"]}


@app.post("/api/decode")
def decode(file: UploadFile = File(...)):
    """Upload one HDF5 file; results stream back one JSON line per trial.

    Only `input_features` and the session/block/trial ids are read. The upload is
    a temporary copy that is deleted as soon as decoding ends."""
    tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        f = h5py.File(tmp.name, "r")
    except Exception as e:
        tmp.close()
        os.unlink(tmp.name)
        return JSONResponse({"error": f"Could not read this as an HDF5 file ({e})."}, status_code=400)

    def stream():
        try:
            trials = find_trials(f)
            yield _line({"type": "start", "file": file.filename, "n_trials": len(trials),
                         "demo": STATE["demo"]})
            if not trials:
                yield _line({"type": "fatal",
                             "error": "No trials found: expected groups containing an 'input_features' dataset."})
                return
            for i, (key, g) in enumerate(trials):
                msg = {"type": "trial", "index": i, "key": key}
                try:
                    features = g["input_features"][:].astype(np.float32, copy=False)
                    info = {"key": key, "filename": file.filename,
                            "session": _to_str(g.attrs.get("session")),
                            "block_num": _to_int(g.attrs.get("block_num")),
                            "trial_num": _to_int(g.attrs.get("trial_num"))}
                    t0 = time.perf_counter()
                    out = run_model(features, info, i)
                    msg.update(out, ms=round((time.perf_counter() - t0) * 1000, 1),
                               session=info["session"], block=info["block_num"], trial=info["trial_num"],
                               steps=int(features.shape[0]), preview=make_preview(features))
                except Exception as e:
                    traceback.print_exc()
                    msg["error"] = f"{type(e).__name__}: {e}"
                yield _line(msg)
            yield _line({"type": "done"})
        finally:
            f.close()
            os.unlink(tmp.name)

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    # 0.0.0.0 lets teammates on the same network reach the app.
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
