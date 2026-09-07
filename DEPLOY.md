# Deploying the Plant Diagnosis System for Free

> Note: Hugging Face Spaces moved Docker/Gradio Spaces behind a paid plan
> (only static Spaces are free now), so the recommended free host is
> **Modal** (modal.com): every account gets **$30/month of free compute
> credits** on the Starter plan, sign-in is via GitHub, and containers scale
> to zero when idle — a demo workload stays well inside the free credits.
> Modal runs the FULL system: real TensorFlow, the gate model, feedback,
> and auto-retraining.

## Option 1 (recommended) — Modal

### One-time setup

```bash
pip install modal
modal setup
```

`modal setup` opens the browser — sign in with a GitHub account (free
Starter plan). If it ever demands a payment card, stop and use Option 2.

### Deploy

From this project folder:

```bash
modal deploy modal_app.py
```

Modal builds the container image (TensorFlow install takes a few minutes the
first time, then it's cached) and prints your public URL, which looks like:

```
https://<workspace>--plant-diagnosis-fastapi-app.modal.run
```

### Test

```bash
curl https://<workspace>--plant-diagnosis-fastapi-app.modal.run/health

curl -X POST -F "file=@test_dataset/fresh/R.jpg" \
  https://<workspace>--plant-diagnosis-fastapi-app.modal.run/predict
```

Any web or mobile frontend can call `POST /predict` on that URL directly —
CORS is already open in `prediction.py`.

### Things to know

- **Cold starts.** After idle time the container shuts down (that's what
  makes it free). The first request afterwards takes ~30–60 s while
  TensorFlow and the models load; requests after that are fast. Hit
  `/health` a couple of minutes before your demo to warm it up.
- **Ephemeral storage.** Feedback images and retrained weights live in the
  container and are lost when it scales down. The feedback → auto-retrain
  demo works live within a warm session; it just doesn't survive idle
  shutdown. (Fixable later with a `modal.Volume` if needed.)
- **Credits.** Usage is billed per second the container runs against the
  free $30/month. Scale-to-zero means an occasional demo uses cents.
- If `modal deploy` reports an API/version error, the Modal SDK may have
  changed since this file was written — paste the error to Claude to fix.

## Option 2 (fallback) — Render free tier + TFLite

Render (render.com) still has a genuinely free web-service tier — no credit
card, 512 MB RAM, 750 h/month, sleeps after 15 min idle (~1 min wake).

512 MB is too small for full TensorFlow, so this path requires converting
both models to **TFLite** and serving them with the lightweight runtime.
Consequences: retraining is not possible on the server (TFLite is
inference-only), and `prediction.py` needs a modified serving variant.
Ask Claude to generate the TFLite conversion + `prediction_lite.py` if you
need this path.

## Option 3 (demo-day backup) — tunnel from your own PC

Zero signup, zero cost, full functionality including retraining — but only
live while your PC is running the servers:

1. Download `cloudflared.exe` from
   https://github.com/cloudflare/cloudflared/releases (no account needed).
2. Start the system locally: `python prediction.py`
3. In another terminal:
   `cloudflared tunnel --url http://localhost:8000`
4. It prints a public `https://<random>.trycloudflare.com` URL — anyone
   (including your phone on mobile data) can hit `/predict` on it.

This is a good insurance policy for the defense: if the cloud host
misbehaves, the tunnel gets you a live public URL in under a minute.

## File map

| File | Used by |
|---|---|
| `modal_app.py` | Option 1 — Modal deployment wrapper |
| `Dockerfile` | generic Docker hosts (kept for future use; not needed for Modal) |
| `requirements.txt` | Modal image build and local installs |
