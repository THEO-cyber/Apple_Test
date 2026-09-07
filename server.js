require("dotenv").config();

const express  = require("express");
const multer   = require("multer");
const axios    = require("axios");
const FormData = require("form-data");
const cors     = require("cors");

const app            = express();
const PORT           = process.env.PORT || 3000;
const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://localhost:8000";

const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const MAX_FILE_SIZE      = 10 * 1024 * 1024; // 10 MB

app.use(cors());

const upload = multer({
  storage: multer.memoryStorage(),
  limits:  { fileSize: MAX_FILE_SIZE },
  fileFilter: (_req, file, cb) => {
    if (ALLOWED_MIME_TYPES.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type '${file.mimetype}'. Upload a JPEG, PNG, WEBP, or BMP image.`));
    }
  },
});

// ── Helpers ───────────────────────────────────────────────────────────────────
async function proxyGet(path, res) {
  try {
    const { data } = await axios.get(`${PYTHON_API_URL}${path}`, { timeout: 5000 });
    res.json(data);
  } catch (err) {
    const status = err.response?.status || 503;
    res.status(status).json({ success: false, message: err.response?.data?.detail || err.message });
  }
}

async function proxyPost(path, form, res, timeout = 30000) {
  try {
    const { data } = await axios.post(`${PYTHON_API_URL}${path}`, form, {
      headers: form.getHeaders(),
      timeout,
    });
    return data;
  } catch (err) {
    if (err.response) {
      res.status(err.response.status).json({
        success: false,
        message: err.response.data?.detail || "AI engine error.",
      });
    } else if (err.code === "ECONNREFUSED" || err.code === "ECONNRESET") {
      res.status(503).json({ success: false, message: "AI engine is unavailable. Please try again later." });
    } else {
      res.status(500).json({ success: false, message: err.message });
    }
    return null;
  }
}

// ── Routes ────────────────────────────────────────────────────────────────────
app.get("/health", async (_req, res) => {
  try {
    const { data } = await axios.get(`${PYTHON_API_URL}/health`, { timeout: 5000 });
    res.json({ status: "ok", ai_engine: data });
  } catch {
    res.status(503).json({ status: "degraded", ai_engine: "unreachable" });
  }
});

app.get("/classes", (_req, res) => proxyGet("/classes", res));

app.get("/retrain/status", (_req, res) => proxyGet("/retrain/status", res));

app.post("/predict", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, message: "No image file uploaded." });
  }

  const form = new FormData();
  form.append("file", req.file.buffer, {
    filename:    req.file.originalname,
    contentType: req.file.mimetype,
  });

  const data = await proxyPost("/predict", form, res);
  if (!data) return;

  if (!data.success) {
    return res.status(500).json({ success: false, message: "AI engine processing error." });
  }

  res.json({
    success:    true,
    message:    data.message,
    status:     data.status,
    confidence: data.confidence,
    danger:     data.danger,
    advice:     data.advice,
    warning:    data.warning ?? null,
    class_name: data.class_name,   // expose so the frontend can pre-fill the feedback form
  });
});

app.post("/feedback", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, message: "No image file uploaded." });
  }
  if (!req.body.label) {
    return res.status(400).json({ success: false, message: "Missing 'label' field." });
  }

  const form = new FormData();
  form.append("file", req.file.buffer, {
    filename:    req.file.originalname,
    contentType: req.file.mimetype,
  });
  form.append("label", req.body.label);

  const data = await proxyPost("/feedback", form, res);
  if (!data) return;

  res.json(data);
});

// ── Multer error handler ───────────────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({ success: false, message: "File too large. Maximum size is 10 MB." });
  }
  res.status(400).json({ success: false, message: err.message });
});

app.listen(PORT, "0.0.0.0", () =>
  console.log(`Plant Diagnosis Gateway  →  port ${PORT}  |  AI engine: ${PYTHON_API_URL}`)
);
