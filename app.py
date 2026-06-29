"""Flask web UI for the face mask detection pipeline."""

import os
from pathlib import Path

import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import config
from enhance import enhance
from segment import segment
from clean import clean
from detect import detect
from decide import decide
from visualize import visualize


UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path(config.DEFAULT_OUTPUT_DIR)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _bgr_to_hex(bgr) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _stage_urls(base: str) -> dict:
    return {
        "original": f"/output/{base}_1_original.jpg",
        "enhanced": f"/output/{base}_2_enhanced.jpg",
        "segmentation": f"/output/{base}_3_segmentation.jpg",
        "cleaned": f"/output/{base}_4_cleaned.jpg",
        "detection": f"/output/{base}_5_detection.jpg",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    file = request.files["image"]

    if not file.filename or not _allowed(file.filename):
        return jsonify({"status": "error", "message": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    image = cv2.imread(str(save_path))
    if image is None:
        return jsonify({"status": "error", "message": "Cannot read image"}), 400

    enhanced = enhance(image)
    face_data = segment(enhanced)

    base = os.path.splitext(filename)[0]

    if not face_data:
        visualize(image, enhanced, [], [], str(OUTPUT_DIR), filename)
        return jsonify({
            "status": "ok",
            "image_name": filename,
            "stages": _stage_urls(base),
            "faces": [],
            "message": "Face not detected",
        })

    cleaned = clean(face_data)
    detections = detect(enhanced, cleaned)
    results = decide(detections)

    visualize(image, enhanced, cleaned, results, str(OUTPUT_DIR), filename)

    return jsonify({
        "status": "ok",
        "image_name": filename,
        "stages": _stage_urls(base),
        "faces": [
            {
                "face_id": r["face_id"],
                "label": r["label"],
                "color": _bgr_to_hex(r["color"]),
                "skin_ratio": round(r["skin_ratio"], 3),
                "upper_skin_ratio": round(r["upper_skin_ratio"], 3),
                "lower_skin_ratio": round(r["lower_skin_ratio"], 3),
                "non_skin_ratio": round(r["non_skin_ratio"], 3),
            }
            for r in results
        ],
    })


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
