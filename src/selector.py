from quality_metrics import (
    compute_sharpness,
    compute_brightness,
    compute_contrast
)

from face_detector import detect_faces_and_eyes
from brisque import compute_brisque
from nima import compute_nima

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-6)


def compute_raw_metrics(image):

    sharpness = compute_sharpness(image)
    brightness = compute_brightness(image)
    contrast = compute_contrast(image)

    face_data = detect_faces_and_eyes(image)

    face_count = len(face_data)

    if face_count > 0:
        open_eyes = sum(f["eyes_open"] for f in face_data)
        eye_score = open_eyes / face_count

        total_face_area = sum(f["area"] for f in face_data)
        img_area = image.shape[0] * image.shape[1]
        face_area_ratio = total_face_area / img_area
    else:
        eye_score = 0
        face_area_ratio = 0

    # -------------------------
    # NEW: BRISQUE
    # -------------------------
    brisque_score = compute_brisque(image)
    nima_score = compute_nima(image)
    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "face_count": face_count,
        "eye_score": eye_score,
        "face_area_ratio": face_area_ratio,
        "brisque": brisque_score,
        "nima": nima_score
    }


def compute_cluster_scores(cluster_images):

    raw = []

    for name, img in cluster_images:
        metrics = compute_raw_metrics(img)
        metrics["name"] = name
        raw.append(metrics)

    def get_range(key):
        vals = [r[key] for r in raw]
        return min(vals), max(vals)

    s_min, s_max = get_range("sharpness")
    b_min, b_max = get_range("brightness")
    c_min, c_max = get_range("contrast")
    br_min, br_max = get_range("brisque")
    nima_min, nima_max = get_range("nima")

    expected_faces = max(r["face_count"] for r in raw) + 1e-6

    results = []

    for r in raw:

        sharp_n = normalize(r["sharpness"], s_min, s_max)
        bright_n = normalize(r["brightness"], b_min, b_max)
        contrast_n = normalize(r["contrast"], c_min, c_max)
        nima_n = normalize(r["nima"], nima_min, nima_max)

        # BRISQUE: lower is better → invert
        brisque_n = normalize(r["brisque"], br_min, br_max)
        brisque_score = 1 - brisque_n

        face_score = r["face_count"] / expected_faces

        final_score = (
            0.35 * nima_n +
            0.20 * face_score +
            0.15 * r["eye_score"] +
            0.10 * r["face_area_ratio"] +
            0.10 * sharp_n +
            0.05 * bright_n +
            0.05 * (1 - brisque_n)
        )   

        results.append({
            "image": r["name"],
            "score": final_score,
            "brisque": r["brisque"]
        })

    return results