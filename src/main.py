import os
import cv2
import pandas as pd
import numpy as np

from clustering import cluster_images
from selector import compute_cluster_scores


DATA_FOLDER = r"D:\01\Photo Selector\data"
OUTPUT_FOLDER = r"D:\01\Photo Selector\output"


# -------------------------------
# Load images
# -------------------------------
def load_images(folder):

    images = []

    for file in os.listdir(folder):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(folder, file)

            image = cv2.imread(path)

            if image is not None:
                images.append((file, image))

    return images


# -------------------------------
# Resize ONLY for processing
# -------------------------------
def resize_image(image, size=(224, 224)):  # FIXED (not 2048)
    return cv2.resize(image, size)


# -------------------------------
# Visualization
# -------------------------------
def create_cluster_visual(cluster_id, images, save_path):

    thumbs = []

    for name, img in images:

        thumb = cv2.resize(img, (150, 150))

        cv2.putText(
            thumb,
            name[:10],
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

        thumbs.append(thumb)

    if len(thumbs) == 0:
        return

    cols = 5
    rows = (len(thumbs) + cols - 1) // cols

    grid = []

    for r in range(rows):

        row_imgs = thumbs[r * cols:(r + 1) * cols]

        while len(row_imgs) < cols:
            row_imgs.append(255 * np.ones_like(thumbs[0]))

        grid.append(cv2.hconcat(row_imgs))

    grid_image = cv2.vconcat(grid)

    cv2.imwrite(save_path, grid_image)


# -------------------------------
# Main
# -------------------------------
def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load ORIGINAL images
    images = load_images(DATA_FOLDER)

    if len(images) == 0:
        print("No images found.")
        return

    # Create lookup for original images
    original_dict = dict(images)

    # Create resized versions for processing
    resized_images = [
        (name, resize_image(img))
        for name, img in images
    ]

    # -------------------------------
    # CLUSTERING
    # -------------------------------
    clusters = cluster_images(resized_images)

    print(f"\nTotal clusters found: {len(clusters)}")

    all_results = []

    # -------------------------------
    # PROCESS CLUSTERS
    # -------------------------------
    for cluster_id, cluster_imgs in clusters.items():

        print(f"\nProcessing Cluster {cluster_id} | Images: {len(cluster_imgs)}")

        # -------------------------------
        # NEW: CLUSTER-BASED SCORING
        # -------------------------------
        scores = compute_cluster_scores(cluster_imgs)

        # Find best image
        best = max(scores, key=lambda x: x["score"])
        best_name = best["image"]

        # Get ORIGINAL high-quality image
        best_image = original_dict[best_name]

        # Save best image
        out_path = os.path.join(
            OUTPUT_FOLDER,
            f"cluster_{cluster_id}_best.jpg"
        )

        cv2.imwrite(out_path, best_image)

        print(f"Best in cluster {cluster_id}: {best_name}")

        # Save results
        for s in scores:
            all_results.append({
                "cluster": cluster_id,
                "image": s["image"],
                "score": s["score"]
            })

        # -------------------------------
        # VISUALIZATION
        # -------------------------------
        vis_path = os.path.join(
            OUTPUT_FOLDER,
            f"cluster_{cluster_id}_grid.jpg"
        )

        create_cluster_visual(cluster_id, cluster_imgs, vis_path)

    # -------------------------------
    # SAVE CSV
    # -------------------------------
    df = pd.DataFrame(all_results)

    df.to_csv(
        os.path.join(OUTPUT_FOLDER, "cluster_scores.csv"),
        index=False
    )

    print("\nProcessing complete.")


# -------------------------------
if __name__ == "__main__":
    main()