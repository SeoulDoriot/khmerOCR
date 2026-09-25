"""Small-text dataset generation — your original script, two fixes applied.

WHAT CHANGED FROM YOUR VERSION, AND WHY
---------------------------------------
1. create_small_sample() no longer feeds the recognizer.
   prep() rescales by HEIGHT to 64px. A 1024x256 canvas is divided by 4, so
   text pasted at 24px arrives at the model 6px tall; the 1536x384 canvas
   divides by 6, so 12px text arrives 2px tall. Khmer subscripts do not
   survive that. The canvas is correct for the DETECTOR, which sees the whole
   page -- it is wrong for the recognizer, which sees crops.

2. Added create_degraded_sample() for the recognizer.
   Same heights, same blur, same JPEG range you chose -- applied to the line
   crop itself and then restored to its original size. Detail is destroyed,
   geometry is preserved, so the aspect ratio the model trained on is intact.

3. labels.csv now carries x0,y0,x1,y1 in detector mode. Without the box the
   canvas images cannot train a detector, which is step 3 of your flow.
"""
from PIL import Image, ImageFilter
import pandas as pd
import os
import io
import random

# =========================
# PATHS
# =========================

def _base():
    """Kaggle writes to /kaggle/working, Colab to /content, otherwise cwd."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    if os.path.isdir("/content"):
        return "/content"
    return os.getcwd()


BASE = _base()

SOURCE_IMAGES = os.path.join(BASE, "existing_dataset", "images")
SOURCE_CSV    = os.path.join(BASE, "existing_dataset", "labels.csv")

SMALL_DIR      = os.path.join(BASE, "dataset_07_small")
VERY_SMALL_DIR = os.path.join(BASE, "dataset_08_very_small")

os.makedirs(f"{SMALL_DIR}/images", exist_ok=True)
os.makedirs(f"{VERY_SMALL_DIR}/images", exist_ok=True)


# =========================
# SETTINGS
# =========================

# Text height after shrinking
SMALL_HEIGHTS      = [24, 28, 32, 36]
VERY_SMALL_HEIGHTS = [12, 14, 16, 18, 20]

# Large background so text appears small inside image (DETECTOR ONLY)
SMALL_CANVAS      = (1024, 256)
VERY_SMALL_CANVAS = (1536, 384)

SEED = 42
random.seed(SEED)


# =========================
# RECOGNIZER  (use this to train the OCR model)
# =========================

def create_degraded_sample(image, target_height, add_blur=True, jpeg=True):
    """Shrink to target_height, then restore -- keeps shape, destroys detail.

    This is what a photo of small print actually loses. The recognizer sees a
    normal-shaped line that happens to be soft, which is learnable.
    """
    w, h = image.size
    small_width = max(4, int(w * target_height / max(h, 1)))

    # down -- the information is lost here
    out = image.resize(
        (small_width, target_height),
        Image.Resampling.BILINEAR
    )

    if add_blur and random.random() < 0.30:
        out = out.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.3, 0.8)
            )
        )

    # back up to the original size -- blurry, same aspect ratio
    out = out.resize((w, h), Image.Resampling.BICUBIC)

    if jpeg:
        buf = io.BytesIO()
        out.convert("RGB").save(buf, "JPEG", quality=random.randint(40, 80))
        out = Image.open(buf)

    return out


# =========================
# DETECTOR  (use this to train text detection, step 2-4 of the flow)
# =========================

def create_small_sample(image, target_height, canvas_size, add_blur=False):
    """Your original function. Now returns the bounding box as well."""
    w, h = image.size
    new_width = int(w * target_height / max(h, 1))

    resized = image.resize(
        (new_width, target_height),
        Image.Resampling.LANCZOS
    )

    canvas_w, canvas_h = canvas_size

    # prevent crop larger than canvas
    if new_width >= canvas_w:
        new_width = canvas_w - 40
        new_height = int(resized.height * new_width / resized.width)
        resized = resized.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    canvas = Image.new("RGB", canvas_size, "white")

    max_x = max(10, canvas_w - resized.width - 10)
    max_y = max(10, canvas_h - resized.height - 10)
    x = random.randint(10, max_x)
    y = random.randint(10, max_y)

    canvas.paste(resized, (x, y))

    if add_blur and random.random() < 0.25:
        canvas = canvas.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.2, 0.7)
            )
        )

    box = (x, y, x + resized.width, y + resized.height)
    return canvas, box


# =========================
# BUILD
# =========================

def build(out_dir, heights, canvas_size, mode):
    """mode = 'recognizer' (degrade in place) or 'detector' (paste on canvas)"""
    df = pd.read_csv(SOURCE_CSV)
    labels = []

    for i, row in df.iterrows():
        image_path = os.path.join(SOURCE_IMAGES, row["filename"])
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        target_h = random.choice(heights)
        new_name = f"{mode}_{i:07d}.jpg"
        rec = {"filename": new_name, "text": row["text"], "text_height": target_h}

        if mode == "recognizer":
            output = create_degraded_sample(img, target_h, add_blur=True)
        else:
            output, box = create_small_sample(
                img, target_h, canvas_size, add_blur=True
            )
            rec.update({
                "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                "canvas_w": canvas_size[0], "canvas_h": canvas_size[1],
            })

        output.save(
            os.path.join(out_dir, "images", new_name),
            quality=random.randint(75, 95)
        )
        labels.append(rec)

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(df)}", flush=True)

    pd.DataFrame(labels).to_csv(f"{out_dir}/labels.csv", index=False)
    print(f"{mode} dataset -> {out_dir}: {len(labels)} samples")
    return labels


if __name__ == "__main__":
    # dataset 07 -- small text, for training the recognizer
    build(SMALL_DIR, SMALL_HEIGHTS, SMALL_CANVAS, mode="recognizer")

    # dataset 08 -- very small text on a page, for training the detector
    build(VERY_SMALL_DIR, VERY_SMALL_HEIGHTS, VERY_SMALL_CANVAS, mode="detector")
