from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional

def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    兼容多版本 Pillow 的文本测量函数，返回 (width, height)
    优先使用 draw.textbbox -> font.getsize -> font.getmask 三种方式回退
    """
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            mask = font.getmask(text)
            return mask.size

def stitch_images_horizontal(
    image_paths: List[str],
    labels: List[str],
    caption: str,
    out_path: str = "stitched.jpg",
    label_font_path: Optional[str] = None,   # e.g. "arial.ttf" or None to use default
    caption_font_path: Optional[str] = None,
    label_font_size: int = 10,
    caption_font_size: int = 20,
    padding: int = 10,        # 间距：图片间与文本间
    bg_color=(255, 255, 255)  # 画布背景色
):
    assert len(image_paths) >= 1, "至少需要一张图"
    assert len(labels) == len(image_paths), "labels 数量必须和 image_paths 一致"

    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    heights = [im.height for im in imgs]
    target_h = min(heights)  # 按最短高度缩放

    target_h = 540 #FIXME

    resized = []
    for im in imgs:
        w, h = im.size
        scale = target_h / h
        new_w = int(round(w * scale))
        im_resized = im.resize((new_w, target_h), Image.LANCZOS)
        resized.append(im_resized)

    def load_font(path: Optional[str], size: int):
        try:
            if path:
                return ImageFont.truetype(path, size=size)
            for name in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                try:
                    return ImageFont.truetype(name, size=size)
                except Exception:
                    pass
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    label_font = load_font(label_font_path, label_font_size)
    caption_font = load_font(caption_font_path, caption_font_size)

    total_width = sum(im.width for im in resized) + padding * (len(resized) - 1)

    # 用临时 draw 测量 label 与 caption 的高度
    dummy = Image.new("RGB", (10, 10))
    draw_dummy = ImageDraw.Draw(dummy)
    label_heights = [measure_text(draw_dummy, lbl, label_font)[1] for lbl in labels]
    label_area_h = max(label_heights) + 6
    caption_h = measure_text(draw_dummy, caption, caption_font)[1] + 8

    canvas_h = caption_h + padding + target_h + padding + label_area_h + padding
    canvas = Image.new("RGB", (total_width, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)

    # 写 caption（上方居中）
    caption_w, _ = measure_text(draw, caption, caption_font)
    caption_x = (total_width - caption_w) // 2
    caption_y = padding
    draw.text((caption_x, caption_y), caption, fill=(0, 0, 0), font=caption_font)

    x = 0
    img_y = caption_y + caption_h + padding
    for im, lbl in zip(resized, labels):
        canvas.paste(im, (x, img_y))

        lbl_w, lbl_h = measure_text(draw, lbl, label_font)
        lbl_x = x + (im.width - lbl_w) // 2
        lbl_y = img_y + target_h + 4
        draw.text((lbl_x, lbl_y), lbl, fill=(0, 0, 0), font=label_font)

        x += im.width + padding

    canvas.save(out_path)
    print(f"Saved stitched image to {out_path}")

# === Example usage ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stitch images horizontally with a caption.")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths.")
    parser.add_argument("--caption", default=" ", help="Caption shown above the row.")
    parser.add_argument("--out-path", default="./out_stitched.jpg")
    args = parser.parse_args()

    labels = [" "] * len(args.images)
    stitch_images_horizontal(args.images, labels, args.caption, out_path=args.out_path)
