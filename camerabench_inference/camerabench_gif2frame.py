import argparse
import os

from PIL import Image
from tqdm import tqdm


def extract_first_frame(gif_path, output_path):
    try:
        with Image.open(gif_path) as im:
            im.seek(0)
            # Convert to RGB: GIFs are usually palette ("P") mode, and JPEG
            # does not support transparency, so conversion is required.
            rgb_im = im.convert("RGB")
            rgb_im.save(output_path, "JPEG", quality=95)
            print(f"Saved first frame to: {output_path}")
    except Exception as e:
        print(f"Failed to process {gif_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract the first frame of each GIF as a JPEG.")
    parser.add_argument("--gif-dir", required=True, help="Directory containing .gif files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write .jpg frames to.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for gif in tqdm(os.listdir(args.gif_dir)):
        gif_name = gif.split(".gif")[0]
        input_gif = os.path.join(args.gif_dir, gif)
        output_jpg = os.path.join(args.output_dir, f"{gif_name}.jpg")
        extract_first_frame(input_gif, output_jpg)


if __name__ == "__main__":
    main()
