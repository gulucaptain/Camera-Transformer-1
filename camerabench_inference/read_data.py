"""Build a CameraBench inference JSON from the test.jsonl annotations.

For each annotation, locate the extracted first-frame image and record the
(image path, prompt, labels) triple. Output is a JSON list consumable by
camerabench_inference.py.
"""

import argparse
import json
import os

import jsonlines


def main():
    parser = argparse.ArgumentParser(description="Build CameraBench inference JSON.")
    parser.add_argument("--jsonl-path", required=True, help="CameraBench test.jsonl annotations.")
    parser.add_argument("--frame-dir", required=True, help="Directory of extracted first-frame JPEGs.")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    args = parser.parse_args()

    infos = []
    with open(args.jsonl_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            video_name = item["path"].split("/")[1].split(".mp4")[0]
            image_pth = os.path.join(args.frame_dir, f"{video_name}.jpg")
            if os.path.exists(image_pth):
                infos.append(
                    {
                        "image_pth": image_pth,
                        "prompt": item["caption"],
                        "labels": item["labels"],
                    }
                )

    print(len(infos))
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
