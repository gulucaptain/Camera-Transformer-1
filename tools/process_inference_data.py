"""Uniformly sample a fixed number of frames from each video listed in a CSV
and re-encode them as short MP4 clips (used to build inference cases).
"""

import argparse
import os

import imageio
import numpy as np
import pandas as pd
from decord import VideoReader
from tqdm import tqdm


def sample_frames_as_mp4(video_pth, out_pth, num_frames=75, fps=None):
    vr = VideoReader(video_pth)
    video_length = len(vr)
    height, width = vr[0].shape[0], vr[0].shape[1]

    # Uniformly sample `num_frames` frame indices across the whole video.
    indices = np.linspace(0, video_length - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()  # shape = [num_frames, H, W, 3]

    writer = imageio.get_writer(out_pth, fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved sampled {num_frames}-frame video to {out_pth}")

    return video_length, height, width


def main():
    parser = argparse.ArgumentParser(description="Sample frames from videos into MP4 clips.")
    parser.add_argument("--csv-file", required=True,
                        help="CSV with 'video_pth' and 'caption' columns.")
    parser.add_argument("--output-dir", required=True, help="Directory to write clips to.")
    parser.add_argument("--num-frames", type=int, default=75)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = pd.read_csv(args.csv_file)
    for i in tqdm(range(len(metadata))):
        video_pth = metadata.iloc[i].to_dict()["video_pth"]
        out_pth = os.path.join(args.output_dir, os.path.basename(video_pth))
        sample_frames_as_mp4(video_pth, out_pth=out_pth, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
