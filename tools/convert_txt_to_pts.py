import torch
import numpy as np

def txt_to_camera_pt_no_homogeneous(
    txt_pth,
    intrinsics_pt_pth,
    extrinsics_pt_pth,
    dtype=torch.float32,
    device="cpu"
):
    """
    Parse the txt produced by convert_camera_pose_to_txt into:
      - intrinsics: (T, 3, 3)
      - extrinsics: (T, 3, 4)  # the [0, 0, 0, 1] row is not stored
    """

    Ks = []
    Es = []

    with open(txt_pth, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 19:
                raise ValueError(
                    f"Line {line_idx} has {len(parts)} fields (<19): {line}"
                )

            # ---------- intrinsics ----------
            fx = float(parts[1])
            fy = float(parts[2])
            cx = float(parts[3])
            cy = float(parts[4])

            K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            # ---------- extrinsics (3x4 only) ----------
            ext_vals = list(map(float, parts[7:19]))  # 12 values
            E34 = np.array(ext_vals, dtype=np.float32).reshape(3, 4)

            Ks.append(K)
            Es.append(E34)

    Ks = torch.tensor(np.stack(Ks, axis=0), dtype=dtype, device=device)
    Es = torch.tensor(np.stack(Es, axis=0), dtype=dtype, device=device)

    torch.save(Ks, intrinsics_pt_pth)
    torch.save(Es, extrinsics_pt_pth)

    print(f"[✓] intrinsics saved to {intrinsics_pt_pth}, shape={Ks.shape}")
    print(f"[✓] extrinsics saved to {extrinsics_pt_pth}, shape={Es.shape}")

    return Ks, Es


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a camera-pose txt file into intrinsic.pt / extrinsic.pt tensors."
    )
    parser.add_argument("--txt-path", required=True)
    parser.add_argument("--intrinsics-out", required=True)
    parser.add_argument("--extrinsics-out", required=True)
    args = parser.parse_args()

    txt_to_camera_pt_no_homogeneous(
        txt_pth=args.txt_path,
        intrinsics_pt_pth=args.intrinsics_out,
        extrinsics_pt_pth=args.extrinsics_out,
    )
