import torch
import os
import glob
import pandas as pd
from tqdm import tqdm

from wavelet import wavelet_validate_extrinsics
from visual import visualize_empirical_validation, visualize_empirical_validation2

def extract_stats_from_single_result(out, identifier_name):
    """
    修复版：能够处理 [1, 6] 格式的 Tensor，取平均值作为标量。
    """
    # --- 辅助函数：安全地提取标量 ---
    def get_val(tensor_or_val):
        if torch.is_tensor(tensor_or_val):
            t = tensor_or_val.detach().cpu()
            # 关键修复：如果 Tensor 包含多个元素 (例如 6个自由度)，取平均值
            if t.numel() > 1:
                return t.mean().item()
            return t.item()
        return tensor_or_val

    # 1. 提取能量信息
    energy_dict = out["energy"]
    
    # ratio_low 是 [1, 6]，get_val 会自动取平均
    ratio_low = get_val(energy_dict["ratio_low"])
    
    # 计算所有高频分量的总和 (D1 + D2 + ...)
    # 你的 key 是 ratio_D1, ratio_D2...
    ratio_high_sum = sum([get_val(v) for k, v in energy_dict.items() if k.startswith("ratio_D")])
    
    # 单独提取 D1 (Jitter)
    ratio_d1 = get_val(energy_dict.get("ratio_D1", 0.0))

    # 2. 提取平滑度信息 (注意你的字典结构是 acc_energy)
    sm_dict = out["smoothness"]
    # gt/low/high 里面的 acc_energy 已经是 [1] 或 标量了，直接 get_val 即可
    acc_gt = get_val(sm_dict["gt"]["acc_energy"])
    acc_low = get_val(sm_dict["low_only"]["acc_energy"])
    acc_high = get_val(sm_dict["high_only"]["acc_energy"])
    
    # 3. 提取误差信息
    err_dict = out["errors"]
    err_low_rmse = get_val(err_dict["low_only"]["trans_rmse"])
    err_high_rmse = get_val(err_dict["high_only"]["trans_rmse"])

    # 4. 组装成一行数据
    row = {
        "batch_idx": identifier_name,
        
        # --- 论点 1: 能量分布 ---
        "Energy_Low (A)": ratio_low,
        "Energy_High (D_all)": ratio_high_sum,
        "Energy_D1 (Jitter)": ratio_d1,
        
        # --- 论点 3: 语义趋势 vs 局部扰动 (平滑度) ---
        "Smoothness_GT": acc_gt,
        "Smoothness_Low (Semantic)": acc_low,
        "Smoothness_High (Noise)": acc_high,
        
        # --- 论点 2: 重构能力 ---
        "Error_Low_RMSE": err_low_rmse,
        "Error_High_RMSE": err_high_rmse,
    }
    return row

# ==========================================
# 2. 批量处理器 (Batch Processor)
# ==========================================
def process_dataset(data_source, device="cuda"):
    """
    data_source: 可以是文件夹路径 (str)，也可以是包含 .pt 文件路径的 list
    """
    stats_list = []

    extrinsic_dir = os.listdir(data_source)
    extrinsic_dir = extrinsic_dir[:2000]
    
    extrinsic_dir = new_extrinsic_dir
    file_paths = []
    for extr_dir in extrinsic_dir:
        extrinsic_pth = os.path.join(data_source, extr_dir, "extrinsic.pt")
        if os.path.exists(extrinsic_pth):
            file_paths.append(extrinsic_pth)
    
    # # 1. 获取文件列表
    # if isinstance(data_source, str) and os.path.isdir(data_source):
    #     # 假设是文件夹，搜索所有 .pt 文件
    #     file_paths = glob.glob(os.path.join(data_source, "*.pt"))
    #     file_paths.sort()
    # elif isinstance(data_source, list):
    #     file_paths = data_source
    # else:
    #     raise ValueError("data_source must be a folder path or a list of file paths")

    print(f"Found {len(file_paths)} trajectory files. Processing...")

    # 2. 循环处理
    for fpath in tqdm(file_paths):
        try:
            # 加载数据 [F, 3, 4] 或 [1, F, 3, 4]
            extr = torch.load(fpath, map_location=device)
            
            # 确保维度是 [1, F, 3, 4]
            if extr.dim() == 3:
                extr = extr.unsqueeze(0)
            
            # === 核心调用: 你的验证函数 ===
            out = wavelet_validate_extrinsics(extr, levels=3)
            
            # 提取统计量
            file_id = os.path.basename(fpath) # 用文件名做 ID
            row = extract_stats_from_single_result(out, file_id)
            stats_list.append(row)
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # 3. 转为 DataFrame
    df = pd.DataFrame(stats_list)
    return df

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate wavelet statistics over a trajectory dataset.")
    parser.add_argument("--data-folder", required=True,
                        help="Directory of per-instance folders each containing extrinsic.pt.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_results = process_dataset(args.data_folder, device=device)
    
    print("\nVisualizing...")
    try:
        visualize_empirical_validation2(df_results)
        print("Visualization complete.")
    except NameError:
        print("Please paste the 'visualize_empirical_validation' function definition before running.")
