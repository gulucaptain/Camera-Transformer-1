import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def visualize_empirical_validation(df):
    sns.set_theme(style="white", context="talk", font_scale=1.0)
    
    # 颜色定义
    colors = {
        "GT":   "#2a9d8f",  # 青绿
        "Low":  "#4c72b0",  # 深蓝
        "High": "#c44e52"   # 柔红
    }

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle("Empirical Validation of Wavelet Decomposition\n"
                 r"Separation of Semantic Trend ($A$) vs. Local Disturbance ($D$)", 
                 fontsize=22, fontweight='bold', y=0.96)
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # =========================================================================
    # 辅助函数：网格设置
    # =========================================================================
    def set_clean_grid(ax, log_scale=False):
        ax.set_box_aspect(1)
        ax.set_axisbelow(True)
        # 主网格：线宽适当，颜色淡
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)
        
        if log_scale:
            # 次网格：只在对数轴保留
            ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=1.0, color='#f0f0f0')
        else:
            ax.grid(False, which='minor', axis='y')
            
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
            spine.set_linewidth(1.2)

    # -------------------------------------------------------------------------
    # Chart 1: 能量占比 (Violin + Strip)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    set_clean_grid(ax1)
    
    df_melt_energy = df.melt(id_vars=["batch_idx"], 
                             value_vars=["Energy_Low (A)", "Energy_High (D_all)"],
                             var_name="Component", value_name="Energy Ratio")
    
    # 1. Violin 背景
    sns.violinplot(x="Component", y="Energy Ratio", data=df_melt_energy, 
                   palette=[colors["Low"], colors["High"]],
                   inner=None, linewidth=0, alpha=0.15, saturation=0.7, ax=ax1)
    
    # 2. Box 统计框 (加宽)
    sns.boxplot(x="Component", y="Energy Ratio", data=df_melt_energy, 
                width=0.2, boxprops={'facecolor':'none', 'edgecolor':'#333333', 'linewidth': 1.5},
                showfliers=False, zorder=2, ax=ax1)
    
    # 3. Strip 数据点 (加大)
    sns.stripplot(x="Component", y="Energy Ratio", data=df_melt_energy, 
                  palette=[colors["Low"], colors["High"]],
                  size=4, alpha=0.6, jitter=0.25, zorder=3, ax=ax1)

    ax1.set_title("1. Energy Dominance", fontweight='bold', pad=15)
    ax1.set_ylabel("Energy Ratio")
    ax1.set_xlabel("")

    # -------------------------------------------------------------------------
    # Chart 2: 平滑度分析 (加大尺寸版)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    set_clean_grid(ax2)

    df_melt_smooth = df.melt(id_vars=["batch_idx"], 
                             value_vars=["Smoothness_GT", "Smoothness_Low (Semantic)", "Smoothness_High (Noise)"],
                             var_name="Signal Type", value_name="Accel Energy")
    
    palette_c2 = [colors["GT"], colors["Low"], colors["High"]]

    sns.violinplot(x="Component", y="Energy Ratio", data=df_melt_energy, 
                palette=[colors["Low"], colors["High"]],
                inner=None, linewidth=0, alpha=0.15, saturation=0.7, ax=ax2)
    
    # 增大 Box 宽度 (width=0.65)
    sns.boxplot(x="Signal Type", y="Accel Energy", data=df_melt_smooth,
                palette=palette_c2, width=0.65, 
                boxprops=dict(alpha=0.5, linewidth=1.5), 
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                fliersize=0, ax=ax2)
    
    # 增大散点尺寸 (size=10)
    sns.stripplot(x="Signal Type", y="Accel Energy", data=df_melt_smooth,
                  palette=palette_c2, size=4, alpha=0.85, jitter=0.2, 
                  edgecolor="white", linewidth=1.2, ax=ax2)
    
    ax2.set_yscale("log")
    ax2.set_title("2. Smoothness Analysis", fontweight='bold', pad=15)
    ax2.set_ylabel("Mean Accel Energy (Log Scale)")
    ax2.set_xlabel("")
    ax2.set_xticklabels(["GT", "Low ($A$)", "High ($D$)"])

    # -------------------------------------------------------------------------
    # Chart 3: 重构偏差 (加大尺寸版)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    set_clean_grid(ax3)
    
    df_melt_err = df.melt(id_vars=["batch_idx"], 
                          value_vars=["Error_Low_RMSE", "Error_High_RMSE"],
                          var_name="Reconstruction Mode", value_name="RMSE")
    
    palette_c3 = [colors["Low"], colors["High"]]
    
    # 增大 Box 宽度
    sns.boxplot(x="Reconstruction Mode", y="RMSE", data=df_melt_err,
                palette=palette_c3, width=0.6, 
                boxprops=dict(alpha=0.5, linewidth=1.5), 
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                fliersize=0, ax=ax3)
    
    # 增大散点尺寸
    sns.stripplot(x="Reconstruction Mode", y="RMSE", data=df_melt_err,
                  palette=palette_c3, size=4, alpha=0.85, jitter=0.2, 
                  edgecolor="white", linewidth=1.2, ax=ax3)
    
    ax3.set_title("3. Information Content", fontweight='bold', pad=15)
    ax3.set_ylabel("Translation RMSE (m)")
    ax3.set_xticklabels(["Keep Low\n(Fidelity)", "Keep High\n(Loss)"])

    # -------------------------------------------------------------------------
    # Chart 4: 闭环验证 (深度美化版)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    set_clean_grid(ax4)
    # X轴也加网格
    ax4.grid(True, which='major', axis='x', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)
    ax1.grid(True, which='major', axis='x', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)
    ax2.grid(True, which='major', axis='x', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)
    ax3.grid(True, which='major', axis='x', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)

    x = df["Energy_High (D_all)"]
    y = df["Error_Low_RMSE"]
    c_val = df["Smoothness_GT"]
    
    # 1. 计算相关系数
    r_val, _ = pearsonr(x, y)
    
    # 2. 绘制带阴影的回归线 (Regplot)
    # color设为深灰色，避免干扰散点颜色
    sns.regplot(x=x, y=y, scatter=False, ax=ax4, ci=95,
                color='#555555', 
                line_kws={'linestyle':'-', 'linewidth': 2, 'alpha': 0.8}, # 实线显得更确定
                scatter_kws={'alpha':0}) # 隐藏regplot自带的点
    
    # 3. 绘制高级气泡散点
    # 使用 plasma 色谱，s=180 (大气泡)
    sc = ax4.scatter(x, y, c=c_val, cmap="plasma", 
                     s=20, alpha=0.9, 
                     edgecolor="white", linewidth=1.5, # 增加白边厚度，制造立体感
                     zorder=3)
    
    # 4. 标注相关系数文本
    # 放在左上角或合适位置
    # ax4.text(0.05, 0.9, f"Correlation $r={r_val:.2f}$", 
    #          transform=ax4.transAxes, fontsize=16, fontweight='bold',
    #          bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    
    ax4.set_title("4. Jitter vs. Cost (Causality)", fontweight='bold', pad=15)
    ax4.set_xlabel("High-Freq Energy Ratio (Disturbance)")
    ax4.set_ylabel("Low-Only RMSE (Cost)")
    
    # 5. Colorbar 优化
    cbar = plt.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_ticks([])
    cbar.set_label("Original Trajectory Roughness", rotation=270, labelpad=25, fontweight='bold')
    cbar.outline.set_visible(False)
    # 加一点透明度让colorbar看起来不那么硬
    cbar.solids.set_alpha(1) 

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()
    plt.savefig("./visual.png")


def visualize_empirical_validation2(df):
    sns.set_theme(style="white", context="talk", font_scale=1.0)
    
    # 颜色定义
    colors = {
        "GT":   "#2a9d8f",  # 青绿
        "Low":  "#4c72b0",  # 深蓝
        "High": "#c44e52"   # 柔红
    }

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle("Empirical Validation of Wavelet Decomposition\n"
                 r"Separation of Semantic Trend ($A$) vs. Local Disturbance ($D$)", 
                 fontsize=22, fontweight='bold', y=0.96)
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # =========================================================================
    # 辅助函数：网格设置
    # =========================================================================
    def set_clean_grid(ax, log_scale=False):
        ax.set_box_aspect(1)
        ax.set_axisbelow(True)
        # 主网格：线宽适当，颜色淡
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)
        
        if log_scale:
            # 次网格：只在对数轴保留
            ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=1.0, color='#f0f0f0')
        else:
            ax.grid(False, which='minor', axis='y')
            
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
            spine.set_linewidth(1.2)

    # =========================================================================
    # 辅助函数：设置 Violin 透明度
    # (Seaborn 的 alpha 参数有时不生效，此函数确保背景变淡)
    # =========================================================================
    def style_violin_background(violin_ax, alpha=0.15):
        for collection in violin_ax.collections:
            if isinstance(collection, plt.matplotlib.collections.PolyCollection):
                collection.set_alpha(alpha)

    # -------------------------------------------------------------------------
    # Chart 1: 能量占比 (Violin + Box + Strip)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    set_clean_grid(ax1)
    
    df_melt_energy = df.melt(id_vars=["batch_idx"], 
                             value_vars=["Energy_Low (A)", "Energy_High (D_all)"],
                             var_name="Component", value_name="Energy Ratio")
    
    # 1. Violin 背景
    sns.violinplot(x="Component", y="Energy Ratio", data=df_melt_energy, 
                   palette=[colors["Low"], colors["High"]],
                   inner=None, linewidth=0, saturation=0.7, ax=ax1)
    style_violin_background(ax1, 0.15) # 强制设置透明度
    
    # 2. Box 统计框
    sns.boxplot(x="Component", y="Energy Ratio", data=df_melt_energy, 
                width=0.2, boxprops={'facecolor':'none', 'edgecolor':'#333333', 'linewidth': 1.5},
                showfliers=False, zorder=2, ax=ax1)
    
    # 3. Strip 数据点
    sns.stripplot(x="Component", y="Energy Ratio", data=df_melt_energy, 
                  palette=[colors["Low"], colors["High"]],
                  size=4, alpha=0.6, jitter=0.25, zorder=3, ax=ax1)

    ax1.set_title("1. Energy Dominance", fontweight='bold', pad=15)
    ax1.set_ylabel("Energy Ratio")
    ax1.set_xlabel("")

    # -------------------------------------------------------------------------
    # Chart 2: 平滑度分析 (已修正：正确的数据源 + 背景 Violin)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    set_clean_grid(ax2, log_scale=True) # 这里 log_scale=True 对网格很重要

    df_melt_smooth = df.melt(id_vars=["batch_idx"], 
                             value_vars=["Smoothness_GT", "Smoothness_Low (Semantic)", "Smoothness_High (Noise)"],
                             var_name="Signal Type", value_name="Accel Energy")
    
    palette_c2 = [colors["GT"], colors["Low"], colors["High"]]

    # === [新增/修正] Violin 背景 ===
    # 注意：这里改用了 df_melt_smooth 而不是 df_melt_energy
    sns.violinplot(x="Signal Type", y="Accel Energy", data=df_melt_smooth, 
                   palette=palette_c2,
                   inner=None, linewidth=0, saturation=0.7, ax=ax2)
    style_violin_background(ax2, 0.15) # 强制设置透明度
    
    # Box 统计框
    sns.boxplot(x="Signal Type", y="Accel Energy", data=df_melt_smooth,
                palette=palette_c2, width=0.65, 
                boxprops=dict(alpha=0.5, linewidth=1.5), 
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                fliersize=0, ax=ax2)
    
    # Strip 数据点
    sns.stripplot(x="Signal Type", y="Accel Energy", data=df_melt_smooth,
                  palette=palette_c2, size=4, alpha=0.85, jitter=0.2, 
                  edgecolor="white", linewidth=1.2, ax=ax2)
    
    ax2.set_yscale("log")
    ax2.set_title("2. Smoothness Analysis", fontweight='bold', pad=15)
    ax2.set_ylabel("Mean Accel Energy (Log Scale)")
    ax2.set_xlabel("")
    ax2.set_xticklabels(["GT", "Low ($A$)", "High ($D$)"])

    # -------------------------------------------------------------------------
    # Chart 3: 重构偏差 (已修正：新增背景 Violin)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    set_clean_grid(ax3)
    
    df_melt_err = df.melt(id_vars=["batch_idx"], 
                          value_vars=["Error_Low_RMSE", "Error_High_RMSE"],
                          var_name="Reconstruction Mode", value_name="RMSE")
    
    palette_c3 = [colors["Low"], colors["High"]]
    
    # === [新增] Violin 背景 ===
    sns.violinplot(x="Reconstruction Mode", y="RMSE", data=df_melt_err,
                   palette=palette_c3,
                   inner=None, linewidth=0, saturation=0.7, ax=ax3)
    style_violin_background(ax3, 0.15) # 强制设置透明度

    # Box 统计框
    sns.boxplot(x="Reconstruction Mode", y="RMSE", data=df_melt_err,
                palette=palette_c3, width=0.6, 
                boxprops=dict(alpha=0.5, linewidth=1.5), 
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                fliersize=0, ax=ax3)
    
    # Strip 数据点
    sns.stripplot(x="Reconstruction Mode", y="RMSE", data=df_melt_err,
                  palette=palette_c3, size=4, alpha=0.85, jitter=0.2, 
                  edgecolor="white", linewidth=1.2, ax=ax3)
    
    ax3.set_title("3. Information Content", fontweight='bold', pad=15)
    ax3.set_ylabel("Translation RMSE (m)")
    ax3.set_xticklabels(["Keep Low\n(Fidelity)", "Keep High\n(Loss)"])

    # -------------------------------------------------------------------------
    # Chart 4: 闭环验证
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    set_clean_grid(ax4)
    ax4.grid(True, which='major', axis='x', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)

    # 补充 X 轴网格给其他图（保持统一性）
    ax1.grid(True, which='major', axis='x', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)
    ax2.grid(True, which='major', axis='x', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)
    ax3.grid(True, which='major', axis='x', linestyle='--', linewidth=1.5, color='#e0e0e0', alpha=0.8)

    x = df["Energy_High (D_all)"]
    y = df["Error_Low_RMSE"]
    c_val = df["Smoothness_GT"]
    
    # 1. 计算相关系数
    r_val, _ = pearsonr(x, y)
    
    # 2. 回归线
    sns.regplot(x=x, y=y, scatter=False, ax=ax4, ci=95,
                color='#555555', 
                line_kws={'linestyle':'-', 'linewidth': 2, 'alpha': 0.8},
                scatter_kws={'alpha':0}) 
    
    # 3. 气泡散点
    sc = ax4.scatter(x, y, c=c_val, cmap="plasma", 
                     s=20, alpha=0.9, 
                     edgecolor="white", linewidth=1.5, 
                     zorder=3)
    
    ax4.set_title("4. Jitter vs. Cost (Causality)", fontweight='bold', pad=15)
    ax4.set_xlabel("High-Freq Energy Ratio (Disturbance)")
    ax4.set_ylabel("Low-Only RMSE (Cost)")
    
    # 5. Colorbar
    cbar = plt.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_ticks([])
    cbar.set_label("Original Trajectory Roughness", rotation=270, labelpad=25, fontweight='bold')
    cbar.outline.set_visible(False)
    cbar.solids.set_alpha(1) 

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./visual.png")
    # plt.show()
