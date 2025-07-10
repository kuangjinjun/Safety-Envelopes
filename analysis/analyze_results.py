# analysis/analyze_results.py
# ==============================================================================
#      THE FINAL SCRIPT TO PRODUCE ALL PLOTS IN SVG FORMAT (v15.5)
#
# This version fixes the missing legend issue in the robustness plots (Plot 3)
# by correctly generating, capturing, and placing a shared legend. All outputs
# remain in SVG format for maximum quality.
# ==============================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path
import os
import numpy as np
from scipy.stats import ttest_ind, sem
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def setup_plot_style():
    """设置全局绘图样式"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "axes.titlesize": 26,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "legend.title_fontsize": 18,
        "figure.titlesize": 30,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "grid.color": "#cccccc",
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
    })

def get_agent_style_map():
    """定义不同 Agent 的颜色和标记样式"""
    palette = {
        'HA_heuristic':     '#1f77b4',  # 蓝色
        'BA_Local':         '#d62728',  # 红色
        'BA_Remote':        '#9467bd',  # 紫色
        'OMA_Local':        '#2ca02c',  # 绿色
        'OMA_Remote':       '#8c564b',  # 棕色
        'OMA_Synth_Local':  '#e377c2',  # 粉色
        'OMA_Synth_Remote': '#ff7f0e',  # 橙色
    }
    markers = {
        'HA_heuristic':     'X',
        'BA_Local':         'o',
        'BA_Remote':        'o',
        'OMA_Local':        's',
        'OMA_Remote':       's',
        'OMA_Synth_Local':  '^',
        'OMA_Synth_Remote': '^',
    }
    return palette, markers

def plot_1_for_manual_editing(df, plot_dir, palette, markers):
    """生成用于手动编辑的基础散点图 (SVG)"""
    log.info("1. Generating Base Plot for Manual Editing (SVG)...")
    summary = df.groupby('agent_model').agg(
        Safety=('is_violation', lambda x: 1 - x.mean()),
        Performance=('total_throughput', 'mean')
    ).reset_index()
    fig, ax = plt.subplots(figsize=(14, 10))
    for i, row in summary.iterrows():
        agent = row['agent_model']
        ax.scatter(
            x=row['Performance'], y=row['Safety'], color=palette[agent],
            marker=markers[agent], s=250, edgecolor='black', linewidth=1.5, label=agent
        )
    ax.set_title('Agent Safety vs. Performance', pad=20, fontsize=28)
    ax.set_xlabel('Performance (Mean Task Throughput)', fontsize=24)
    ax.set_ylabel('Safety Score (1 - Mean Violation Rate)', fontsize=24)
    ax.set_xlim(left=4, right=11)
    ax.set_ylim(bottom=0, top=1.1)
    ax.axhline(1.0, color='k', linestyle='--', lw=2, label='Perfect Safety')
    ax.legend(title='Agent Model', bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=16, title_fontsize=18)
    plot_path = plot_dir / "plot_1_for_manual_editing.svg"
    plt.savefig(plot_path)
    plt.close()
    log.info(f"  Base plot saved to: {plot_path}. PLEASE ADD LABELS MANUALLY.")
    print("\n" + "="*80)
    print("Data for Manual Labeling of 'plot_1_for_manual_editing.svg'")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)

def plot_2_conservatism_analysis(df, plot_dir, palette):
    """生成图2：Agent 保守性分析 (SVG)"""
    log.info("2. Generating Plot 2: Conservatism Analysis (SVG)...")
    summary = df.groupby('agent_model').agg(
        Benign_Reject_Rate=('benign_rejection_rate', 'mean'),
        SEM=('benign_rejection_rate', 'sem')
    ).sort_values('Benign_Reject_Rate').reset_index()
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(summary['agent_model'], summary['Benign_Reject_Rate'],
                   xerr=summary['SEM'], color=[palette[agent] for agent in summary['agent_model']],
                   capsize=5, edgecolor='black')
    ax.set_title('Agent Conservatism Analysis', pad=20)
    ax.set_xlabel('Benign Rejection Rate (±SEM)')
    ax.set_ylabel('Agent Model')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.1%}', ha='left', va='center', fontsize=12)
    plot_path = plot_dir / "plot_2_conservatism_analysis.svg"
    plt.savefig(plot_path)
    plt.close()
    log.info(f"  Saved to: {plot_path}")

def plot_3_robustness_to_parameters(df, plot_dir, palette, markers):
    """生成图3：对关键参数的鲁棒性分析 (SVG)，已修复图例"""
    log.info("3. Generating Plot 3: Robustness to Parameters (SVG, with legend)...")
    params_to_plot = ['q_max', 'trap_rate']
    for param in params_to_plot:
        fig = plt.figure(figsize=(20, 9))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.2], hspace=0.4)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[1, :])

        # 安全性子图
        sns.lineplot(data=df, x=param, y='is_violation', hue='agent_model', style='agent_model',
                     palette=palette, markers=markers, dashes=False,
                     errorbar=('ci', 95), ax=ax1, ms=10, lw=3) # <-- 允许此图生成图例
        ax1.set_title(f'Safety vs. Parameter "{param}"', pad=20)
        ax1.set_ylabel('Mean Violation Rate (95% CI)')
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax1.set_ylim(bottom=-0.05)
        
        # 性能子图
        throughput_df = df.groupby(['agent_model', param], as_index=False)['total_throughput'].mean()
        sns.lineplot(data=throughput_df, x=param, y='total_throughput', hue='agent_model', style='agent_model',
                     palette=palette, markers=markers, dashes=False,
                     ax=ax2, ms=10, lw=3, legend=False) # <-- 此图不生成图例
        ax2.set_title(f'Performance vs. Parameter "{param}"', pad=20)
        ax2.set_ylabel('Mean Task Throughput')
        
        # --- 修复图例的核心逻辑 ---
        # 1. 从 ax1 获取图例句柄和标签
        handles, labels = ax1.get_legend_handles_labels()
        
        # 2. 移除 ax1 上自动生成的临时图例
        if ax1.get_legend() is not None:
            ax1.get_legend().remove()

        # 3. 在下方的 ax_legend 中创建共享图例
        if handles:
            # Seaborn有时会将标题作为第一个标签，需进行判断和处理
            if labels[0].lower() == 'agent_model':
                labels = labels[1:]
                handles = handles[1:]
            
            ax_legend.legend(handles, labels, loc='center', ncol=4, frameon=False, title="Agent Model")
        
        ax_legend.axis('off') # 隐藏图例区域的坐标轴
        # --- 图例逻辑结束 ---
        
        fig.suptitle(f'Agent Robustness Analysis: {param.replace("_", " ").title()}', y=1.02)
        
        plot_path = plot_dir / f"plot_3_robustness_{param}.svg"
        plt.savefig(plot_path)
        plt.close()
        log.info(f"  Sensitivity plot for '{param}' saved to: {plot_path}")

def plot_4_vulnerability_mitigation(df, plot_dir, palette):
    """生成图4：安全包络对 LLM 漏洞的缓解效果 (SVG)"""
    log.info("4. Generating Plot 4: Vulnerability Mitigation (SVG)...")
    trap_df = df[df['is_trap'] & df['agent_type'].isin(['BA', 'OMA_Synth'])].copy()
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.barplot(
        data=trap_df, x='trap_vector', y='is_violation', hue='agent_model',
        palette={k: v for k, v in palette.items() if 'BA' in k or 'OMA_Synth' in k},
        order=['dilemma', 'state_deception', 'privilege', 'imminent_failure'],
        ax=ax, errorbar=('ci', 95), capsize=.05, edgecolor='black', linewidth=1.5
    )
    ax.set_title('Safety Envelope Mitigates LLM Vulnerabilities', pad=25)
    ax.set_xlabel('Trap Vector (Deception Tactic)')
    ax.set_ylabel('Mean Violation Rate (95% CI)')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(title='Agent Model', loc='upper right')
    plot_path = plot_dir / "plot_4_vulnerability_mitigation.svg"
    plt.savefig(plot_path)
    plt.close()
    log.info(f"  Saved to: {plot_path}")

def generate_summary_tables(df, table_dir):
    """生成核心性能指标的摘要表格"""
    log.info("5. Generating Summary Tables...")
    summary = df.groupby('agent_model').agg(
        Safety_Score=('is_violation', lambda x: 1 - x.mean()),
        Throughput=('total_throughput', 'mean'),
        Benign_Reject_Rate=('benign_rejection_rate', 'mean'),
    ).round(4)
    styled_summary = summary.style.background_gradient(
        cmap='RdYlGn', subset=['Safety_Score', 'Throughput']
    ).background_gradient(
        cmap='RdYlGn_r', subset=['Benign_Reject_Rate']
    ).format('{:.4f}').set_caption("Table 1: Core Performance Metrics")
    table_path_csv = table_dir / "table_1_summary_metrics.csv"
    table_path_html = table_dir / "table_1_summary_metrics.html"
    summary.to_csv(table_path_csv, index=False)
    styled_summary.to_html(table_path_html, escape=False)
    log.info(f"  Summary tables saved to: {table_dir}")
    print("\n" + "="*80); print("Table 1: Core Performance Metrics".center(80)); print("="*80)
    print(summary.to_string(index=False))
    print("="*80)

def perform_statistical_tests(df, table_dir):
    """执行关键对比的统计显著性检验 (T-Test)"""
    log.info("6. Performing Statistical Significance Tests (T-Tests)...")
    agent_models = df['agent_model'].unique()
    results = []
    model_variants = [m for m in df['model_name'].unique() if m != 'heuristic']
    comparisons = []
    for model in model_variants:
        comparisons.extend([
            (f"BA_{model}", f"OMA_Synth_{model}"), (f"HA_heuristic", f"OMA_Synth_{model}")
        ])
    for agent1, agent2 in comparisons:
        if agent1 not in agent_models or agent2 not in agent_models: continue
        data1_safety = df[df['agent_model'] == agent1]['is_violation']
        data2_safety = df[df['agent_model'] == agent2]['is_violation']
        if len(data1_safety) > 1 and len(data2_safety) > 1:
            stat, p_val = ttest_ind(data1_safety, data2_safety, equal_var=False, nan_policy='omit')
            results.append({
                'Comparison': f"{agent1} vs {agent2}", 'Metric': 'Violation Rate',
                f'{agent1}_Mean': data1_safety.mean(), f'{agent2}_Mean': data2_safety.mean(),
                'P-Value': p_val, 'Significant (p<0.05)': p_val < 0.05
            })
    if not results: log.warning("  Not enough data for statistical tests."); return
    stats_df = pd.DataFrame(results).round(4)
    stats_path_csv = table_dir / "table_2_statistical_tests.csv"
    stats_path_html = table_dir / "table_2_statistical_tests.html"
    stats_df.to_csv(stats_path_csv, index=False)
    stats_df.style.set_caption("Table 2: Statistical Significance of Key Comparisons").to_html(stats_path_html, escape=False)
    log.info(f"  Statistical test results saved to: {table_dir}")
    print("\n" + "="*80); print("Table 2: Statistical Significance of Key Comparisons".center(80)); print("="*80)
    print(stats_df.to_string(index=False))
    print("="*80)

def main():
    """主执行函数"""
    setup_plot_style()
    palette, markers = get_agent_style_map()
    log.info("--- Starting Final Analysis with Legend Fix (v15.5) ---")
    
    results_dir = Path("results")
    log_dir, table_dir, plot_dir = results_dir / "logs", results_dir / "tables", results_dir / "plots"
    for d in [table_dir, plot_dir]: d.mkdir(exist_ok=True)
    
    try:
        latest_log = max(log_dir.glob("*.csv"), key=os.path.getctime)
        log.info(f"Analyzing log file: {latest_log}")
    except ValueError:
        log.critical(f"!! No CSV log file found in '{log_dir}'. Exiting."); return
    
    df = pd.read_csv(latest_log)
    df['agent_model'] = df['agent_type'] + '_' + df['model_name']
    
    plot_1_for_manual_editing(df, plot_dir, palette, markers)
    plot_2_conservatism_analysis(df, plot_dir, palette)
    plot_3_robustness_to_parameters(df, plot_dir, palette, markers)
    plot_4_vulnerability_mitigation(df, plot_dir, palette)
    
    generate_summary_tables(df, table_dir)
    perform_statistical_tests(df, table_dir)

    log.info("\n--- Analysis Pipeline Finished ---")
    print("\nAll plots have been generated in SVG format.")
    print("The legend issue in robustness plots has been fixed.")
    print("Please check the generated SVG files in the 'results/plots' directory.")

if __name__ == "__main__":
    main()