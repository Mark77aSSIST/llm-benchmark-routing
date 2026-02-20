# ========================================
# 05. Visualization
# ========================================

import pickle
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Result file path
RESULTS_FILE = '/content/routerbench_results/evaluation_results_FINAL.pkl'  # File path uploaded to Colab
OUTPUT_DIR = '/content'  # Output Directory

print("=" * 80)
print("Figure & Table Generation")
print("=" * 80)

# Set global font settings for Times New Roman, 12pt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

# Data Load
print("\n" + "=" * 80)
print("Data Loading...")
print("=" * 80)

with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

print(f"Data Load Complete")
print(f"  Methods: {list(results.keys())}")

# Figure 1: Performance Comparison Bar Graph
# ========================================
print("\n" + "=" * 80)
print("Figure 1: Performance Comparison")
print("=" * 80)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

methods = ['Best Single', 'Uniform', 'ABW', 'SPW', 'Hybrid', 'Top-K']
accuracies = [
    results['best_single']['accuracy'] * 100,
    results['uniform']['accuracy'] * 100,
    results['abw']['accuracy'] * 100,
    results['spw']['accuracy'] * 100,
    results['hybrid']['accuracy'] * 100,
    results['topk']['accuracy'] * 100
]

oracle_acc = results['oracle']['accuracy'] * 100

colors = ['#d62728', '#ff7f0e', '#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd']

bars = ax.bar(methods, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

# Oracle Line
ax.axhline(y=oracle_acc, color='red', linestyle='--', linewidth=2.5,
           label=f'Oracle ({oracle_acc:.1f}%)', alpha=0.7)

# Print Values
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold', fontname='Times New Roman')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', fontname='Times New Roman')
ax.set_xlabel('Method', fontsize=12, fontweight='bold', fontname='Times New Roman')
ax.set_title('Figure 1: Performance Comparison', fontsize=12, fontweight='bold', pad=20, fontname='Times New Roman')
ax.legend(loc='lower right', prop={'size': 12, 'family': 'Times New Roman'})
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 110)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure1_performance.png', dpi=300, bbox_inches='tight')
print("Figure 1 Save: figure1_performance.png")
plt.close()

# Figure 2: AIQ Comparison Bar Graph
# ========================================
print("\n" + "=" * 80)
print("Figure 2: AIQ Comparison")
print("=" * 80)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

methods_aiq = ['Uniform', 'ABW', 'SPW', 'Hybrid', 'Top-K']
aiqs = []

baseline = results['best_single']['accuracy']
oracle = results['oracle']['accuracy']

for method in ['uniform', 'abw', 'spw', 'hybrid', 'topk']:
    acc = results[method]['accuracy']
    aiq = (acc - baseline) / (oracle - baseline) * 100
    aiqs.append(aiq)

colors_aiq = ['#ff7f0e', '#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd']

bars = ax.bar(methods_aiq, aiqs, color=colors_aiq, alpha=0.85, edgecolor='black', linewidth=1.5)

# Oracle Line
ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Oracle (100%)', alpha=0.7)

# Print Values
for bar, aiq in zip(bars, aiqs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{aiq:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold', fontname='Times New Roman')

ax.set_ylabel('AIQ (%)', fontsize=12, fontweight='bold', fontname='Times New Roman')
ax.set_xlabel('Method', fontsize=12, fontweight='bold', fontname='Times New Roman')
ax.set_title('Figure 2: AIQ Comparison of Ensemble methods', fontsize=12, fontweight='bold', pad=20, fontname='Times New Roman')
ax.legend(loc='upper right', prop={'size': 12, 'family': 'Times New Roman'})
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 110)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure2_aiq.png', dpi=300, bbox_inches='tight')
print("Figure 2 Save: figure2_aiq.png")
plt.close()

# Figure 3: Cost-Performance Trade-off
# ========================================
print("\n" + "=" * 80)
print("Figure 3: Cost-Performance Trade-off")
print("=" * 80)

fig, ax = plt.subplots(1, 1, figsize=(12, 9))

methods_scatter = ['best_single', 'uniform', 'abw', 'spw', 'hybrid', 'topk']
method_labels = ['Best Single', 'Uniform', 'ABW', 'SPW', 'Hybrid', 'Top-K']
colors_scatter = ['#d62728', '#ff7f0e', '#2ca02c', '#bcbd22', '#1f77b4', '#9467bd']
sizes = [250, 250, 350, 250, 350, 400]

# Point
for method, label, color, size in zip(methods_scatter, method_labels, colors_scatter, sizes):
    acc = results[method]['accuracy'] * 100
    calls = results[method]['avg_calls']

    ax.scatter(calls, acc, s=size, alpha=0.7, c=[color], edgecolors='black', linewidth=2.5, zorder=5)

# Label Position
label_configs = {
    'best_single': {
        'xytext': (0.5, 3),
        'ha': 'left',
        'arrowprops': dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='black', lw=1.5, alpha=0.6)
    },
    'uniform': {
        'xytext': (-2.5, 6),
        'ha': 'right',
        'arrowprops': dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', lw=1.5, alpha=0.6)
    },
    'abw': {
        'xytext': (0.5, 6),
        'ha': 'left',
        'arrowprops': dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', lw=1.5, alpha=0.6)
    },
    'spw': {
        'xytext': (-2.5, -5),
        'ha': 'right',
        'arrowprops': dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='black', lw=1.5, alpha=0.6)
    },
    'hybrid': {
        'xytext': (0.5, -5),
        'ha': 'left',
        'arrowprops': dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='black', lw=1.5, alpha=0.6)
    },
    'topk': {
        'xytext': (-1.5, -4),
        'ha': 'right',
        'arrowprops': dict(arrowstyle='->', connectionstyle='arc3,rad=-0.1', color='black', lw=1.5, alpha=0.6)
    }
}

for method, label, color in zip(methods_scatter, method_labels, colors_scatter):
    acc = results[method]['accuracy'] * 100
    calls = results[method]['avg_calls']

    config = label_configs[method]

    ax.annotate(label,
                xy=(calls, acc),
                xytext=config['xytext'],
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                fontname='Times New Roman',
                ha=config['ha'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3, edgecolor='black', linewidth=1.5),
                arrowprops=config['arrowprops'])

# Oracle Line
oracle_acc = results['oracle']['accuracy'] * 100
ax.axhline(y=oracle_acc, color='red', linestyle='--', linewidth=2.5, alpha=0.6, zorder=3)

# Oracle Label
ax.text(11.5, oracle_acc + 1, 'Oracle (100%)', fontsize=12, fontweight='bold', fontname='Times New Roman',
        color='red', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'))

ax.set_xlabel('Average Model Calls', fontsize=12, fontweight='bold', fontname='Times New Roman')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', fontname='Times New Roman')
ax.set_title('Figure 3: Analysis of cost versus accuracy (average number of model calls) for each method', fontsize=12, fontweight='bold', pad=20, fontname='Times New Roman')
ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(40, 105)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure3_cost_performance.png', dpi=300, bbox_inches='tight')
print("Figure 3 Save: figure3_cost_performance.png")
plt.close()

# Figure 4: Top-K Distribution Circle Graph
# ========================================
print("\n" + "=" * 80)
print("Figure 4: Top-K Distribution")
print("=" * 80)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

k_dist = results['topk']['k_distribution']
k_counts = [k_dist.get(1, 0), k_dist.get(2, 0), k_dist.get(3, 0)]
total_samples = sum(k_counts)

wedges, texts, autotexts = ax.pie([100],
                                    labels=[f'K=1\n({k_counts[0]:,} samples)'],
                                    colors=['#2ca02c'],
                                    autopct='%1.0f%%',
                                    startangle=90,
                                    explode=[0],
                                    shadow=True,
                                    textprops={'fontsize': 12, 'fontweight': 'bold', 'fontname': 'Times New Roman'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
    autotext.set_fontname('Times New Roman')

ax.set_title('Figure 4: Top-K Distribution\n(All samples use K=1)',
             fontsize=12, fontweight='bold', pad=20, fontname='Times New Roman')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure4_k_distribution.png', dpi=300, bbox_inches='tight')
print("Figure 4 Save: figure4_k_distribution.png")
plt.close()

# Table 1: Performance Comparison
# ========================================
print("\n" + "=" * 80)
print("Table 1: Performance Comparison")
print("=" * 80)

table1_data = []

methods_order = ['oracle', 'best_single', 'uniform', 'abw', 'spw', 'hybrid', 'topk']
method_names = {
    'oracle': 'Oracle',
    'best_single': 'Best Single',
    'uniform': 'Uniform Ensemble',
    'abw': 'ABW',
    'spw': 'SPW',
    'hybrid': 'Hybrid',
    'topk': 'Top-K'
}

for method in methods_order:
    result = results[method]
    acc = result.get('accuracy', 0)
    calls = result.get('avg_calls', 0)

    # K distribution
    k_dist = result.get('k_distribution', {})
    if k_dist and sum(k_dist.values()) > 0:
        k_str = f"K1:{k_dist.get(1,0)} K2:{k_dist.get(2,0)} K3:{k_dist.get(3,0)}"
    else:
        k_str = "-"

    table1_data.append({
        'Method': method_names[method],
        'Accuracy': acc,
        'Avg Calls': calls,
        'K Distribution': k_str
    })

df_table1 = pd.DataFrame(table1_data)

print(f"\n{'Method':<20s} {'Accuracy':>10s} {'Avg Calls':>12s} {'K Distribution':>30s}")
print("-" * 80)

for _, row in df_table1.iterrows():
    print(f"{row['Method']:<20s} {row['Accuracy']:>9.2%} {row['Avg Calls']:>11.2f} {row['K Distribution']:>30s}")

df_table1.to_csv(f'{OUTPUT_DIR}/table1_performance.csv', index=False)
print("\nTable 1 Save: table1_performance.csv")

# Table 2: AIQ & Improvements
# ========================================
print("\n" + "=" * 80)
print("Table 2: Accuracy IQ and Improvements")
print("=" * 80)

oracle_acc = results['oracle']['accuracy']
baseline_acc = results['best_single']['accuracy']

table2_data = []

for method in ['uniform', 'abw', 'spw', 'hybrid', 'topk']:
    acc = results[method]['accuracy']
    improvement = (acc - baseline_acc) * 100

    if oracle_acc > baseline_acc:
        aiq = (acc - baseline_acc) / (oracle_acc - baseline_acc) * 100
    else:
        aiq = 0.0

    table2_data.append({
        'Method': method_names[method],
        'Accuracy': acc,
        'Improvement (pp)': improvement,
        'AIQ (%)': aiq
    })

df_table2 = pd.DataFrame(table2_data)

print(f"\n{'Method':<20s} {'Accuracy':>10s} {'Improvement':>15s} {'AIQ':>10s}")
print("-" * 60)

for _, row in df_table2.iterrows():
    print(f"{row['Method']:<20s} {row['Accuracy']:>9.2%} {row['Improvement (pp)']:>+14.2f}p {row['AIQ (%)']:>9.1f}%")

df_table2.to_csv(f'{OUTPUT_DIR}/table2_aiq.csv', index=False)
print("\nTable 2 Save: table2_aiq.csv")

# Table 3: Statistics Significance
# ========================================
print("\n" + "=" * 80)
print("Table 3: Statistical Significance")
print("=" * 80)

table3_data = []

comparisons = [
    ('ABW', 'Uniform'),
    ('SPW', 'Uniform'),
    ('Hybrid', 'Uniform'),
    ('Top-K', 'Hybrid'),
]

method_map = {
    'ABW': 'abw',
    'SPW': 'spw',
    'Uniform': 'uniform',
    'Hybrid': 'hybrid',
    'Top-K': 'topk'
}

print(f"\n{'Comparison':<25s} {'Acc1':>10s} {'Acc2':>10s} {'t-stat':>10s} {'p-value':>12s} {'Sig':>6s}")
print("-" * 75)

for method1, method2 in comparisons:
    m1_key = method_map[method1]
    m2_key = method_map[method2]

    preds1 = np.array(results[m1_key]['predictions'])
    preds2 = np.array(results[m2_key]['predictions'])

    acc1 = results[m1_key]['accuracy']
    acc2 = results[m2_key]['accuracy']

    if not np.array_equal(preds1, preds2):
        t_stat, p_value = stats.ttest_rel(preds1, preds2)
        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
    else:
        t_stat = 0.0
        p_value = 1.0
        sig = "ns"

    comparison_str = f"{method1} vs {method2}"

    print(f"{comparison_str:<25s} {acc1:>9.2%} {acc2:>9.2%} {t_stat:>10.4f} {p_value:>12.6f} {sig:>6s}")

    table3_data.append({
        'Comparison': comparison_str,
        'Method 1': method1,
        'Accuracy 1': acc1,
        'Method 2': method2,
        'Accuracy 2': acc2,
        't-statistic': t_stat,
        'p-value': p_value,
        'Significance': sig
    })

df_table3 = pd.DataFrame(table3_data)
df_table3.to_csv(f'{OUTPUT_DIR}/table3_significance.csv', index=False)

print("\nTable 3 Save: table3_significance.csv")
print("\n주: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant")

# Summary Statistics
# ========================================

print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)

print(f"\nBaseline:")
print(f"  Oracle:      {oracle_acc:.4f} ({oracle_acc*100:.2f}%)")
print(f"  Best Single: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")

print(f"\nKey Results:")
print(f"  ABW:         {results['abw']['accuracy']:.4f} ({results['abw']['accuracy']*100:.2f}%)")
print(f"  SPW:         {results['spw']['accuracy']:.4f} ({results['spw']['accuracy']*100:.2f}%)")
print(f"  Hybrid:      {results['hybrid']['accuracy']:.4f} ({results['hybrid']['accuracy']*100:.2f}%)")
print(f"  Top-K:       {results['topk']['accuracy']:.4f} ({results['topk']['accuracy']*100:.2f}%)")

print(f"\nTop-K Details:")
k_dist = results['topk']['k_distribution']
total = sum(k_dist.values())
print(f"  K=1: {k_dist.get(1, 0):5d} ({k_dist.get(1, 0)/total*100:5.1f}%)")
print(f"  K=2: {k_dist.get(2, 0):5d} ({k_dist.get(2, 0)/total*100:5.1f}%)")
print(f"  K=3: {k_dist.get(3, 0):5d} ({k_dist.get(3, 0)/total*100:5.1f}%)")
print(f"  Average Calls: {results['topk']['avg_calls']:.2f}")
print(f"  Reduce costs: {(1 - results['topk']['avg_calls']/11)*100:.1f}%")

print("\n" + "=" * 80)
print("All of Figure & Table Generation Completed")
print("=" * 80)

print("\nGenerated and Saved Files:")
print("Figure 1: figure1_performance.png")
print("Figure 2: figure2_aiq.png")
print("Figure 3: figure3_cost_performance.png")
print("Figure 4: figure4_k_distribution.png")
print("Table 1: table1_performance.csv")
print("Table 2: table2_aiq.csv")
print("Table 3: table3_significance.csv")

print("\n" + "=" * 80)
print("Visualization Complete")
print("=" * 80)