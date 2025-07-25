#eda.py
# ─────────────────────────────────────────────────────────────
# eda.py – exploratory data analysis for LLM responses
# ─────────────────────────────────────────────────────────────

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_ind
from IPython.display import display, Markdown
import numpy as np

# 1. Load enriched data and build DataFrame ──────────────────────────────────────────────────────────────────
with open("data/intermediate/outputs_enriched.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. Responses by Model and Hallucination (bars) - Stacked by Prompt Category ──────────────────────────────────────────────────────────────────

#Identify models and categories
models     = sorted(df['model'].unique())
categories = sorted(df['category'].unique())

# Pivot counts: (model, hallucination, category) → count
counts = df.groupby(['model','hallucination','category']) \
           .size() \
           .unstack(fill_value=0)

# Build arrays: rows=models, cols=categories
data_non = np.array([
    counts.loc[(m, 0), categories] if (m,0) in counts.index else [0]*len(categories)
    for m in models
])
data_hall = np.array([
    counts.loc[(m, 1), categories] if (m,1) in counts.index else [0]*len(categories)
    for m in models
])

# Choose a color for each category
palette = plt.get_cmap('tab10')
colors  = {cat: palette(i) for i, cat in enumerate(categories)}

# Plot grouped, stacked bars with labels
fig, ax = plt.subplots(figsize=(12, 6))
x     = np.arange(len(models))
width = 0.35

# Non-hallucinated
bottom = np.zeros(len(models))
for idx, cat in enumerate(categories):
    bars = ax.bar(
        x - width/2,
        data_non[:, idx],
        width,
        bottom=bottom,
        color=colors[cat]
    )
    # Annotate each segment
    for bar, val in zip(bars, data_non[:, idx]):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_y() + val/2,
                str(val),
                ha='center', va='center',
                color='white', fontsize=8
            )
    bottom += data_non[:, idx]

# Hallucinated
bottom = np.zeros(len(models))
for idx, cat in enumerate(categories):
    bars = ax.bar(
        x + width/2,
        data_hall[:, idx],
        width,
        bottom=bottom,
        color=colors[cat]
    )
    # Annotate each segment
    for bar, val in zip(bars, data_hall[:, idx]):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_y() + val/2,
                str(val),
                ha='center', va='center',
                color='white', fontsize=8
            )
    bottom += data_hall[:, idx]

# Primary X-axis: Model names at group centers
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0, ha='right')
ax.set_xlabel("Model")
ax.set_xlabel("Hallucination Label (0=Non, 1=Yes)")
ax.spines['bottom'].set_position(('outward', 20))
ax.tick_params(axis='x', which='both', length=0)        # no tick marks
ax.spines['bottom'].set_visible(False)  

# Secondary X-axis: Hallucination label below each bar
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())

# positions for the 0/1 ticks
h_ticks, h_labels = [], []
for xi in x:
    h_ticks += [xi - width/2, xi + width/2]
    h_labels += ['0','1']
ax2.set_xticks(h_ticks)
ax2.set_xticklabels(h_labels)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 0))  # move it down

# Legend for categories
handles = [Patch(color=colors[c], label=f"Category {c}") for c in categories]
ax.legend(handles=handles, title="Prompt Category", 
          bbox_to_anchor=(1.02,1), loc='upper left')

ax.set_ylabel("Response Count")
ax.set_title("Responses by Model and Hallucination (bars) - Stacked by Prompt Category")


# Save
fig.savefig("result/figures/model_halluc_category_counts.png", dpi=300, bbox_inches='tight')
print("Saved model hallucination category counts to result/figures/model_halluc_category_counts.png")

# 3. BERTScore F1 vs SBERT Cosine Similarity by Hallucination──────────────────────────────────────────────────────────────────

# 3a) Scatter plot of BERTScore F1 vs SBERT Cosine Similarity for all categories
# Your scatter plot code
plt.figure()
plt.scatter(df['bert_f1'], df['cos_sim'], c=df['hallucination'], cmap='bwr', alpha=0.6)
plt.xlabel('BERTScore F1')
plt.ylabel('SBERT Cosine Similarity')
plt.title('BERTScore F1 vs SBERT Cosine Similarity by Hallucination (All Categories)')
plt.colorbar(label='hallucination label')

# Save to file 
plt.savefig("result/figures/bertf1_vs_cosim_scatter.png", dpi=300, bbox_inches="tight")
print("Saved BERTScore F1 vs SBERT Cosine Similarity scatter plot to result/figures/bertf1_vs_cosim_scatter.png")

# 3b) Scatter plot of BERTScore F1 vs SBERT Cosine Similarity by Category
#Prepare faceted scatter by category
categories = sorted(df['category'].dropna().unique())
n = len(categories)
cols = 3
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=True, sharey=True)

# Plot each facet with discrete red/blue colors
for ax, cat in zip(axes.flatten(), categories):
    sub = df[df['category'] == cat]
    colors = sub['hallucination'].map({0: 'blue', 1: 'red'})
    ax.scatter(
        sub['bert_f1'],
        sub['cos_sim'],
        c=colors,
        alpha=0.6,
        edgecolor='k',
        linewidth=0.2
    )
    ax.set_title(f'Category {cat}')
    ax.set_xlabel('BERTScore F1')
    ax.set_ylabel('SBERT Cosine')

# Hide any unused subplots
for ax in axes.flatten()[n:]:
    ax.set_visible(False)

# Create a single legend for the figure
legend_handles = [
    Patch(color='blue', label='Non-Hallucinated (0)'),
    Patch(color='red',  label='Hallucinated (1)')
]
fig.legend(
    handles=legend_handles,
    loc='upper right',
    title='Label',
    bbox_to_anchor=(1, 1.05)
)

plt.suptitle('BERTScore F1 vs SBERT Cosine Similarity by Prompt Category', y=1.02, fontsize=16)
plt.tight_layout(rect=[0, 0, 0.93, 1])  # leave room on right for legend
  
# Save
fig.savefig("/Users/hugomoreno/thesis-hallucination/result/figures/bertf1_vs_cosim_by_category.png", dpi=300, bbox_inches='tight')
print("Saved BERTScore F1 vs SBERT Cosine Similarity by Category to result/figures/bertf1_vs_cosim_by_category.png")

# 4. Prompt‐Level Features by Hallucination Label ──────────────────────────────────────────────────────────────────

# Define the prompt‐level features you want to summarize
prompt_features = [
    'token_len', 'char_len', 'entity_count',
    'pronoun_ratio', 'imperative_flag',
    'readability_score', 'punct_density'
]

# Group by hallucination and compute mean, std, count
grouped = (
    df
    .groupby('hallucination')[prompt_features]
    .agg(['mean','std'])
    .round(3)
)

# Flatten the MultiIndex columns
grouped.columns = ['_'.join(col) for col in grouped.columns]
grouped = (
    grouped
    .reset_index()
    .rename(columns={'hallucination':'Hallucination'})
)



# Save to CSV 
os.makedirs("result/tables", exist_ok=True)
grouped.to_csv("result/tables/prompt_features_summary.csv", index=False)
print("Saved split summary to result/tables/prompt_features_by_hallucination.csv")

# 5. Summary Statistics for Core Metrics by Hallucination Label ──────────────────────────────────────────────────────────────────

# Calculate summary statistics for core metrics
stats = df.groupby('hallucination')[['bert_f1', 'cos_sim']].agg(['mean', 'std', 'count']).round(3)

# Reformat MultiIndex columns
stats.columns = ['_'.join(col) for col in stats.columns]
stats = stats.reset_index().rename(columns={'hallucination': 'Hallucination'})

# Save CSV
csv_path = "result/tables/summary_stats_core_metrics.csv"
stats.to_csv(csv_path, index=False)
print(f"Saved core metrics summary to result/tables/summary_stats_core_metrics.csv")

# 6. Features to Tests ──────────────────────────────────────────────────────────────────

features_to_test = ['bert_f1', 'cos_sim']

# Calculate t-test results
results = []
for feat in features_to_test:
    grp0 = df[df['hallucination'] == 0][feat]
    grp1 = df[df['hallucination'] == 1][feat]
    if len(grp0) > 1 and len(grp1) > 1:
        t_stat, p_val = ttest_ind(grp0, grp1, equal_var=False)
        results.append({'Feature': feat, 't_stat': round(t_stat, 2), 'p_value': round(p_val, 3)})
    else:
        results.append({'Feature': feat, 't_stat': None, 'p_value': None})

# Build DataFrame
tt_df = pd.DataFrame(results)

# Save as CSV
csv_path = "result/tables/core_metrics_t_tests.csv"
tt_df.to_csv(csv_path, index=False)
print(f"Saved t-test results to result/tables/core_metrics_t_tests.csv")

# 7. Question Type Distribution by Hallucination ──────────────────────────────────────────────────────────────────

# Crosstab: rows = question_type, columns = hallucination flag
qt_ct = pd.crosstab(df['question_type'], df['hallucination'])

# Plot a grouped bar chart
fig, ax = plt.subplots(figsize=(8, 5))
qt_ct.plot(
    kind='bar',
    ax=ax,
    width=0.8,    # narrower bars so groups don’t touch
    rot=45        # rotate x‐labels if they’re long
)
ax.set_xlabel('Question Type')
ax.set_ylabel('Count')
ax.set_title('Question Type Distribution by Hallucination')
ax.legend(title='Hallucination', labels=['No', 'Yes'])
plt.tight_layout()

# Save to file
fig.savefig(
    "result/figures/question_type_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
print("Saved question type distribution plot to result/figures/question_type_distribution.png")

# 8. Baseline threshold evaluations ──────────────────────────────────────────────────────────────────

print("\nBaseline evaluations:")
for t in [0.85, 0.90]:
    preds = (df['bert_f1'] < t).astype(int)
    acc = accuracy_score(df['hallucination'], preds)
    f1 = f1_score(df['hallucination'], preds)
    print(f"BERT F1 < {t}: Acc = {acc:.2f}, F1 = {f1:.2f}")

# Combined threshold rule
preds_comb = ((df['bert_f1'] < 0.90) | (df['cos_sim'] < 0.75)).astype(int)
acc_c = accuracy_score(df['hallucination'], preds_comb)
f1_c = f1_score(df['hallucination'], preds_comb)
print(f"\nCombined rule (F1<0.90 or cos<0.75): Acc = {acc_c:.2f}, F1 = {f1_c:.2f}")