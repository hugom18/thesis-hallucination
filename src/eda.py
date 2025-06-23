import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_ind

# Load enriched data
with open("data/outputs_enriched.json", "r") as f:
    data = json.load(f)

# Build DataFrame and filter A & F
df = pd.DataFrame(data)
df_af = df[df['category'].isin(['A', 'F'])]

# Scatter plot
plt.scatter(df_af['bert_f1'], df_af['cos_sim'],
            c=df_af['hallucination'], cmap='bwr', alpha=0.7)
plt.xlabel('BERTScore F1')
plt.ylabel('SBERT Cosine Similarity')
plt.title('Similarity Metrics by Hallucination (A vs. F)')
plt.show()

# Boxplots (use tick_labels to avoid 3.11 warning)
plt.boxplot([df_af[df_af['hallucination']==0]['bert_f1'],
             df_af[df_af['hallucination']==1]['bert_f1']],
            tick_labels=['A (non-hall.)','F (hall.)'])
plt.title('BERTScore F1 Distribution')
plt.show()

plt.boxplot([df_af[df_af['hallucination']==0]['cos_sim'],
             df_af[df_af['hallucination']==1]['cos_sim']],
            tick_labels=['A (non-hall.)','F (hall.)'])
plt.title('Cosine Similarity Distribution')
plt.show()

# Summary stats
stats = df_af.groupby('hallucination')[['bert_f1','cos_sim']].agg(['mean','std'])
print("Summary statistics (mean ± std):")
print(stats)

# Statistical significance tests
f1_A = df_af[df_af.hallucination==0]['bert_f1']
f1_F = df_af[df_af.hallucination==1]['bert_f1']
t_f1, p_f1 = ttest_ind(f1_A, f1_F)
cos_A = df_af[df_af.hallucination==0]['cos_sim']
cos_F = df_af[df_af.hallucination==1]['cos_sim']
t_cos, p_cos = ttest_ind(cos_A, cos_F)
print(f"\nBERTScore F1 t-test: t={t_f1:.2f}, p={p_f1:.3f}")
print(f"Cosine Sim t-test:   t={t_cos:.2f}, p={p_cos:.3f}")

# Threshold‐based baseline
print("\nSingle‐metric baselines:")
for t in [0.85, 0.90]:
    preds = (df_af['bert_f1'] < t).astype(int)
    print(f" F1 < {t}:  Acc = {accuracy_score(df_af.hallucination, preds):.2f}, "
          f"F1 = {f1_score(df_af.hallucination, preds):.2f}")

# Combined threshold baseline
preds_comb = ((df_af['bert_f1'] < 0.90) | (df_af['cos_sim'] < 0.75)).astype(int)
acc_c = accuracy_score(df_af.hallucination, preds_comb)
f1_c  = f1_score(df_af.hallucination, preds_comb)
print(f"\nCombined rule (F1<0.90 or cos<0.75): Acc = {acc_c:.2f}, F1 = {f1_c:.2f}")
