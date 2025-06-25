import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\muska\Downloads\fake_news_dataset (1).csv")

print("First 5 entries: \n")
print(df.head())

print(df.info())
print("\n")
print("Sum of missing values in each column: \n")
print(df.isnull().sum())

# Bar Chart – News Label Distribution

sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(9,6))
ax = sns.countplot(data=df, x='label', hue='label', palette='pastel', edgecolor='black', legend=False)

df['label'] = df['label'].map({0: 'Fake News', 1: 'Real News'})


# Add value labels on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("News Classification Distribution", fontsize=18, weight='bold')
plt.xlabel("News Category", fontsize=14)
plt.ylabel("Number of Articles", fontsize=14)
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.5)

# Save figure
plt.tight_layout()
plt.savefig("bar_news_distribution.png", dpi=300, bbox_inches='tight')
plt.show()


# Histogram – Article Length Distribution

# Calculate word count per article
df['text_length'] = df['text'].astype(str).apply(lambda x: len(x.split()))

plt.style.use('dark_background')

plt.figure(figsize=(10,6))
sns.histplot(data=df, x='text_length', bins=60, color="#00BFFF", kde=True, edgecolor=None)

# Add mean and median lines
mean_len = df['text_length'].mean()
median_len = df['text_length'].median()
plt.axvline(mean_len, color='orange', linestyle='--', linewidth=2, label=f'Mean: {int(mean_len)}')
plt.axvline(median_len, color='lime', linestyle='--', linewidth=2, label=f'Median: {int(median_len)}')


plt.title("Distribution of Article Lengths (Words)", fontsize=18, weight='bold')
plt.xlabel("Number of Words", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()


plt.savefig("hist_article_lengths.png", dpi=300, bbox_inches='tight')
plt.show()
