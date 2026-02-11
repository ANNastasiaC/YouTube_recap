# %%
# Set up
HTML_FILE =    # path to your exported YouTube history, which you could download at https://takeout.google.com
API_KEY =      # YouTube Data API key, which should be free
SAMPLE_SIZE = 200                      # number of videos to analyze

# %%
# =========================
# 1. Imports
# =========================
import re
import math
import requests
import pandas as pd
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# =========================
# 2. Load and Parse HTML
# =========================

with open(HTML_FILE, 'r', encoding='utf-8') as file:
    html_content = file.read()

video_data = []

video_links = re.findall(
    r'<a href="(https://www.youtube.com/watch\?v=.+?)">(.+?)</a>',
    html_content
)

for link, title in video_links:
    video_id = link.split('=')[-1][:11]

    # Extract watch time using regex directly from HTML block
    watch_time = None

    timestamp_pattern = r'[A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}\u202f?[AP]M [A-Z]+'

    search_start = html_content.find(link)
    search_region = html_content[search_start: search_start + 300]

    match = re.search(timestamp_pattern, search_region)

    if match:
        watch_time_raw = match.group(0)

        # Normalize weird spaces
        watch_time_clean = watch_time_raw.replace('\u202f', ' ')

        try:
            watch_time = datetime.strptime(
                watch_time_clean,
                '%b %d, %Y, %I:%M:%S %p %Z'
            )
        except:
            watch_time = None

    video_data.append({
        "video_id": video_id,
        "title": title.replace('-', ' '),
        "watch_time": watch_time
    })

df = pd.DataFrame(video_data)

# =========================
# 3. Systematic Sampling
# =========================


sampling_interval = math.ceil(len(df) / SAMPLE_SIZE)
df_sampled = df.iloc[::sampling_interval].head(SAMPLE_SIZE).copy()


# %%

# =========================
# 4. Fetch Categories + Descriptions
# =========================

def fetch_video_metadata(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={API_KEY}&part=snippet"
    response = requests.get(url).json()

    if "items" in response and response["items"]:
        snippet = response["items"][0]["snippet"]
        return snippet.get("categoryId"), snippet.get("description")
    return None, None


def fetch_category_name(category_id):
    url = f"https://www.googleapis.com/youtube/v3/videoCategories?id={category_id}&key={API_KEY}&part=snippet"
    response = requests.get(url).json()

    if "items" in response and response["items"]:
        return response["items"][0]["snippet"]["title"]
    return None


category_cache = {}

categories = []
descriptions = []

for vid in df_sampled["video_id"]:
    cat_id, desc = fetch_video_metadata(vid)

    if cat_id:
        if cat_id not in category_cache:
            category_cache[cat_id] = fetch_category_name(cat_id)
        categories.append(category_cache[cat_id])
    else:
        categories.append(None)

    descriptions.append(desc)

df_sampled["category"] = categories
df_sampled["description"] = descriptions

# =========================
# 5. Export to CSV
# =========================

OUTPUT_FILE = "youtube_sample_analysis.csv"
df_sampled.to_csv(OUTPUT_FILE, index=False)

print(f"CSV saved as {OUTPUT_FILE}")


# %%

# =========================
# 6. Plots you could try
# =========================

stop_words = set(stopwords.words('english'))

all_text = ' '.join(df_sampled["title"].dropna())
words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())

filtered_words = [w for w in words if w not in stop_words]
word_counts = Counter(filtered_words)
top_words = word_counts.most_common(50)

plt.figure(figsize=(12, 8))
plt.bar([w for w, _ in top_words], [c for _, c in top_words])
plt.xticks(rotation=90)
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.title("Most Frequent Words in Sampled Video Titles")
plt.tight_layout()
plt.show()


# %%
df_sampled["month"] = df_sampled["watch_time"].dt.to_period("M")

category_month = (
    df_sampled.groupby(["month", "category"])
      .size()
      .unstack(fill_value=0)
)

category_month.plot(kind="area", figsize=(12,6))
plt.ylabel("Watch Count")
plt.title("Category Distribution Over Time")
plt.show()

# %%
import numpy as np

def entropy(series):
    counts = series.value_counts(normalize=True)
    return -np.sum(counts * np.log(counts))

monthly_entropy = (
    df_sampled.groupby(df_sampled["watch_time"].dt.to_period("M"))["category"]
      .apply(entropy)
)

monthly_entropy.plot(figsize=(10,5))
plt.title("Category Diversity Over Time")
plt.show()

# %%
df_embed = df_sampled.dropna(subset=["description", "watch_time"]).copy()
df_embed = df_embed.sort_values("watch_time")

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

descriptions = df_embed["description"].tolist()
embeddings = model.encode(descriptions, show_progress_bar=True)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

import umap
import matplotlib.cm as cm
import numpy as np

reducer = umap.UMAP(n_components=2, random_state=42)
coords = reducer.fit_transform(embeddings)

df_embed["x"] = coords[:, 0]
df_embed["y"] = coords[:, 1]


time_numeric = (
    df_embed["watch_time"] - df_embed["watch_time"].min()
).dt.total_seconds()

norm = time_numeric / time_numeric.max()

plt.figure(figsize=(8,8))
plt.scatter(df_embed["x"], df_embed["y"], c=norm, cmap="viridis", alpha=0.8)

plt.colorbar(label="Time Progression")
plt.title("Semantic Trajectory Over Time")
plt.show()



