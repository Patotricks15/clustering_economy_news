from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
import numpy as np
from collections import Counter
import plotly.graph_objects as go
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

embedder = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv("US-Economic-News.csv", encoding='unicode_escape')

corpus_embeddings = embedder.encode(df["headline"].tolist())

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(corpus_embeddings)

df['x'] = embeddings_2d[:, 0]
df['y'] = embeddings_2d[:, 1]

sse = []
K_range = range(1, 11)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(embeddings_2d)
    sse.append(kmeans_temp.inertia_)

differences = [sse[i] - sse[i+1] for i in range(len(sse) - 1)]
best_k = differences.index(max(differences)) + 2

print(f"Best k for k-means: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(embeddings_2d)


stop_words = set(stopwords.words('english'))
cluster_words = {}
for cluster in range(best_k):
    cluster_sentences = df[df['Cluster'] == cluster]['headline'].tolist()
    words = [word.lower() for sentence in cluster_sentences for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]
    common_words = [word for word, _ in Counter(words).most_common(5)]
    cluster_words[cluster] = " ".join(common_words)

cluster_centers = []
for cluster in range(best_k):
    x_mean = np.mean(df[df['Cluster'] == cluster]['x'])
    y_mean = np.mean(df[df['Cluster'] == cluster]['y'])
    cluster_centers.append((x_mean, y_mean))

df['Cluster_Label'] = df['Cluster'].map(cluster_words)

fig = px.scatter(
    df,
    x='x',
    y='y',
    color='Cluster_Label',  
    hover_data=['headline'],
    title="t-SNE + K-Means Clustering (Economy news)",
    opacity=0.7
)

for cluster, (x_center, y_center) in enumerate(cluster_centers):
    fig.add_trace(
        go.Scatter(
            x=[x_center],
            y=[y_center],
            text=[cluster_words[cluster]],
            mode='text',                     
            textfont=dict(size=20, color='black'),
            showlegend=False                
        )
    )

fig.show()