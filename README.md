# News Headline Clustering with Sentence Embeddings

This project performs **semantic clustering and visualization** of economic news headlines using **Sentence Transformers**, **t-SNE**, and **K-Means**. It extracts the most frequent words in each cluster and visualizes them with **Plotly** for an intuitive understanding of topic groupings.

---

## Objective

- Embed news headlines using [SentenceTransformers](https://www.sbert.net/).
- Reduce dimensionality with **t-SNE**.
- Find optimal number of clusters using the elbow method.
- Cluster headlines using **K-Means**.
- Extract top 5 frequent words per cluster.
- Visualize everything using **Plotly**.

---

## Dataset

The dataset used is:
```
US-Economic-News.csv
```
It must contain a `headline` column with economic news headlines (strings).

---

## Installation

1. Clone the repository and navigate into the folder:

```bash
git clone https://github.com/your-username/news-headline-clustering.git
cd news-headline-clustering
```

2. Install dependencies:

```bash
pip install pandas numpy scikit-learn plotly nltk sentence-transformers
```

3. Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## How to Run

Make sure your dataset file is named `US-Economic-News.csv` and contains a `headline` column.

Then simply run:

```bash
python your_script.py
```

The script will:
- Embed the headlines using `"all-MiniLM-L6-v2"`
- Apply t-SNE for dimensionality reduction
- Automatically determine the best number of clusters
- Run K-Means clustering
- Visualize the clusters in an interactive Plotly chart

---

## Output

- Each point in the plot represents a news headline.
- Clusters are color-coded and labeled using the most frequent words in that cluster.
- Hovering over a point shows the full headline.
- Cluster centers are annotated with top keywords.

---

## Libraries Used

- [SentenceTransformers](https://github.com/UKPLab/sentence-transformers)
- [scikit-learn](https://scikit-learn.org/)
- [Plotly](https://plotly.com/python/)
- [NLTK](https://www.nltk.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

---

## Notes

- The `perplexity` parameter for t-SNE is set to 30, which works well for medium-sized datasets.
- The optimal number of clusters is automatically selected using the elbow method (max ΔSSE).
- Cluster keywords are extracted after removing common English stopwords.

---