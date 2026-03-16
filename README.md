# Link Prediction for Recommender Systems (RecSys)

This project explores the concept of **Link Prediction** as a foundation for building robust recommendation engines. Using the classic **MovieLens 100k** dataset, we model user-item interactions as a **Bipartite Graph** and predict missing edges to generate personalized recommendations.

## 🚀 Project Overview

In traditional RecSys, we often think of a matrix of ratings. Here, we pivot to **Network Science**:
- **Nodes:** Users and Movies.
- **Edges:** A link exists if a user "liked" a movie (Rating $\ge$ 4).
- **Goal:** Predict which movie nodes a user node is most likely to connect with in the future.

## 🛠️ Methodology

The project is structured into a progressive pipeline:

1.  **Exploratory Data Analysis (EDA):** Analyzing sparsity, degree distribution, and popularity bias.
2.  **Bipartite Graph Construction:** Building a `NetworkX` graph with typed nodes (`u_` for users, `m_` for movies).
3.  **Heuristic Baseline:** Implementing structural similarity metrics like **Jaccard Coefficient** and **Common Neighbors**.
4.  **Hybrid Approach:** Combining **Graph Topology** with **Content Features** (User Demographics & Movie Genres) using a weighted similarity score.
5.  **Graph Representation Learning (Node2Vec):** Transitioning to learned 64D embeddings via uniform random walks and Word2Vec.
6.  **Specialized Heterogeneous Embeddings (Metapath2Vec):** Optimizing random walks for the bipartite structure using **User-Movie-User (UMU)** metapaths to capture deep collaborative filtering intent.
7.  **Unified Evaluation Benchmark:** A head-to-head comparison of all models using **Precision@10**, **Recall@10**, and **MRR**.

## 📊 Key Findings

From our **Unified Benchmark**, we observed:
- **Jaccard Heuristic** is a surprisingly strong performer on this dataset, outperforming basic random-walk embeddings.
- **Hybrid Model** provides robust "Cold Start" recommendations by incorporating user demographics.
- **Embedding Models** (Node2Vec/Metapath2Vec) currently struggle with graph sparsity, providing the technical motivation to move towards **Graph Neural Networks (GNNs)**.

## 📂 Project Structure

- `eda.ipynb`: Initial data exploration and visualization.
- `graph_construction.ipynb`: Converting tabular data into a `NetworkX` bipartite graph.
- `heuristic_recommender.ipynb`: Building the baseline Jaccard-based recommender.
- `feature_engineering.ipynb`: Engineering user/movie vectors and implementing the Hybrid model.
- `node2vec_recommender.ipynb`: Implementing latent representation learning via random walks.
- `metapath2vec_recommender.ipynb`: Specialized bipartite walks for superior collaborative embeddings.
- `unified_benchmark.ipynb`: Comprehensive evaluation and performance analysis across all models.
- `data/`: (Ignored in Git) Raw MovieLens 100k dataset.

## ⚙️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AbdellatifSatir/Link-Prediction-RecSys.git
    cd Link-Prediction-RecSys
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy networkx matplotlib seaborn scikit-learn gensim
    ```

3.  **Download Dataset:**
    Run the provided utility script:
    ```bash
    python download_data.py
    ```

## 📈 Future Roadmap

- [x] **Node2Vec Embeddings:** Learning latent representations via random walks.
- [x] **Metapath2Vec:** Specialized walks for bipartite structures.
- [x] **Unified Benchmark:** Systematic evaluation comparing all methods.
- [ ] **Graph Neural Networks (GNNs):** Implementing **LightGCN** for state-of-the-art link prediction.

## 📄 License
This project uses the MovieLens 100k dataset. Please refer to the [GroupLens website](https://grouplens.org/datasets/movielens/100k/) for licensing details.
