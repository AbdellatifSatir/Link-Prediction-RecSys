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
4.  **Hybrid Approach:** Combining **Graph Topology** with **Content Features** (User Demographics & Movie Genres).
5.  **Graph Representation Learning (Node2Vec):** Transitioning to learned 64D embeddings via uniform random walks.
6.  **Heterogeneous Embeddings (Metapath2Vec):** Optimizing random walks for the bipartite structure using **User-Movie-User (UMU)** metapaths.
7.  **Graph Neural Networks (LightGCN):** Implementing state-of-the-art message passing to explicitly propagate the collaborative filtering signal.
8.  **Unified Evaluation Benchmark:** A head-to-head comparison of all models using **Precision@10**, **Recall@10**, and **MRR**.

## 📊 Key Findings

From our **Unified Benchmark (The Grand Finale)**, we observed:
- **Winner:** **LightGCN** is the superior model for discovery, achieving the highest **Recall@10 (0.179)** while matching the Jaccard baseline in precision.
- **Structural Power:** **Jaccard Similarity** remains a surprisingly strong baseline (Precision@10: 0.146), outperforming shallow embeddings.
- **Content vs. Behavior:** Actual viewing behavior (graph edges) is a much stronger predictor of interest than user demographics (Age/Gender/Occupation).
- **GNN Advantage:** Multi-hop linear message passing in LightGCN captures more complex latent relationships than random-walk based methods like Node2Vec.

## 📂 Project Structure

- `eda.ipynb`: Initial data exploration and visualization.
- `graph_construction.ipynb`: Converting tabular data into a `NetworkX` bipartite graph.
- `heuristic_recommender.ipynb`: Building the baseline Jaccard-based recommender.
- `feature_engineering.ipynb`: Engineering user/movie vectors and implementing the Hybrid model.
- `node2vec_recommender.ipynb`: Implementing latent representation learning via random walks.
- `metapath2vec_recommender.ipynb`: Specialized bipartite walks for superior collaborative embeddings.
- `lightgcn_recommender.ipynb`: GNN implementation using PyTorch Geometric and BPR loss.
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
    pip install pandas numpy networkx matplotlib seaborn scikit-learn gensim torch torch-geometric
    ```

3.  **Download Dataset:**
    Run the provided utility script:
    ```bash
    python download_data.py
    ```

## 📈 Future Roadmap

- [x] **Node2Vec & Metapath2Vec Embeddings**
- [x] **Unified Benchmark Completion**
- [x] **LightGCN Implementation**
- [ ] **Hyperparameter Fine-Tuning:** Optimizing LR, embedding dims, and walk counts.
- [ ] **Graph Attention (GAT):** weighing neighbor influence for better precision.

## 📄 License
This project uses the MovieLens 100k dataset. Please refer to the [GroupLens website](https://grouplens.org/datasets/movielens/100k/) for licensing details.
