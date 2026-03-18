# Project: Link Prediction Recommender System (RecSys)

## 🎯 Project Overview
Building a movie recommendation engine using **Link Prediction** concepts. We treat the user-item interaction as a **Bipartite Graph** where recommendations are predicted missing edges.

---

## 🛠️ Current State (Completed)

### Phase 1: Data Preparation & EDA
- **Dataset:** MovieLens 100k.
- **EDA:** Analyzed rating distributions, sparsity (~93.7%), and popularity bias.
- **Decision:** Ratings $\ge 4$ are defined as "Positive Links" (Ground Truth edges).

### Phase 2: Graph Construction
- **Library:** `NetworkX`.
- **Structure:** Bipartite graph ($U, V$) with unique prefixes (`u_` for users, `m_` for movies).
- **Metadata:** Movie titles injected as node attributes for explainability.

### Phase 3: Heuristic Recommender (Baseline)
- **Algorithm:** Collaborative Filtering using **Jaccard Similarity** on the bipartite graph.
- **Logic:** Neighborhood overlap used to identify "peer users" and predict new links.

### Phase 4: Baseline Evaluation
- **Strategy:** "Link Hide" (80% Train / 20% Test split).
- **Metric:** **Precision @ 5** calculated across a sample of test users.

### Phase 5: Feature Engineering & Hybrid Model (Finalized)
- **User Features:** Normalized Age, One-Hot Encoded Gender and Occupation.
- **Movie Features:** Binary Genre flags (19 categories).
- **Hybrid Logic:** Combined Graph (Jaccard) + Content (Cosine) into a weighted similarity score.
- **Explainability:** Implemented `recommend_with_explanations` to distinguish between **Behavioral Peers** (shared taste) and **Demographic Peers** (similar profile).

### Phase 6: Graph Representation Learning (Node2Vec)
- **Algorithm:** DeepWalk-style uniform random walks.
- **Embeddings:** 64D vectors learned using `gensim`'s Word2Vec.
- **Logic:** Recommendations based on cosine similarity in the latent embedding space.

### Phase 7: Heterogeneous Embeddings (Metapath2Vec)
- **Algorithm:** Structured random walks following the **User-Movie-User (UMU)** metapath.
- **Goal:** Capture deep collaborative filtering intent by forcing the walker to alternate between node types.

### Phase 9: Unified Benchmark & The Grand Finale (Finalized)
- **Goal:** Systematic head-to-head battle between Jaccard, Hybrid, Node2Vec, Metapath2Vec, and LightGCN.
- **Result:** **LightGCN** emerged as the superior model for discovery, achieving the highest **Recall@10 (0.179)** while matching the Jaccard baseline in precision.

### Phase 10: Feature-Augmented GNN (The Breakthrough)
- **Algorithm:** Hybrid LightGCN with Linear Projection Layers.
- **Logic:** Reused Phase 5 features (Demographics & Genres) to initialize node embeddings (Warm-Start).
- **Results:** Achieved a **Precision@10 of 0.3390** and **MRR of 0.6085**, doubling the performance of the pure LightGCN baseline.
- **Insight:** Proved that combining multi-hop structural signals with rich node metadata creates a significantly more accurate and certain recommender system.

---

## 🚀 Next Steps (Roadmap)

### Step 1: Weighted Link Prediction
- Incorporate the actual 1-5 star rating values into the GNN message-passing layers to distinguish between 'Liking' and 'Loving' a movie.

### Step 2: Advanced GNN Architectures
- Experiment with **Graph Attention Networks (GAT)** to see if weighing neighbor influence further boosts precision.
- Implement **Self-Supervised Learning (SSL)** (like Contrastive Learning) to improve embedding robustness in sparse regions.

---

## 📂 File Structure
- `data/`: Raw MovieLens 100k files.
- `eda.ipynb`: Detailed dataset analysis.
- `graph_construction.ipynb`: NetworkX graph initialization and metadata.
- `heuristic_recommender.ipynb`: Jaccard-based baseline logic.
- `feature_engineering.ipynb`: Hybrid model combining content and graph data.
- `node2vec_recommender.ipynb`: Node2Vec implementation and T-SNE.
- `metapath2vec_recommender.ipynb`: Metapath2Vec implementation.
- `lightgcn_recommender.ipynb`: GNN implementation with PyG and BPR loss.
- `feature_augmented_gnn.ipynb`: Advanced GNN utilizing demographics and genres.
- `unified_benchmark.ipynb`: Comprehensive evaluation and professional report.
- `download_data.py`: Setup script.
