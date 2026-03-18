# Movie Recommender System (Link Prediction on Graphs)

A movie recommendation engine built by reframing the problem as **Link Prediction on a Bipartite Graph**. Instead of traditional matrix factorization, we treat users and movies as nodes in a network, where an "edge" represents a high rating (>= 4).

## 🚀 Key Highlights
- **Architecture:** Transitioned from simple heuristics (Jaccard) to modern Deep Graph Learning (**LightGCN**).
- **Hybrid Signal:** Combines **Collaborative Filtering** (graph topology) with **Content-Based Filtering** (demographics and genres).
- **Breakthrough:** Our **Feature-Augmented GNN** achieved a **Precision@10 of 0.3390**, doubling the performance of pure structural models.
- **Explainability:** Includes a `recommend_with_explanations` logic to distinguish between Behavioral and Demographic peers.

## 📊 Performance Leaderboard
| Model | Precision@10 | Recall@10 | MRR |
| :--- | :--- | :--- | :--- |
| **Augmented GNN (Phase 10)** | **0.3390** | **0.1697** | **0.6085** |
| Jaccard Baseline | 0.1460 | 0.1680 | 0.3815 |
| LightGCN (Pure) | 0.1440 | 0.1793 | 0.3754 |
| GAT / GraphSAGE | ~0.120 | - | - |
| Hybrid Model | 0.1030 | 0.0888 | 0.2750 |
| Node2Vec/Metapath2Vec | < 0.010 | < 0.010 | < 0.010 |

## 📂 File Structure
- `graph_construction.ipynb`: Builds the bipartite graph with NetworkX.
- `heuristic_recommender.ipynb`: Jaccard similarity baseline.
- `feature_engineering.ipynb`: Demographic and genre vectorization.
- `node2vec_recommender.ipynb`: Random walk-based shallow embeddings.
- `lightgcn_recommender.ipynb`: Modern GNN implementation using PyTorch Geometric.
- **`feature_augmented_gnn.ipynb`**: **CHAMPION MODEL (Phase 10).**
- `weighted_link_prediction.ipynb`: Phase 11 exploration.
- `gat_recommender.ipynb`: Phase 12 exploration (GAT).
- `graphsage_recommender.ipynb`: Phase 13 exploration (SAGE).
- `unified_benchmark.ipynb`: Head-to-head comparison and final report.

## 🛠️ Installation
```bash
pip install torch torch-geometric pandas numpy networkx scikit-learn gensim
python download_data.py
```

## 🎯 Conclusion & Verdict
After exploring weighted edges, attention mechanisms (GAT), and inductive learning (GraphSAGE), we discovered a **"Complexity Tax."** The most effective model for this bipartite link prediction task remains the **Feature-Augmented LightGCN (Phase 10)**. Linear structural aggregation combined with raw feature projection provides the cleanest signal for recommendation.
