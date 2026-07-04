"""Lab 2.5 — explore pretrained embeddings: similarity and nearest neighbours.

Run: python examples/05_embeddings.py
Needs the ``viz`` extra to draw the t-SNE plot (matplotlib + scikit-learn).
"""

from small_llm.embeddings import load_embeddings, most_similar, similarity

URL = (
    "https://storage.googleapis.com/dm-educational/assets/"
    "ai_foundations/gemma_embeddings.npz"
)

embeddings, labels = load_embeddings(URL)
print(f"Loaded {len(labels)} embeddings of dimension {embeddings.shape[1]}.\n")

for a, b in [("king", "queen"), ("joy", "happy"), ("king", "bus")]:
    if a in labels and b in labels:
        print(f"sim({a!r}, {b!r}) = {similarity(a, b, embeddings, labels):.2f}")

if "king" in labels:
    print("\nNearest to 'king':", most_similar("king", embeddings, labels, top_k=5))

# Uncomment to visualize (requires the `viz` extra):
# import matplotlib.pyplot as plt
# from small_llm.embeddings import plot_embeddings_tsne
# plot_embeddings_tsne(embeddings, labels)
# plt.show()
