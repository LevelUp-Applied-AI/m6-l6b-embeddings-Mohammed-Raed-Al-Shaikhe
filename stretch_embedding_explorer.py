import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel


def load_glove(filepath):

    embeddings = {}

    with open(filepath, "r", encoding="utf-8") as file:

        for line in file:

            parts = line.strip().split()

            word = parts[0]

            vector = np.array(parts[1:], dtype=np.float32)

            embeddings[word] = vector

    return embeddings


def extract_bert_embedding(text, tokenizer, model):
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"]

    # Expand mask shape for broadcasting
    mask = attention_mask.unsqueeze(-1)

    # Zero out padding tokens
    masked_embeddings = last_hidden_state * mask

    # Sum token embeddings
    summed = masked_embeddings.sum(dim=1)

    # Count valid tokens
    counts = mask.sum(dim=1)

    # Mean pooling
    mean_pooled = summed / counts

    return mean_pooled.squeeze().numpy()


def build_word_visualization(glove):

    # 5 semantic categories
    word_categories = {

        "sports": [
            "football", "basketball", "tennis", "soccer",
            "baseball", "golf", "cricket", "hockey",
            "athlete", "coach", "tournament", "stadium",
            "league", "championship", "goal", "match",
            "player", "team", "score", "victory",
            "defense", "offense", "training", "competition",
            "olympics", "medal", "referee", "captain",
            "fitness", "running", "swimming", "cycling",
            "boxing", "wrestling", "volleyball", "rugby",
            "fans", "season", "record", "strategy"
        ],

        "technology": [
            "computer", "software", "internet", "database",
            "network", "server", "python", "programming",
            "algorithm", "hardware", "application", "digital",
            "artificial", "intelligence", "robot", "cybersecurity",
            "cloud", "system", "developer", "website",
            "mobile", "device", "processor", "memory",
            "keyboard", "monitor", "laptop", "smartphone",
            "engineering", "automation", "machine", "data",
            "analytics", "virtual", "crypto", "bitcoin",
            "code", "technology", "innovation", "electronics"
        ],

        "countries": [
            "jordan", "france", "germany", "italy",
            "spain", "canada", "brazil", "china",
            "japan", "india", "egypt", "mexico",
            "russia", "america", "england", "turkey",
            "argentina", "australia", "sweden", "norway",
            "iran", "iraq", "syria", "lebanon",
            "palestine", "saudi", "qatar", "uae",
            "africa", "europe", "asia", "london",
            "paris", "tokyo", "berlin", "madrid",
            "rome", "moscow", "beijing", "amman"
        ],

        "emotions": [
            "happy", "sad", "angry", "joy",
            "fear", "love", "hate", "excited",
            "depressed", "anxious", "surprised", "nervous",
            "smile", "cry", "laugh", "hope",
            "peace", "stress", "worried", "calm",
            "emotion", "feeling", "grief", "pleasure",
            "pain", "confidence", "proud", "embarrassed",
            "lonely", "friendly", "kind", "upset",
            "enthusiasm", "motivation", "passion", "anger",
            "delight", "disappointment", "frustration", "relaxed"
        ],

        "finance": [
            "money", "bank", "market", "finance",
            "stock", "investment", "economy", "trade",
            "business", "profit", "loss", "salary",
            "income", "tax", "budget", "capital",
            "debt", "credit", "cash", "insurance",
            "loan", "fund", "currency", "payment",
            "wealth", "financial", "corporation", "industry",
            "revenue", "shares", "inflation", "economic",
            "customer", "retail", "startup", "entrepreneur",
            "commerce", "banking", "dollar", "bitcoin"
        ]
    }

    vectors = []
    labels = []
    words = []

    # Collect valid GloVe words
    for category, category_words in word_categories.items():

        for word in category_words:

            if word in glove:

                vectors.append(glove[word])

                labels.append(category)

                words.append(word)

    X = np.array(vectors)

    print(f"Word embedding matrix shape: {X.shape}")

    # t-SNE reduction
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    X_2d = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(14, 10))

    for category in word_categories.keys():

        indices = [
            i for i, label in enumerate(labels)
            if label == category
        ]

        plt.scatter(
            X_2d[indices, 0],
            X_2d[indices, 1],
            label=category
        )

    # Annotate important words
    important_words = [
        "football",
        "computer",
        "jordan",
        "happy",
        "bank",
        "bitcoin",
        "technology",
        "economy",
        "love",
        "basketball"
    ]

    for i, word in enumerate(words):

        if word in important_words:

            plt.annotate(
                word,
                (X_2d[i, 0], X_2d[i, 1])
            )

    plt.title("GloVe Word Embeddings Visualization (t-SNE)")
    plt.legend()
    plt.tight_layout()

    plt.savefig("word_embeddings.png")

    print("Saved: word_embeddings.png")


def build_document_visualization(df, tokenizer, model):
    
    # Select 4 articles per category
    sampled = (
        df.groupby("category")
        .head(4)
        .reset_index(drop=True)
    )

    texts = sampled["text"].tolist()

    categories = sampled["category"].tolist()

    titles = sampled["text"].tolist()

    print(f"Selected {len(texts)} articles")

    embeddings = []

    for i, text in enumerate(texts):

        print(f"Embedding article {i + 1}/{len(texts)}")

        embedding = extract_bert_embedding(
            text,
            tokenizer,
            model
        )

        embeddings.append(embedding)

    X = np.array(embeddings)

    print(f"Document embedding matrix shape: {X.shape}")

    # t-SNE reduction
    tsne = TSNE(
        n_components=2,
        perplexity=5,
        random_state=42
    )

    X_2d = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(14, 10))

    unique_categories = list(set(categories))

    for category in unique_categories:

        indices = [
            i for i, c in enumerate(categories)
            if c == category
        ]

        plt.scatter(
            X_2d[indices, 0],
            X_2d[indices, 1],
            label=category
        )

    # Annotate article titles
    for i, text in enumerate(titles):

        short_text = text[:30].replace("\n", " ") + "..."

        plt.annotate(
            short_text,
            (X_2d[i, 0], X_2d[i, 1]),
            fontsize=8
        )

    plt.title("BBC News DistilBERT Embeddings Visualization (t-SNE)")
    plt.legend()
    plt.tight_layout()

    plt.savefig("document_embeddings.png")

    print("Saved: document_embeddings.png")


if __name__ == "__main__":

    # Load BBC dataset
    df = pd.read_csv("data/bbc_news.csv")

    print(f"Loaded {len(df)} BBC News articles")

    # Load GloVe embeddings
    glove = load_glove("data/glove_50k_50d.txt")

    print(f"Loaded {len(glove)} GloVe vectors")

    # Build word visualization
    build_word_visualization(glove)

    # Load DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )

    model.eval()

    print("Loaded DistilBERT model")

    # Build document visualization
    build_document_visualization(
        df,
        tokenizer,
        model
    )

    print("\nStretch assignment complete!")