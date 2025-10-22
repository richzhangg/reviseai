# ==========================
# Essay Grader (Semantic AI Version)
# ==========================
from sentence_transformers import SentenceTransformer, util
import pysbd
import textstat

# ==========================
#  Load Model
# ==========================
def load_model():
    print("Loading model...")
    model = SentenceTransformer('sentence-transformers/roberta-base-nli-mean-tokens')
    print("Model loaded successfully!\n")
    return model


# ==========================
#  Input Handling 
# ==========================
def get_essay_input():
    print("Enter your essay below (press Enter when done):")
    essay = input("> ").strip()

    # Optional fallback: load from essay.txt if user presses Enter
    if not essay:
        try:
            with open("essay.txt", "r", encoding="utf-8") as f:
                essay = f.read().strip()
                print("Loaded essay from essay.txt\n")
        except FileNotFoundError:
            print("No essay entered or found in essay.txt. Exiting.")
            exit()
    return essay


# ==========================
#  Preprocess Essay
# ==========================
def preprocess_essay(essay, model):
    seg = pysbd.Segmenter(language="en", clean=True)
    sentences = seg.segment(essay)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return sentences, embeddings


# ==========================
#  Semantic Scoring Helpers
# ==========================
def score_with_reference(model, sentences, sentence_embeddings, reference_texts):
    ref_embeddings = model.encode(reference_texts, convert_to_tensor=True)
    sims = util.cos_sim(sentence_embeddings, ref_embeddings)
    avg_similarity = sims.max(dim=1).values.mean().item()
    return round(avg_similarity * 100, 2)


# ==========================
#  Category-Based Scoring
# ==========================
def semantic_scores(essay, model):
    sentences, embeddings = preprocess_essay(essay, model)

    # Define criteria and example reference texts
    categories = {
        "Compelling Story": [
            "I overcame a personal challenge that changed me.",
            "This story reveals who I am through action."
        ],
        "Reflection & Growth": [
            "I learned something meaningful about myself.",
            "This experience helped me grow as a person."
        ],
        "Unique Voice": [
            "My essay feels authentic and personal.",
            "It expresses my individuality and unique perspective."
        ],
        "School Fit": [
            "I align with the university’s mission and community.",
            "I explain how I would contribute to the campus."
        ],
        "Writing & Craft": [
            "The essay is clear, engaging, and well structured.",
            "It has a strong opening and maintains focus throughout."
        ],
    }

    # Semantic scores
    results = {cat: score_with_reference(model, sentences, embeddings, refs)
               for cat, refs in categories.items()}

    # Add readability as part of Writing & Craft
    readability = textstat.flesch_reading_ease(essay)
    readability_score = min(max((readability / 100) * 100, 0), 100)
    results["Writing & Craft"] = round((results["Writing & Craft"] * 0.8 + readability_score * 0.2), 2)

    # Overall average
    results["Overall"] = round(sum(results.values()) / len(results), 2)
    return results


# ==========================
#  Main Execution
# ==========================
if __name__ == "__main__":
    model = load_model()
    essay = get_essay_input()
    results = semantic_scores(essay, model)

    print("\n=== Essay Evaluation ===")
    for k, v in results.items():
        print(f"{k}: {v}%")

    print("\nGrading complete!")




