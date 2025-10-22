# ==========================
# 1. Imports
# ==========================
from sentence_transformers import SentenceTransformer, util
import language_tool_python
import nltk
import pysbd

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)

# ==========================
# 2. Load AI Model
# ==========================
print("Loading model...")
model = SentenceTransformer('sentence-transformers/roberta-base-nli-mean-tokens')
print("Model loaded successfully!")

# ==========================
# 3. Scoring Functions
# ==========================
def score_structure(essay):
    paragraphs = [p.strip() for p in essay.strip().split('\n\n') if p.strip()]
    transitions = ["however", "therefore", "moreover", "furthermore", "as a result", "in conclusion"]
    transition_count = sum(essay.lower().count(t) for t in transitions)
    if len(paragraphs) >= 3 and transition_count >= 2:
        return 90
    elif len(paragraphs) >= 2:
        return 75
    else:
        return 60

def score_grammar(essay):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(essay)
    error_count = len(matches)
    length = len(essay.split())
    error_rate = error_count / length if length else 1
    if error_rate < 0.02:
        return 95
    elif error_rate < 0.05:
        return 85
    elif error_rate < 0.1:
        return 70
    else:
        return 50

def score_authenticity(essay):
    pronouns = sum(essay.lower().count(p) for p in [" i ", " me ", " my ", " mine ", " myself "])
    reflection_words = ["learned", "realized", "understood", "changed", "discovered", "grew"]
    reflection_hits = sum(essay.lower().count(w) for w in reflection_words)
    score = 50 + (pronouns * 3) + (reflection_hits * 5)
    return min(score, 95)

def score_emotional_impact(essay, model):
    emotional_templates = [
        "I was afraid", "I cried", "I felt proud", "I was ashamed",
        "I failed", "I overcame something difficult", "I was inspired", "I felt grateful"
    ]
    template_embeddings = model.encode(emotional_templates, convert_to_tensor=True)
    seg = pysbd.Segmenter(language="en", clean=True)
    sentences = seg.segment(essay)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    hits = 0
    for i, sentence in enumerate(sentences):
        sim = util.cos_sim(sentence_embeddings[i], template_embeddings)
        if sim.max().item() > 0.65:
            hits += 1
    ratio = hits / len(sentences) if sentences else 0
    return round(ratio * 100, 2)

def score_reflection(essay, model):
    reflection_templates = [
        "I learned something important about myself.",
        "That experience changed me.",
        "I grew from that challenge.",
        "I gained a new perspective.",
        "I became more self-aware.",
        "It taught me a lesson.",
        "I realized something I hadn't before.",
        "This helped me understand myself better."
    ]
    template_embeddings = model.encode(reflection_templates, convert_to_tensor=True)
    seg = pysbd.Segmenter(language="en", clean=True)
    sentences = seg.segment(essay)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    total_hits = 0
    for i, sentence in enumerate(sentences):
        sim_scores = util.cos_sim(sentence_embeddings[i], template_embeddings)
        if sim_scores.max().item() > 0.6:
            total_hits += 1
    return round((total_hits / len(sentences)) * 100, 2) if sentences else 0

def score_word_count(essay, max_words):
    word_count = len(essay.split())
    return word_count <= max_words, word_count

# ==========================
# 4. Main Grading Function
# ==========================
def grade_full_essay(essay, model, max_words):
    reflection = score_reflection(essay, model)
    structure = score_structure(essay)
    grammar = score_grammar(essay)
    authenticity = score_authenticity(essay)
    emotion = score_emotional_impact(essay, model)
    within_limit, word_count = score_word_count(essay, max_words)
    overall = round(
        0.3 * reflection +
        0.2 * grammar +
        0.2 * structure +
        0.15 * authenticity +
        0.15 * emotion,
        2
    )
    return {
        "Overall": overall,
        "Reflection": reflection,
        "Structure": structure,
        "Grammar": grammar,
        "Authenticity": authenticity,
        "Emotional Impact": emotion,
        "Word Count": word_count,
        "Within Limit": within_limit
    }

# ==========================
# 5. Essay Input Prompt
# ==========================
def get_essay_from_user():
    print("\nPaste or type your essay below. Press Enter on an empty line to finish:")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


# ==========================
# 6. Run It
# ==========================
if __name__ == "__main__":
    print("\n=== College Essay Evaluator ===")
    try:
        max_words = int(input("Enter your essay word count requirement: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        exit()

    essay_text = get_essay_from_user()
    results = grade_full_essay(essay_text, model, max_words)

    print("\n=== Essay Evaluation Results ===")
    for k, v in results.items():
        if k == "Within Limit":
            print(f"{k}: {'Yes' if v else 'No (Too long!)'}")
        elif k == "Word Count":
            print(f"{k}: {v} words")
        else:
            print(f"{k}: {v}%")
