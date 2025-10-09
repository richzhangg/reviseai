import os
import nltk
from sentence_transformers import SentenceTransformer, util
import pysbd
import language_tool_python

# Optional environment setup (prevents tokenizer parallelism warnings)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Make sure nltk data is available
nltk.download('punkt', download_dir='/tmp')

# Load SentenceTransformer model (RoBERTa-based)
print("Loading SentenceTransformer model...")
model = SentenceTransformer('sentence-transformers/roberta-base-nli-mean-tokens')

# Example reflective templates
reflection_templates = [
    "I learned something important about myself.",
    "That experience changed me.",
    "I grew from that challenge.",
    "I gained a new perspective.",
    "I became more self-aware.",
    "It taught me a lesson.",
    "I realized something I hadn't before.",
    "This helped me understand myself better.",
]

# Encode the reflection templates
print("Encoding reflection templates...")
template_embeddings = model.encode(reflection_templates, convert_to_tensor=True)

# Your essay text — replace this with user input or load from a file
essay_text = """
I used to hate group projects because I preferred working alone.
But after joining the robotics team, I realized that collaboration brings out the best in everyone.
I learned how to trust my teammates and communicate better.
This experience changed how I view leadership and teamwork.
"""

# Sentence segmentation using pysbd
print("Segmenting sentences...")
seg = pysbd.Segmenter(language="en", clean=True)
sentences = seg.segment(essay_text)

# Encode each sentence
print("Encoding essay sentences...")
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

# Calculate similarity to reflection templates
reflective_sentences = []
threshold = 0.6
total_reflective = 0

for i, sentence in enumerate(sentences):
    sim_scores = util.cos_sim(sentence_embeddings[i], template_embeddings)
    max_score = sim_scores.max().item()

    if max_score > threshold:
        reflective_sentences.append((sentence, round(max_score, 3)))
        total_reflective += 1

# Calculate reflection score (percentage)
reflection_score = round((total_reflective / len(sentences)) * 100, 2)

# Display results
print("\n===============================")
print(f" Reflection Score: {reflection_score}/100")
print("===============================")
print("\n Reflective Sentences Detected:")

if reflective_sentences:
    for sent, sim in reflective_sentences:
        print(f"- \"{sent}\" (similarity: {sim})")
else:
    print("No strongly reflective sentences found.")

print("\nAnalysis complete.")

#### GRAMMAR SECTION

def analyze_grammar(essay_text):

    tool = language_tool_python.LanguageTool('en-US')

    matches = tool.check(essay_text)
    num_errors = len(matches)
    word_count = len(essay_text.split())
    
    if word_count == 0:
        return {"grammar_score": 0, "num_errors": 0, "example_error": "No text provided."}

    # Softer penalty curve for short essays
    penalty = (num_errors / word_count) * 500
    grammar_score = max(0, 100 - penalty)

    feedback = {
        "grammar_score": round(grammar_score, 2),
        "num_errors": num_errors,
        "example_error": matches[0].message if matches else "No errors detected."
    }
    return feedback

essay_text = """This is an example text with speling mistackes."""

grammar_results = analyze_grammar(essay_text)
print(grammar_results)
