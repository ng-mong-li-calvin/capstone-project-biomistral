import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import json


training_prompt_template = """
### SYSTEM INSTRUCTION
You are an expert medical research assistant. Your task is to provide a \
concise, factual answer to the USER QUESTION based **only** on the provided \
CONTEXT. Do not use external knowledge.

### CONTEXT
{truncated_context_string}

### USER QUESTION
{question}

### GROUND-TRUTH ANSWER
{answer}
"""


inference_prompt_template = """
You are an expert medical information assistant. Your answers must be \
factually accurate and based **only** on the information provided in the \
CONTEXT section. Do not use external knowledge or fabricate details.

If the CONTEXT is insufficient to answer the question, state that clearly.

**CONTEXT**:
---
{truncated_context_string} 
---

**USER QUESTION**:
{question}

---
**RESPONSE**:
1. Answer the question directly and concisely, using only the provided CONTEXT.
2. Format your response clearly using bullet points for lists (if applicable).

***"""


# Data preprocessing

def preprocess_data(data):
    """ Flattens the json, joins the context list into a single string
        (full_context_string) and standardizes lowercase variables """
    preprocessed_data = []
    for doc_id, record in data:
        entry = {}
        full_context_string = '\n'.join(record.get("CONTEXTS", []))
        entry["doc_id"] = doc_id
        entry["question"] = record.get("QUESTION", "")
        entry["contexts_raw"] = record.get("CONTEXTS", [])
        entry["full_context_string"] = full_context_string
        entry["long_answer"] = record.get("LONG_ANSWER", "")
        entry["final_decision"] = record.get("final_decision", "")
        preprocessed_data.append(entry)
    return preprocessed_data


def fit_tfidf_vectorizer(preprocessed_data, max_features=1000):
    """ Fits the TfidfVectorizer on the entire corpus of context strings. """
    print("Fitting TF-IDF Vectorizer...")
    corpus = [entry['full_context_string'] for entry in preprocessed_data]
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features,
                                       stop_words='english')
    tfidf_vectorizer.fit(corpus)
    print(f"TF-IDF features created:\
            {len(tfidf_vectorizer.get_feature_names_out())}")
    return tfidf_vectorizer


def calculate_sentence_scores(sentences, vectorizer):
    """ Calculate importance of each sentence
        based on TF-IDF and token length """
    sentence_scores = []
    for sentence in enumerate(sentences):
        sentence_vector = vectorizer.transform([sentence[1]])
        sentence_score = sentence_vector.sum()
        sentence_length = len(word_tokenize(sentence[1]))
        # Index, sentence, score, token length
        sentence_scores.append((sentence[0], sentence[1], sentence_score, sentence_length))
    return sentence_scores


def truncate_tfidf(sentences, token_target):
    """ Remove the sentence with the lowest sentence_score
        until total_tokens hits a defined maximum token_target """
    total_tokens = 0
    for sentence in sentences:
        total_tokens += sentence[3]
    while total_tokens > token_target:
        sentence_to_remove = min(sentences, key=lambda x: x[2])
        sentences.remove(sentence_to_remove)
        total_tokens -= sentence_to_remove[3]
    return sentences


def find_cosine_similarity(original_text, truncated_text,
                           vectorizer, display=True):
    """ Finds the cosine similarity between a truncated text
        and the original text """
    documents = [original_text, truncated_text]
    tfidf_vectors = vectorizer.transform(documents)
    original_vector = tfidf_vectors[0]
    truncated_vector = tfidf_vectors[1]
    similarity_score = cosine_similarity(original_vector, truncated_vector)[0][0]
    if display:
        print(f"""**Cosine Similarity (Original vs. Truncated):\
                {similarity_score:.4f}**""")
    return similarity_score


def find_all_cosine_similarity_stats(data, length, vectorizer):
    """ Takes a token length and truncates all documents to that length,
        then finds the min/max/median cosine similarity """
    cosine_similarities = []
    for entry in data:
        original_text = entry.get('full_context_string', '')
        truncated_text = ' '.join([sentence[1] for sentence in truncate_tfidf(
            calculate_sentence_scores(sent_tokenize(original_text), vectorizer), length)])
        cosine_similarities.append(find_cosine_similarity(
            original_text, truncated_text, vectorizer, display=False))

    print(f"\nMin cosine similarity: {min(cosine_similarities)}")
    print(f"Max cosine similarity: {max(cosine_similarities)}")
    print(f"Median cosine similarity: {np.median(cosine_similarities)}")


def generate_formatted_prompt(entry, token_limit, vectorizer):
    """ Takes in the entire entry and generates a prompt for the entry,
    using the truncated context. """

    original_text = entry.get('full_context_string', '')
    sentences = sent_tokenize(original_text)
    if not sentences:
        truncated_text = "No context available."
    else:
        scored_sentences = calculate_sentence_scores(sentences, vectorizer)
        truncated_sentences = truncate_tfidf(scored_sentences, token_limit)
        truncated_text = ' '.join(
            [sentence[1] for sentence in truncated_sentences])

    formatted_prompt = training_prompt_template.format(
        truncated_context_string=truncated_text,
        question=entry['question'],
        answer=entry['long_answer']  # Use 'long_answer' for the RAG/QA task
    ).strip()

    return formatted_prompt


TOKEN_TARGET = 220


def generate_chunks_for_db(preprocessed_data, vectorizer):
    """
    Creates context chunks by truncating the full context to a token limit,
    simulates the vector embedding using the TF-IDF vectorizer, and prepares
    the data for insertion into the vector database.

    Returns: List of (chunk_id, vector_embedding, source_id, text_content)
    """
    db_entries = []
    for doc in preprocessed_data:
        source_id = doc.get('doc_id')
        context_text = doc.get('full_context_string', '')
        if not context_text:
            continue
        sentences = sent_tokenize(context_text)
        if not sentences:
            continue
        # Calculate scores and truncate based on TF-IDF
        scored_sentences = calculate_sentence_scores(sentences, vectorizer)
        truncated_sentences = truncate_tfidf(scored_sentences, TOKEN_TARGET)
        chunk_content = ' '.join([s[1] for s in truncated_sentences])

        # NB: Temporary embedding simulation, will replace later

        sparse_vector = vectorizer.transform([chunk_content])
        # Convert sparse vector to a dense NumPy array for pgvector simulation
        dense_vector = sparse_vector.toarray()[0]

        chunk_id = f"{source_id}_chunk_0"
        db_entries.append((
            chunk_id,  # Unique ID for the chunk
            dense_vector,  # Simulated vector embedding (NumPy array)
            source_id,  # Original Document ID
            chunk_content  # The final text chunk
        ))

    return db_entries


if __name__ == "__main__":
    with open('../data/PQA-L/pqal_test_set.json', 'r') as f:
        test_data = json.load(f)
    with open('../data/PQA-L/pqal_train_dev_set.json', 'r') as f:
        train_data = json.load(f)
    all_data = {**test_data, **train_data}

    preprocessed_data = preprocess_data(all_data.items())
    vectorizer = fit_tfidf_vectorizer(preprocessed_data)

    kb_upload_records = [
        {
            'doc_id': entry['doc_id'],
            'context_text': entry['full_context_string'],
            'question': entry['question'],
            'long_answer': entry['long_answer']
        }
        for entry in preprocessed_data
    ]

    df_upload = pd.DataFrame(kb_upload_records)
    output_path = '../data/bedrock_kb_source_data.csv'
    df_upload.to_csv(output_path, index=False, encoding='utf-8')