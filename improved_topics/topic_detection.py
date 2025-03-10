import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Define a more comprehensive set of topics with related keywords
topic_keywords = {
    "Roman": [
        "roman", "rome", "empire", "caesar", "augustus", "emperor", "gladiator", 
        "senate", "republic", "colosseum", "toga", "legion", "chariot"
    ],
    "Art": [
        "art", "painting", "sculpture", "artist", "masterpiece", "gallery", "exhibit",
        "portrait", "canvas", "palette", "brush", "museum", "artistic", "renaissance"
    ],
    "History": [
        "history", "historical", "ancient", "medieval", "century", "era", "dynasty",
        "civilization", "kingdom", "revolution", "war", "battle", "monarch", "conquest"
    ],
    "Music": [
        "music", "song", "instrument", "composer", "orchestra", "symphony", "melody",
        "rhythm", "concert", "opera", "jazz", "piano", "guitar", "violin", "band"
    ],
    "Science": [
        "science", "scientific", "research", "experiment", "laboratory", "theory",
        "hypothesis", "discovery", "chemistry", "physics", "biology", "equation",
        "scientist", "technology", "innovation"
    ],
    "Literature": [
        "literature", "book", "novel", "author", "poetry", "poem", "playwright",
        "fiction", "character", "narrative", "plot", "writer", "publication", "story"
    ],
    "Calendar": [
        "calendar", "date", "month", "year", "day", "schedule", "anniversary", 
        "holiday", "festival", "season", "chronology", "time", "period"
    ],
    "Alphabet": [
        "alphabet", "letter", "character", "script", "writing", "language",
        "vowel", "consonant", "glyph", "symbol", "phonetic", "pronunciation"
    ],
    "Geography": [
        "geography", "map", "terrain", "location", "continent", "country", "city",
        "river", "mountain", "ocean", "sea", "lake", "island", "region", "border"
    ],
    "Religion": [
        "religion", "religious", "faith", "belief", "worship", "deity", "god",
        "prayer", "ritual", "sacred", "temple", "church", "mosque", "shrine"
    ]
}

# Preprocess function for texts
def preprocess_text(text):
    """Preprocess text for analysis by lowercasing, removing punctuation, and stemming."""
    if not text:
        return ""
        
    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation and numbers
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)

# Function to detect topics using keyword frequency with stemming
def detect_topics_by_frequency(text, min_occurrences=1, max_topics=3):
    """
    Detect topics based on frequency of stemmed keywords in the text.
    
    Args:
        text: Text to analyze
        min_occurrences: Minimum number of occurrences to consider a topic relevant
        max_topics: Maximum number of topics to return
        
    Returns:
        List of detected topics sorted by relevance
    """
    if not text or len(text) < 10:
        return ["Unknown"]
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    words = processed_text.split()
    
    if not words:
        return ["Unknown"]
    
    # Prepare stemmed keywords
    stemmer = PorterStemmer()
    stemmed_keywords = {}
    for topic, keywords in topic_keywords.items():
        stemmed_keywords[topic] = [stemmer.stem(keyword) for keyword in keywords]
    
    # Count keyword occurrences for each topic
    topic_scores = {topic: 0 for topic in topic_keywords}
    word_counts = Counter(words)
    
    for topic, stemmed_kws in stemmed_keywords.items():
        for word in stemmed_kws:
            if word in word_counts:
                # Weight by inverse document frequency (common words get lower weight)
                topic_scores[topic] += word_counts[word]
    
    # Filter topics that have at least min_occurrences
    relevant_topics = [(topic, score) for topic, score in topic_scores.items() 
                       if score >= min_occurrences]
    
    # Sort by score and take top max_topics
    relevant_topics.sort(key=lambda x: x[1], reverse=True)
    relevant_topics = relevant_topics[:max_topics]
    
    if not relevant_topics:
        return ["Unknown"]
    
    # Return just the topic names
    return [topic for topic, _ in relevant_topics if _ > 0]

# TF-IDF based approach
def calculate_topic_vectors():
    """Create TF-IDF vectors for each topic based on its keywords."""
    # Combine all topic keywords into a single corpus
    all_keywords = []
    topic_indices = {}
    current_idx = 0
    
    for topic, keywords in topic_keywords.items():
        start_idx = current_idx
        end_idx = start_idx + len(keywords)
        all_keywords.extend(keywords)
        topic_indices[topic] = (start_idx, end_idx)
        current_idx = end_idx
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_keywords)
    
    # Calculate a centroid vector for each topic
    topic_vectors = {}
    for topic, (start, end) in topic_indices.items():
        if end > start:  # Ensure we have keywords for this topic
            topic_vectors[topic] = np.mean(tfidf_matrix[start:end].toarray(), axis=0)
    
    return vectorizer, topic_vectors

def detect_topics_tfidf(text, vectorizer, topic_vectors, threshold=0.05, max_topics=3):
    """
    Detect topics using TF-IDF similarity.
    
    Args:
        text: Text to analyze
        vectorizer: Fitted TF-IDF vectorizer
        topic_vectors: Pre-calculated topic centroid vectors
        threshold: Minimum similarity score to consider a topic relevant
        max_topics: Maximum number of topics to return
        
    Returns:
        List of detected topics sorted by similarity
    """
    if not text or len(text) < 10:
        return ["Unknown"]
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Vectorize the text
    try:
        text_vector = vectorizer.transform([processed_text])
    except:
        return ["Unknown"]  # Handle cases where vectorizer fails
    
    # Calculate similarity to each topic
    similarities = {}
    for topic, topic_vector in topic_vectors.items():
        similarity = cosine_similarity(text_vector, topic_vector.reshape(1, -1))[0][0]
        similarities[topic] = similarity
    
    # Filter and sort by similarity
    relevant_topics = [(topic, sim) for topic, sim in similarities.items() 
                      if sim > threshold]
    relevant_topics.sort(key=lambda x: x[1], reverse=True)
    relevant_topics = relevant_topics[:max_topics]
    
    if not relevant_topics:
        return ["Unknown"]
    
    # Return just the topic names
    return [topic for topic, _ in relevant_topics]

# Named entity-based topic detection
def setup_nltk_for_ner():
    """Download and set up NLTK for NER if needed."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('maxent_ne_chunker')
        nltk.data.find('words')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

def detect_topics_with_entities(text, min_occurrences=1, max_topics=3):
    """
    Detect topics using both keyword frequency and named entity recognition.
    
    This function combines keyword-based detection with named entity types to
    enrich topic detection with context from entities.
    """
    if not text or len(text) < 10:
        return ["Unknown"]
    
    # First use the frequency-based method
    topics_by_keywords = detect_topics_by_frequency(text, min_occurrences, max_topics+2)
    
    # Set up NLTK for NER
    setup_nltk_for_ner()
    
    # Extract entities from the text
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    ne_chunks = nltk.ne_chunk(pos_tags)
    
    # Map entity types to possible topics
    entity_topic_map = {
        'PERSON': ['History'],
        'GPE': ['Geography', 'History'],  # Geo-political entity
        'ORGANIZATION': ['History'],
        'LOCATION': ['Geography'],
        'DATE': ['History', 'Calendar'],
        'FACILITY': ['Art', 'Geography']
    }
    
    # Count entity-derived topics
    entity_topics = []
    for chunk in ne_chunks:
        if hasattr(chunk, 'label'):
            entity_type = chunk.label()
            if entity_type in entity_topic_map:
                entity_topics.extend(entity_topic_map[entity_type])
    
    # Add entity-based topics to keyword-based topics
    all_topics = topics_by_keywords + list(set(entity_topics))
    
    # Count frequency of each topic in the combined list
    topic_counts = Counter(all_topics)
    
    # Sort by count and take top max_topics
    top_topics = [topic for topic, _ in topic_counts.most_common(max_topics)]
    
    return top_topics if top_topics else ["Unknown"]

# Helper function to process a batch of documents
def process_documents_with_improved_topics(documents, method='frequency', min_occurrences=1, max_topics=3):
    """
    Process a batch of documents with the specified topic detection method.
    
    Args:
        documents: List of text documents
        method: Topic detection method ('frequency', 'tfidf', or 'entity')
        min_occurrences: Minimum occurrences for frequency method
        max_topics: Maximum topics to return per document
        
    Returns:
        List of topic lists for each document
    """
    # Set up for TF-IDF method if needed
    if method == 'tfidf':
        vectorizer, topic_vectors = calculate_topic_vectors()
    
    results = []
    for doc in documents:
        if method == 'frequency':
            topics = detect_topics_by_frequency(doc, min_occurrences, max_topics)
        elif method == 'tfidf':
            topics = detect_topics_tfidf(doc, vectorizer, topic_vectors, threshold=0.05, max_topics=max_topics)
        elif method == 'entity':
            topics = detect_topics_with_entities(doc, min_occurrences, max_topics)
        else:
            topics = ["Unknown"]  # Default fallback
        
        results.append(topics)
    
    return results 