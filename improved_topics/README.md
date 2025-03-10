# Improved Topic Detection

This directory contains enhanced methods for topic detection in text documents, significantly improving upon the basic regex pattern approach used in the original notebook.

## Key Improvements

1. **Expanded Topic Keywords:** Comprehensive sets of keywords for each topic (e.g., "Roman", "Art", "History")
2. **Text Preprocessing:** Lowercasing, punctuation removal, stopword filtering, and stemming
3. **Multiple Topic Assignment:** Documents can belong to more than one topic category
4. **Scoring Mechanisms:** Topics are ranked by relevance, not just presence
5. **Entity Recognition:** Using named entities to enhance topic detection
6. **Semantic Matching:** TF-IDF vectorization for capturing semantic similarity

## Methods

The package provides three main approaches to topic detection:

### 1. Frequency-Based with Stemming
- Uses stemming to match word variations
- Counts keyword occurrences weighted by significance
- Fast and reliable for most documents

### 2. TF-IDF Similarity
- Creates topic vectors from keywords
- Calculates semantic similarity between documents and topics
- More sophisticated than direct keyword matching

### 3. Entity-Enhanced Detection
- Combines keyword matching with named entity recognition
- Maps entity types to potential topics
- Provides contextual topic detection

## Usage

### Option 1: Using the demo notebook
Run the `demo.ipynb` notebook to see a comparison of all three methods on a sample of documents.

### Option 2: Integrating into your existing notebook
Use the `update_notebook.py` script to add improved topic detection to your existing notebook:

```bash
python improved_topics/update_notebook.py
```

### Option 3: Direct API usage
Import the functions directly in your code:

```python
from improved_topics.topic_detection import detect_topics_by_frequency, detect_topics_tfidf, detect_topics_with_entities

# Choose one of the methods
topics = detect_topics_by_frequency(text, min_occurrences=1, max_topics=3)
```

## Requirements

- pandas
- numpy
- scikit-learn
- nltk
- tqdm


## Performance Considerations

- The frequency-based method is the fastest but less contextual
- The TF-IDF method provides better semantic matching but requires vectorization
- The entity-enhanced method provides the most contextual results but has the highest computational cost

Choose the method that best balances accuracy and performance for your specific use case. 