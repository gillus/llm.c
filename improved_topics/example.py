#!/usr/bin/env python3
"""
Example script demonstrating how to use the improved topic detection methods.
"""

import sys
import pandas as pd
from topic_detection import (
    detect_topics_by_frequency, 
    detect_topics_tfidf, 
    detect_topics_with_entities, 
    calculate_topic_vectors,
    setup_nltk_for_ner
)

def main():
    # Sample texts to test
    sample_texts = [
        "The Roman Empire was a vast political entity that lasted for centuries, with emperors like Augustus and Caesar ruling from Rome.",
        "Mozart composed many symphonies and operas that are still performed in concert halls around the world today.",
        "The scientific method requires developing a hypothesis, conducting experiments, and analyzing the results to reach conclusions.",
        "The Mona Lisa is a famous painting by Leonardo da Vinci displayed in the Louvre Museum in Paris.",
        "The Gregorian calendar is the most widely used civil calendar system in the world today.",
        "Shakespeare wrote many plays and poems that are considered masterpieces of English literature."
    ]

    # Initialize for TF-IDF method
    print("Initializing TF-IDF vectorizer and topic vectors...")
    vectorizer, topic_vectors = calculate_topic_vectors()
    
    # Initialize for Entity method
    print("Setting up NLTK for Named Entity Recognition...")
    setup_nltk_for_ner()
    
    # Test each method on sample texts
    results = []
    
    for i, text in enumerate(sample_texts):
        print(f"\nSample {i+1}: {text[:60]}...")
        
        # Method 1: Frequency-based
        freq_topics = detect_topics_by_frequency(text)
        print(f"  Frequency method: {freq_topics}")
        
        # Method 2: TF-IDF
        tfidf_topics = detect_topics_tfidf(text, vectorizer, topic_vectors)
        print(f"  TF-IDF method: {tfidf_topics}")
        
        # Method 3: Entity-enhanced
        entity_topics = detect_topics_with_entities(text)
        print(f"  Entity method: {entity_topics}")
        
        # Store results
        results.append({
            'Sample': f"Sample {i+1}",
            'Text': text[:100] + "...",
            'Frequency Method': ', '.join(freq_topics),
            'TF-IDF Method': ', '.join(tfidf_topics),
            'Entity Method': ', '.join(entity_topics)
        })
    
    # Create a DataFrame for comparison
    df = pd.DataFrame(results)
    print("\n\nComparison of Methods:")
    print(df[['Sample', 'Frequency Method', 'TF-IDF Method', 'Entity Method']])
    
    print("\nThis example demonstrates the three improved topic detection methods.")
    print("1. Frequency-based with stemming: Fast and reliable")
    print("2. TF-IDF similarity: Better semantic matching")
    print("3. Entity-enhanced: Most comprehensive contextual detection")

if __name__ == "__main__":
    main() 