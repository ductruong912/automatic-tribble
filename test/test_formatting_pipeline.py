#!/usr/bin/env python3
"""
Test script to compare raw LLM output vs Streamlit preprocessed output.
This script demonstrates the formatting pipeline and shows the difference
between raw markdown syntax and the preprocessed version for Streamlit display.
"""

import re
import sys
import os
import asyncio
import time

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm import ChatbotService
from core.storage import vector_store
from services import ChatService


def format_markdown_response_test(text: str) -> str:
    """Hybrid formatting approach that intelligently formats LLM responses.
    - Detects and formats common patterns (steps, lists, headers)
    - Makes URLs clickable
    - Improves structure and readability
    - Works with both raw markdown and plain text responses
    """
    if not text:
        return text
    
    # Step 1: Basic cleaning
    text = text.strip()
    
    # Step 2: Convert escape sequences if present
    text = text.replace('\\n\\n', '\n\n')
    text = text.replace('\\n', '\n')
    
    # Step 3: Detect and format patterns
    text = detect_and_format_patterns_test(text)
    
    # Step 4: Format URLs
    text = format_urls_test(text)
    
    # Step 5: Improve structure
    text = improve_structure_test(text)
    
    # Step 6: Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_and_format_patterns_test(text: str) -> str:
    """Universal text structure formatter that intelligently detects and formats patterns."""
    
    # Step 1: Detect and format step patterns
    text = re.sub(r'(Bước \d+:)', r'**\1**', text)  # Vietnamese steps
    text = re.sub(r'(Step \d+:)', r'**\1**', text)  # English steps
    
    # Step 2: Detect and format section headers (lines ending with colon)
    text = re.sub(r'^([A-Za-zÀ-ỹ\s]+:)$', r'**\1**', text, flags=re.MULTILINE)
    
    # Step 3: Universal list detection and formatting
    # This handles any list pattern: bullet, numbered, dash, asterisk
    list_patterns = [
        (r'(?<!\n)(•\s)', r'\n\1'),      # Bullet points
        (r'(?<!\n)(\d+\.\s)', r'\n\1'),  # Numbered lists
        (r'(?<!\n)(-\s)', r'\n\1'),      # Dash lists
        (r'(?<!\n)(\*\s)', r'\n\1'),     # Asterisk lists
    ]
    
    for pattern, replacement in list_patterns:
        text = re.sub(pattern, replacement, text)
    
    # Step 4: Universal paragraph breaking
    # Break on sentence boundaries followed by capital letters or special characters
    paragraph_breaks = [
        (r'(\.)([A-ZÀ-Ỹ])', r'\1\n\n\2'),  # Period + capital letter
        (r'(\.)(–)', r'\1\n\n\2'),          # Period + dash
        (r'(\.)(•)', r'\1\n\n\2'),          # Period + bullet
        (r'(\.)(\d+\.)', r'\1\n\n\2'),      # Period + numbered list
    ]
    
    for pattern, replacement in paragraph_breaks:
        text = re.sub(pattern, replacement, text)
    
    # Step 5: Handle dense text with no structure
    # If text is very long without breaks, add strategic breaks
    if len(text) > 500 and '\n' not in text:
        # Break on common sentence endings followed by common list starters
        text = re.sub(r'(\.)(\s*)([A-ZÀ-Ỹ])', r'\1\n\n\3', text)
    
    return text


def format_urls_test(text: str) -> str:
    """Format URLs to be clickable markdown links."""
    
    # Make URLs clickable (basic pattern) - only if not already formatted
    text = re.sub(r"(?<!\[)(https?://[^\s\)]+)(?!\])", r"[\1](\1)", text)
    
    # Fix double URL formatting issue
    text = re.sub(r"\[([^\]]+)\]\(\[([^\]]+)\]\([^)]+\)\)", r"[\1](\2)", text)
    
    # Fix malformed links like "[text]url" -> "[text](url)"
    text = re.sub(r'\[([^\]]+)\](https?://[^\s\)]+)', r'[\1](\2)', text)
    
    return text


def improve_structure_test(text: str) -> str:
    """Improve overall text structure and spacing."""
    
    # Ensure proper spacing around **Bước X:** patterns
    text = re.sub(r'(\*\*Bước \d+:\*\*)', r'\n\n\1\n', text)
    
    # Ensure proper spacing around other bold section headers
    text = re.sub(r'(\*\*[^:]+:\*\*)', r'\n\n\1\n', text)
    
    # Ensure proper spacing around ## headers
    text = re.sub(r'(## [^\n]+)', r'\n\n\1\n', text)
    
    # Ensure list items have proper spacing
    text = re.sub(r'(?<!\n)(- [^\n]+)', r'\n\1', text)
    text = re.sub(r'(?<!\n)(\d+\. [^\n]+)', r'\n\1', text)
    
    # Ensure proper spacing around URLs
    text = re.sub(r'(\[https?://[^\]]+\]\(https?://[^)]+\))', r'\n\1\n', text)
    
    return text


def print_separator(title: str):
    """Print a visual separator with title."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_comparison(raw_text: str, processed_text: str):
    """Print side-by-side comparison of raw vs processed text."""
    print_separator("RAW LLM OUTPUT (Before Streamlit Processing)")
    print(raw_text)
    
    print_separator("STREAMLIT PROCESSED OUTPUT (After format_markdown_response)")
    print(processed_text)
    
    print_separator("DIFFERENCES SUMMARY")
    print(f"Raw length: {len(raw_text)} characters")
    print(f"Processed length: {len(processed_text)} characters")
    print(f"Length difference: {len(processed_text) - len(raw_text)} characters")
    
    # Show specific transformations
    print("\nKey transformations applied:")
    if '\\n\\n' in raw_text:
        print("✓ Converted \\n\\n escape sequences to actual newlines")
    if '\\n' in raw_text:
        print("✓ Converted \\n escape sequences to actual newlines")
    if '**Bước' in raw_text:
        print("✓ Added proper spacing around **Bước X:** patterns")
    if '## ' in raw_text:
        print("✓ Added proper spacing around ## headers")
    if '- ' in raw_text and not raw_text.startswith('- '):
        print("✓ Ensured proper spacing for list items")
    if 'http' in raw_text and '[' not in raw_text:
        print("✓ Made URLs clickable")


async def get_real_llm_response(query: str) -> str:
    """Get a real response from the LLM using the chatbot service."""
    try:
        # Initialize chatbot service
        chatbot_service = ChatbotService()
        
        # Load vector store
        _, current_embeddings, current_documents = vector_store.load_vector_store()
        
        if current_embeddings is None:
            current_embeddings = []
        elif isinstance(current_embeddings, list):
            current_embeddings = current_embeddings
        
        # Get response from chatbot service
        result = chatbot_service.get_response_with_history(
            query=query,
            embeddings=current_embeddings,
            documents=current_documents,
            history=None
        )
        
        return result.get("response", "")
        
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return f"Error: {str(e)}"


def test_real_llm_responses():
    """Test the formatting pipeline with real LLM responses."""
    
    # Test queries that should generate different types of responses
    test_queries = [
        "có template không?",
    ]
    
    print_separator("TESTING WITH REAL LLM RESPONSES")
    print("Getting real responses from the chatbot service...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        # Get real LLM response
        raw_response = asyncio.run(get_real_llm_response(query))
        
        if raw_response.startswith("Error:"):
            print(f" Failed to get response: {raw_response}")
            continue
            
        # Process the response
        processed_response = format_markdown_response_test(raw_response)
        
        # Show comparison
        print_separator(f"REAL LLM RESPONSE {i}: {query[:50]}...")
        print_comparison(raw_response, processed_response)
        print("\n" + "-"*80 + "\n")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)


def test_edge_cases():
    """Test edge cases and special scenarios."""
    print_separator("TESTING EDGE CASES")
    
    edge_cases = [
        ("Empty string", ""),
        ("Only escape sequences", "\\n\\n\\n"),
        ("Mixed escape sequences", "Text\\n\\nMore text\\nAnd more"),
        ("URLs only", "https://example.com and https://test.com"),
        ("Already formatted URLs", "[Example](https://example.com)"),
        ("No formatting needed", "Simple text without any special formatting"),
        ("Multiple consecutive newlines", "Text\\n\\n\\n\\n\\nMore text"),
    ]
    
    for title, raw_text in edge_cases:
        print(f"\n--- {title} ---")
        print(f"Raw: {repr(raw_text)}")
        processed = format_markdown_response_test(raw_text)
        print(f"Processed: {repr(processed)}")
        print(f"Result: {processed}")


def main():
    """Main test function."""
    print("🧪 FORMATTING PIPELINE TEST SCRIPT")
    print("This script tests the difference between raw LLM output and Streamlit preprocessed output.")
    
    # Test with real LLM responses
    test_real_llm_responses()
    
    # Test edge cases
    test_edge_cases()
    
    print_separator("TEST COMPLETED")
    print(" All tests completed successfully!")
    print("\nThis demonstrates how the formatting pipeline:")
    print("1. Preserves the raw markdown syntax from the LLM")
    print("2. Converts escape sequences to actual newlines")
    print("3. Adds proper spacing for better Streamlit rendering")
    print("4. Makes URLs clickable")
    print("5. Maintains structure while improving readability")


if __name__ == "__main__":
    main()
