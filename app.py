#!/usr/bin/env python3
"""
SEO Content Analysis Tool
------------------------
Analyzes titles and meta descriptions for patterns, duplicates, 
and n-gram frequencies across different page types.
"""

import pandas as pd
import os
from pathlib import Path
import sys
import argparse
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from tqdm import tqdm
import logging
from datetime import datetime

def initialize_nltk():
    """Initialize NLTK by downloading required resources."""
    try:
        import nltk
        nltk.data.path.append(str(Path.home() / 'nltk_data'))  # Add user's home directory to NLTK path
        
        # Dictionary of required resources and their package types
        required_resources = {
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords',
            'punkt_tab': 'tokenizers/punkt'  # Added this resource
        }
        
        for resource, package_type in required_resources.items():
            try:
                nltk.data.find(f'{package_type}')
            except LookupError:
                print(f"Downloading required NLTK package: {resource}")
                nltk.download(resource, quiet=True)
                
        logger.info("NLTK initialization complete")
        
        # Verify all resources are available
        for resource, package_type in required_resources.items():
            nltk.data.find(f'{package_type}')
            
    except Exception as e:
        logger.error(f"Error initializing NLTK: {e}")
        sys.exit(1)

def setup_logging():
    """Set up logging configuration with a logs directory."""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    log_file = logs_dir / f'seo_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def peek_csv(file_path, delimiter='\t'):
    """Peek at the CSV file to check headers and first few rows."""
    try:
        first_chunk = pd.read_csv(file_path, sep=delimiter, nrows=5)
        return first_chunk.columns.tolist(), first_chunk.head()
    except Exception as e:
        logger.error(f"Error peeking at file: {e}")
        return None, None

def preprocess_text(text):
    """Clean and tokenize text for n-gram analysis."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Simple word tokenization as backup if NLTK fails
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        # Fallback to simple splitting
        tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words]

def extract_ngrams(tokens, n):
    """Extract n-grams from tokenized text."""
    n_grams = list(ngrams(tokens, n))
    return [' '.join(gram) for gram in n_grams]

def read_csv_in_chunks(file_path, delimiter='\t', chunk_size=100000):
    """Read CSV file in chunks with proper header handling and duplicate removal."""
    chunks = []
    
    try:
        file_size = os.path.getsize(file_path)
        total_chunks = file_size // (chunk_size * 100) + 1
        
        header_df = pd.read_csv(file_path, sep=delimiter, nrows=0)
        logger.info("Detected columns:")
        for col in header_df.columns:
            logger.info(f"- {col}")

        with tqdm(desc="Reading file", total=total_chunks) as progress_bar:
            for chunk in pd.read_csv(file_path, sep=delimiter, chunksize=chunk_size):
                chunk = chunk.drop_duplicates()
                chunks.append(chunk)
                progress_bar.update(1)
            
        df = pd.concat(chunks, ignore_index=True)
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows:,} complete duplicate rows")
            
        return df
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return None

def analyze_content(df, content_type, output_dir):
    """Analyze content (titles or meta descriptions) across page types."""
    logger.info(f"\nAnalyzing {content_type.lower()}s...")
    
    # Basic duplicate analysis
    duplicates = df.groupby(content_type).agg({
        'Full URL': 'count',
        'pagetype': lambda x: ', '.join(set(x))
    }).reset_index()
    duplicates.columns = [content_type, 'Duplicate_Count', 'Pagetypes']
    duplicates = duplicates.sort_values('Duplicate_Count', ascending=False)
    
    # Pagetype breakdown
    pagetype_summary = df.groupby('pagetype').agg({
        'Full URL': 'count',
        content_type: lambda x: x.duplicated().sum()
    }).reset_index()
    pagetype_summary.columns = ['Pagetype', 'Total_URLs', f'Duplicate_{content_type}s']
    pagetype_summary['Duplication_Rate'] = (pagetype_summary[f'Duplicate_{content_type}s'] / 
                                          pagetype_summary['Total_URLs'] * 100).round(2)
    
    # Export summary results
    content_type_clean = content_type.lower().replace(' ', '_')
    duplicates.to_csv(output_dir / f'{content_type_clean}_duplicates_summary.csv', index=False)
    pagetype_summary.to_csv(output_dir / f'{content_type_clean}_pagetype_summary.csv', index=False)
    
    # N-gram analysis
    logger.info(f"Analyzing n-grams for {content_type.lower()}s...")
    
    # Ensure NLTK resources are downloaded
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')

    # Store all n-grams across all pagetypes
    all_ngram_data = []
    
    # Analyze bigrams and trigrams
    for n in [2, 3]:
        logger.info(f"Analyzing {n}-grams...")
        
        # Process n-grams for all pagetypes
        for pagetype in tqdm(df['pagetype'].unique(), desc=f"{n}-gram analysis"):
            content = df[df['pagetype'] == pagetype][content_type]
            
            all_ngrams = []
            for text in content:
                tokens = preprocess_text(text)
                if tokens:
                    text_ngrams = extract_ngrams(tokens, n)
                    all_ngrams.extend(text_ngrams)
            
            # Count n-grams for this pagetype
            ngram_counts = Counter(all_ngrams)
            
            # Filter for frequent n-grams and add to overall list
            for ngram, count in ngram_counts.items():
                if count > 1:  # Only include n-grams that appear more than once
                    all_ngram_data.append({
                        'n': n,
                        'pagetype': pagetype,
                        'ngram': ngram,
                        'frequency': count
                    })
        
        if all_ngram_data:
            # Convert to DataFrame
            ngram_df = pd.DataFrame([row for row in all_ngram_data if row['n'] == n])
            
            if not ngram_df.empty:
                # Sort first by frequency (descending) across all pagetypes, then by pagetype
                ngram_df = ngram_df.sort_values(
                    ['frequency', 'pagetype'],
                    ascending=[False, True]
                )
                
                # Drop the 'n' column before saving
                ngram_df = ngram_df.drop(columns=['n'])
                
                # Save to CSV
                output_file = output_dir / f'{content_type_clean}_{n}gram_analysis.csv'
                ngram_df.to_csv(output_file, index=False)
                logger.info(f"Wrote {len(ngram_df):,} {n}-grams to {output_file}")
    
    return duplicates, pagetype_summary

def analyze_seo_content(df, output_dir):
    """Main function to analyze SEO content."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Analyze titles and meta descriptions
    logger.info("\nAnalyzing titles...")
    title_duplicates, title_pagetype_summary = analyze_content(df, 'Title', output_dir)
    
    logger.info("\nAnalyzing meta descriptions...")
    meta_duplicates, meta_pagetype_summary = analyze_content(df, 'Meta Description', output_dir)
    
    # Generate summary report
    report_path = output_dir / 'analysis_summary.txt'
    with open(report_path, 'w') as f:
        f.write("SEO Content Analysis Summary Report\n")
        f.write("=================================\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        f.write("Overall Statistics\n")
        f.write("-----------------\n")
        f.write(f"Total URLs analyzed: {len(df):,}\n")
        f.write(f"Unique URLs: {df['Full URL'].nunique():,}\n")
        f.write(f"Total page types: {df['pagetype'].nunique():,}\n\n")
        
        # Title statistics
        f.write("Title Analysis\n")
        f.write("--------------\n")
        f.write(f"Total titles: {len(df):,}\n")
        f.write(f"Unique titles: {df['Title'].nunique():,}\n")
        duplicate_titles = len(df[df.duplicated(subset=['Title'], keep=False)])
        f.write(f"Duplicate titles: {duplicate_titles:,} ({(duplicate_titles/len(df)*100):.1f}%)\n\n")
        
        # Meta description statistics
        f.write("Meta Description Analysis\n")
        f.write("------------------------\n")
        f.write(f"Total meta descriptions: {len(df):,}\n")
        f.write(f"Unique meta descriptions: {df['Meta Description'].nunique():,}\n")
        duplicate_metas = len(df[df.duplicated(subset=['Meta Description'], keep=False)])
        f.write(f"Duplicate meta descriptions: {duplicate_metas:,} ({(duplicate_metas/len(df)*100):.1f}%)\n")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze SEO content including titles and meta descriptions'
    )
    parser.add_argument('input_file', help='Path to the input CSV/TSV file')
    parser.add_argument('--delimiter', default='\t', help='File delimiter (default: tab)')
    parser.add_argument('--chunk-size', type=int, default=100000, 
                       help='Chunk size for reading large files')
    
    args = parser.parse_args()
    
    # Set up logging
    global logger
    logger = setup_logging()

    # Initialize NLTK (add this line)
    initialize_nltk()
    
    # Check if file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    
    # Examine file structure
    logger.info("Examining file structure...")
    headers, sample_data = peek_csv(input_path, args.delimiter)
    
    if headers:
        logger.info("\nFound headers:")
        for header in headers:
            logger.info(f"- {header}")
        
        logger.info("\nSample data (first 5 rows):")
        logger.info(sample_data)
        
        logger.info("\nDo these columns match what you expect?")
        response = input("Continue with these columns? (y/n): ")
        
        if response.lower() != 'y':
            logger.info("\nPlease specify the correct column names for required fields:")
            title_col = input("Enter the column name for Title (or press Enter to skip): ")
            meta_col = input("Enter the column name for Meta Description (or press Enter to skip): ")
            url_col = input("Enter the column name for Full URL (or press Enter to skip): ")
            pagetype_col = input("Enter the column name for pagetype (or press Enter to skip): ")
            
            # Create column mapping
            column_mapping = {}
            if title_col: column_mapping['Title'] = title_col
            if meta_col: column_mapping['Meta Description'] = meta_col
            if url_col: column_mapping['Full URL'] = url_col
            if pagetype_col: column_mapping['pagetype'] = pagetype_col
        else:
            column_mapping = None
    else:
        logger.error("Could not detect file structure. Please check the file format and delimiter.")
        sys.exit(1)
    
    # Read the file
    logger.info(f"\nReading {args.input_file} in chunks...")
    df = read_csv_in_chunks(input_path, args.delimiter, args.chunk_size)
    
    if df is None:
        logger.error("Error reading the file.")
        sys.exit(1)
    
    # Apply column mapping if provided
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Verify required columns exist
    required_columns = {'Full URL', 'pagetype', 'Title', 'Meta Description'}
    available_columns = set(df.columns)
    missing_columns = required_columns - available_columns
    
    if missing_columns:
        logger.error(f"\nError: Missing required columns: {missing_columns}")
        logger.error("\nAvailable columns:")
        for col in available_columns:
            logger.error(f"- {col}")
        sys.exit(1)
    
    logger.info("\nSuccessfully loaded the file!")
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Total unique URLs: {df['Full URL'].nunique():,}")
    logger.info(f"Total page types: {df['pagetype'].nunique():,}")
    
    try:
        # Perform analysis
        output_dir = Path('url_analysis_results')
        analyze_seo_content(df, output_dir)
        
        logger.info("\nAnalysis complete! Results have been saved to the 'url_analysis_results' directory:")
        logger.info("\nSummary Reports:")
        logger.info("- title_duplicates_summary.csv")
        logger.info("- title_pagetype_summary.csv")
        logger.info("- meta_description_duplicates_summary.csv")
        logger.info("- meta_description_pagetype_summary.csv")
        
        logger.info("\nN-gram Analysis:")
        logger.info("- title_2gram_analysis.csv")
        logger.info("- title_3gram_analysis.csv")
        logger.info("- meta_description_2gram_analysis.csv")
        logger.info("- meta_description_3gram_analysis.csv")
        
        logger.info("\nOverall Summary:")
        logger.info("- analysis_summary.txt")
        
    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nScript interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)