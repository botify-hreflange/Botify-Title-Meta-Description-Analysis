# SEO Content Analysis Tool

A Python tool for analyzing SEO content patterns across web pages, focusing on titles and meta descriptions. This tool helps identify duplicate content, analyze n-gram patterns, and understand content patterns across different page types.

## Example of the Input File
| Full URL | pagetype | Title | Meta Description |
|----------|----------|--------|------------------|
| https://www.example.com/page1 | search/features-facet | Sample Title | Sample Description |
| https://www.example.com/page2 | search/rating-facet | Another Title | Another Description |

## Features

- **Content Analysis**
  - Title and meta description duplication detection
  - N-gram frequency analysis (2-gram, 3-gram)
  - Page type-specific content patterns
  - Summary reports by page type

- **Robust Data Processing**
  - Large file handling through chunked processing
  - Progress tracking with tqdm
  - Organized logging in dedicated logs directory
  - Duplicate removal and data cleaning

## Prerequisites

```bash
pip install pandas nltk tqdm
```

You'll also need the NLTK data for tokenization and stopwords. The script will automatically download these if they're not present.

## Usage

### Basic Usage

```bash
python3 app.py input_file.csv
```

### Advanced Usage

```bash
python3 app.py input_file.csv --delimiter ',' --chunk-size 200000
```

### Arguments

- `input_file`: Path to your input CSV/TSV file
- `--delimiter`: File delimiter (default: tab)
- `--chunk-size`: Size of chunks for processing large files (default: 100000)

## Input File Requirements

Your input file should contain the following columns:
- `Full URL`
- `pagetype`
- `Title`
- `Meta Description`

If your columns have different names, the tool will prompt you to map them during execution.

## Output Files

The tool generates analysis files in two directories:

### Analysis Results (`url_analysis_results/`)
- `title_duplicates_summary.csv`
- `title_pagetype_summary.csv`
- `meta_description_duplicates_summary.csv`
- `meta_description_pagetype_summary.csv`
- `title_2gram_analysis.csv`
- `title_3gram_analysis.csv`
- `meta_description_2gram_analysis.csv`
- `meta_description_3gram_analysis.csv`
- `analysis_summary.txt`

### Logs (`logs/`)
- Timestamped log files: `seo_analysis_YYYYMMDD_HHMMSS.log`

## Example Outputs

### Page Type Analysis
| Pagetype | Total_URLs | Duplicate_Titles | Duplication_Rate |
|----------|------------|------------------|------------------|
| account/cart | 25 | 0 | 0 |
| account/sign-in | 2 | 1 | 50 |
| category/main | 3485 | 4 | 0.11 |
| path-pages | 44247 | 11 | 0.02 |
| pdp/Movies&TV/discs | 228 | 3 | 1.32 |

### N-gram Analysis
| pagetype | ngram | frequency |
|----------|--------|-----------|
| account/cart | best buy | 25 |
| account/cart | gift card | 4 |
| account/cart | digital com | 4 |
| account/cart | qa best | 4 |
| account/cart | cart mount | 2 |

### Duplication Overview
| Title | Duplicate_Count | Pagetypes |
|-------|----------------|------------|
| Customer Reviews: 2-Year Accidental Geek Squad Protection - Best Buy | 771 | reviews/pagination, reviews/main |
| What is the maximum weight capacity? ‚Äì Q&A ‚Äì Best Buy | 597 | questions/individual |
| Is the case compatible with wireless charging? ‚Äì Q&A ‚Äì Best Buy | 559 | questions/individual |

## Error Handling

The tool includes robust error handling for:
- Missing input files
- Invalid file formats
- Missing required columns
- Data processing errors
- User interruptions

## Performance Considerations

- Uses chunked processing for handling large files
- Implements memory-efficient data structures
- Shows progress bars for long-running operations
- Supports interruption and graceful shutdown

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Specify your license here]

## Author

[Your name/organization here]