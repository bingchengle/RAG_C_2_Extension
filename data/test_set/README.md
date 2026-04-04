# Test Set Dataset

This directory contains a small test set with 5 questions and corresponding company reports.
You can use this to study example questions, reports, and system outputs.

## Contents
- `questions.json`: Contains 5 test questions
- `subset.csv`: Metadata about the test documents
- `pdf_reports/`: Directory containing the original PDF reports
- `answers_max_nst_o3m.json`: Sample answers from the winning system
- `answers_max_nst_o3m_debug.json`: Detailed debug output for the sample answers

## Running the System

To run the system on this test set:

1. Unpack the required zip files in this directory:
   - `databases.zip` - Contains processed data needed for the pipeline
   - `debug_data.zip` (optional) - Contains intermediate outputs for debugging

2. Follow the setup and usage instructions in the main README.md at the root of this repository 