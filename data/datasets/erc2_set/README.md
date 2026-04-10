# ERC2 Dataset

This directory contains the full RAG Challenge dataset with questions and corresponding company reports.
You can use this to study example questions, reports, and system outputs.

## Contents
- `questions.json`: Contains the competition questions
- `subset.csv`: Metadata about the test documents
- `subset.json`: Same metadata, but in JSON format
- `answers_1st_place_o3-mini.json`: Sample answers from the winning system using OpenAI's model
- `answers_1st_place_llama_70b.json`: Sample answers using Llama 70B model

## Running the System

To run the system on this dataset:

1. Download and unpack the required files in this directory:

   ### Required for Question Answering
   - `databases` ([google drive](https://drive.google.com/file/d/1mp-hYhMAit4rdi7RURuIsM33zbXq1nQJ/view?usp=sharing))
     - Contains all the processed data needed for running question answering pipeline

   ### Optional Files
   - `pdf_reports` ([google drive](https://drive.google.com/file/d/1MvcN_-KpI-9nS4hDFAcPxFU2lRmwMP7M/view?usp=sharing))
     - Needed if you want to run the PDF parsing pipeline from scratch
     - Or if you want to analyze the original documents

   - `debug_data` ([google drive](https://drive.google.com/file/d/13RT456tZVTAwPIsy8OndZ1EWASNCdfe3/view?usp=sharing))
     - Needed if you want to:
       - Debug specific pipeline stages
       - Run individual preprocessing steps
       - Study the system's intermediate outputs

2. Follow the setup and usage instructions in the main README.md at the root of this repository