## Quick Start🚀

### Download the data

1. Install requirements:
```pip install -r requirements.txt```

2. Choose the waned portion of HebTeaseSum from the metadata/ directory
- **singleDoc.jsonl** All summaries corresponding to a single article  
- **multiDoc.jsonl** Summaries created from multiple related articles  
- **singleDoc_over50.jsonl** Summaries with at least 50 words, rated as the highest-quality summaries by human annotators  
- **multiDoc_over50.jsonl** Summaries with at least 50 words, rated as the highest-quality summaries by human annotators  

3. Download the raw XML and build the dataset
```
python datacollector/get_dataset.py --in=$JSONl_FILE --out=$FOLDER_FOR_XMLS --timeout=1 --debug
```

