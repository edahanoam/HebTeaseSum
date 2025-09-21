1. Install requirements:
```pip install -r requirements.txt```

2. Get accessible Hebrew publication codes:
```python get_publications_codes.py --in='https://www.nli.org.il/en/newspapers/titles?lang=Hebrew&showaccessible=1' --out=pub_codes.txt```

3. Get all article urls:
```
python extract_all_section_ids.py --in=pub_codes.txt --out=sect_codes.txt
```

4. Download all news folders:
```
python get_sections.py --in=$ALL_SECT_CODES --out=$NEWS_FOLDER --timeout=1 --threads=$NUM_OF_THREADS --debug
```
