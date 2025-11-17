""" Usage:
    <file-name> --in_t=TEXT_OF_CANDIDATES --in_m=matching_if_files --n=NUMSAMPLES [--debug]

"""
import json
import csv
import random
from docopt import docopt
from pathlib import Path


def sample_and_tag(input_file_all_texts, input_file_candidates, num_sample, to_pass=100):
    """Samples `num_sample` main docs and writes their text + related docs to a structured CSV that works in Google Sheets."""

    documents = []
    doc_ids = []

    with open(input_file_all_texts, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text = data.get("LogicalSectionTextHTML", "")
                if text:
                    documents.append(text)
                    doc_ids.append(data.get("LogicalSectionID"))


    dic_related ={}
    dic_unrelated ={}

    counter = 0
    passed_prior = False
    with open(input_file_candidates, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if not passed_prior:
                if counter<=to_pass:
                    counter+=1
                    continue
                else:
                    counter=0
                    passed_prior = True

            else:
                if counter==num_sample:
                    break


            # if counter==num_sample:
            #     break

            if line.strip():
                data = json.loads(line)
                main_doc, related_docs = list(data.items())[0]  # Extract key and list

                # Ensure main document exists in our extracted texts
                if main_doc not in doc_ids:
                    continue
                counter+=1

                main_index = doc_ids.index(main_doc)
                main_text = documents[main_index]

                # Collect related documents that exist in doc_ids
                related_texts = []
                related_doc_ids = []
                unrelated_texts = []
                unrelated_doc_ids = []

                for related_doc in related_docs:
                    if related_doc in doc_ids:
                        print(related_doc)
                        print("----------------------------")
                        print("The Summary:")
                        print(main_text)

                        related = input(f"is this related \n {documents[doc_ids.index(related_doc)]}")
                        if related=='y':
                            related_texts.append(documents[doc_ids.index(related_doc)])
                            related_doc_ids.append(related_doc)
                        elif related=='n':
                            unrelated_texts.append(documents[doc_ids.index(related_doc)])
                            unrelated_doc_ids.append(related_doc)
                        else:
                            continue

                if related_docs:
                    dic_related[main_doc]=related_doc_ids
                    dic_unrelated[main_doc]=unrelated_doc_ids

    with open('SecondTestrelatedSample.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dic_related.items():
            writer.writerow([key, value])

    with open('SecondTestunrelatedSample.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dic_unrelated.items():
            writer.writerow([key, value])

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Filter JSONL 1.0')
    input_file_candidates = Path(arguments['--in_m'])
    input_file_all_texts = Path(arguments['--in_t'])
    num_sample=int(arguments['--n'])
    sample_and_tag(input_file_all_texts,input_file_candidates,num_sample)
