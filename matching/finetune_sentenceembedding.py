""" Usage:
    <file-name> --in_t=TEXT_OF_CANDIDATES --r=RELATED_FILES --un_r=UN_RELATED_FILES --n=NUM_WORDS [--debug]

"""

from sentence_transformers import SentenceTransformer, losses, InputExample, util

from torch.utils.data import DataLoader

from utils import load_text_and_ids, first_n_words
import pandas as pd
import os
import json
from docopt import docopt
from pathlib import Path
import pandas as pd
import numpy as np
import csv
import ast  # To safely convert string list to actual list


def finetune_sentencetransforemer(training_examples):
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2", cache_folder="/tmp/huggingface_cache")
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=4)

    # Define Cosine Similarity Loss
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=10
    )

    # Save the fine-tuned model
    model.save("setfit_finetuned_cosine")


def create_pairs(related_file,documents, doc_ids,n):
    #loading the tagged data texts and create pairs [summary, text] or [summary, unrelated texts]
    all_pairs =[]

    with open(related_file, "r", encoding="utf-8") as f_in:
        reader = csv.reader(f_in)  # Read as a list (no headers)

        for row in reader:
            if len(row) < 2:  # Ensure row has at least two elements
                continue

            main_doc = row[0].strip()  # First column = main document ID
            related_docs = ast.literal_eval(row[1].strip())  # Second column = list of related document IDs

            # Ensure main document exists in our extracted texts
            if main_doc not in doc_ids:
                continue

            main_index = doc_ids.index(main_doc)
            main_text = documents[main_index]

            # Iterate through related docs
            for related_doc in related_docs:
                related_doc = related_doc.strip()
                if related_doc in doc_ids:
                    cur_pair = [main_text, first_n_words(documents[doc_ids.index(related_doc)],n)]
                    all_pairs.append(cur_pair)

    return all_pairs




def create_training_data(related_file,unrelated_file,input_file_all_texts,n):

    documents, doc_ids = load_text_and_ids(input_file_all_texts)
    pos_pairs = create_pairs(related_file,documents, doc_ids,n)
    neg_pairs = create_pairs(unrelated_file,documents, doc_ids,n)

    print(len(pos_pairs)+len(neg_pairs))

    training_examples = []
    for pair in pos_pairs:
        training_examples.append(InputExample(texts=pair, label=1.0))
    for pair in neg_pairs:
        training_examples.append(InputExample(texts=pair, label=0.0))

    return training_examples





if __name__ == '__main__':
    # this program supposed to get a jsonl with the candidates and the
    arguments = docopt(__doc__, version='Filter JSONL 1.0')
    input_file_all_texts = Path(arguments['--in_t'])
    n = int(arguments['--n'])
    related_file = Path(arguments['--r'])
    unrelated_file = Path(arguments['--un_r'])

    training_examples= create_training_data(related_file,unrelated_file,input_file_all_texts,n)
    finetune_sentencetransforemer(training_examples)