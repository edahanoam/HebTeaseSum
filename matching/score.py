""" Usage:
    <file-name> --in_t=TEXT_OF_CANDIDATES --in_m=matching_if_files --method=METHOD_NUM --out=OUTFILE[--debug]

"""

import os
import json
from docopt import docopt
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, losses, InputExample,util
from torch.utils.data import DataLoader

from utils import load_text_and_ids, first_n_words
import csv
import ast  # To safely convert string list to actual list

import torch


methods= {0:'tf-idf',1:'bertscore',2:'sentencetransformers'}

def hebrew_tokenizer(text):
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Split by whitespace
    tokens = text.split()
    return tokens

def tfidf(input_file_all_texts):
    documents,doc_ids=load_text_and_ids(input_file_all_texts)


    vectorizer = TfidfVectorizer(tokenizer=hebrew_tokenizer, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(tfidf_matrix.shape)
    return tfidf_matrix, doc_ids


def tf_idf_cosine(tfidf_matrix, doc_ids,input_file_candidates,output_jsonl):
    # tfidf_matrix = np.array(tfidf_matrix)
    with open(input_file_candidates, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.strip():
                data = json.loads(line)
                main_doc, related_docs = list(data.items())[0]  # Extract key and list

                # Find index of the main document in doc_ids
                if main_doc not in doc_ids:
                    continue  # Skip if the document is missing

                main_index = doc_ids.index(main_doc)
                if not main_index:
                    continue

                main_vector = tfidf_matrix[main_index].reshape(1, -1)  # Get main doc vector

                # Compute cosine similarity for related documents
                results = []
                for related_doc in related_docs:
                    if related_doc in doc_ids:
                        related_index = doc_ids.index(related_doc)
                        related_vector = tfidf_matrix[related_index].reshape(1, -1)

                        # Compute cosine similarity
                        similarity = cosine_similarity(main_vector, related_vector)[0][0]
                        results.append((related_doc, round(similarity, 4)))

                # Save results as JSONL
                f_out.write(json.dumps({main_doc: results}, ensure_ascii=False) + "\n")

    print(f"Cosine similarity computed and saved to {output_jsonl}")





def bertscore_similarity(input_file_all_texts,input_file_candidates, output_jsonl):
    """Compute similarity using BERTScore F1 instead of cosine similarity."""
    documents,doc_ids=load_text_and_ids(input_file_all_texts)

    #counter = 0
    with open(input_file_candidates, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            #counter += 1

            # if counter == 10:
            #     break

            if line.strip():
                data = json.loads(line)
                main_doc, related_docs = list(data.items())[0]  # Extract key and list

                # Ensure main document exists in our extracted texts
                if main_doc not in doc_ids:
                    continue
                main_index = doc_ids.index(main_doc)
                main_text = documents[main_index]

                # Collect related documents that exist in doc_ids
                related_texts = []
                related_doc_ids = []
                for related_doc in related_docs:
                    if related_doc in doc_ids:
                        related_texts.append(documents[doc_ids.index(related_doc)])
                        related_doc_ids.append(related_doc)

                if not related_texts:
                    continue  # Skip if there are no valid related texts

                P, R, F1 = score(
                    [main_text] * len(related_texts),
                    related_texts,
                    model_type="bert-base-multilingual-cased",  # ✅ Supported model
                    rescale_with_baseline=False,  # ✅ Allowed for this model
                    lang="he"  # ✅ Required when using a supported model with baseline
                )

                # Store results with F1 score
                results = [(related_doc_ids[i], round(F1[i].item(), 4)) for i in range(len(related_doc_ids))]

                # Save results to output JSONL
                f_out.write(json.dumps({main_doc: results}, ensure_ascii=False) + "\n")

    print(f"BERTScore similarity computed and saved to {output_jsonl}")


def embedding_similarity(input_file_all_texts, input_file_candidates, output_jsonl,
                         model_path="setfit_finetuned_cosine"):
    """Compute similarity using fine-tuned SentenceTransformer embeddings and cosine similarity."""

    # Load fine-tuned model
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2", cache_folder="/tmp/huggingface_cache")

    documents,doc_ids=load_text_and_ids(input_file_all_texts)

    # Compute embeddings for all documents
    document_embeddings = model.encode(documents, convert_to_tensor=True)

    with open(input_file_candidates, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.strip():
                data = json.loads(line)
                main_doc, related_docs = list(data.items())[0]  # Extract key and list

                # Ensure main document exists
                if main_doc not in doc_ids:
                    continue
                main_index = doc_ids.index(main_doc)
                main_embedding = document_embeddings[main_index]

                # Collect related document embeddings
                related_texts = []
                related_doc_ids = []
                related_embeddings = []

                for related_doc in related_docs:
                    if related_doc in doc_ids:
                        related_index = doc_ids.index(related_doc)
                        related_texts.append(documents[related_index])
                        related_doc_ids.append(related_doc)
                        related_embeddings.append(document_embeddings[related_index])

                if not related_embeddings:
                    continue  # Skip if no valid related texts

                # Convert list of tensors to a single tensor
                related_embeddings = torch.stack(related_embeddings)

                # Compute cosine similarity
                similarity_scores = util.pytorch_cos_sim(main_embedding, related_embeddings).squeeze(0)

                # Store results with similarity score
                results = [(related_doc_ids[i], round(similarity_scores[i].item(), 4)) for i in
                           range(len(related_doc_ids))]

                # Save results to output JSONL
                f_out.write(json.dumps({main_doc: results}, ensure_ascii=False) + "\n")

    print(f"Embedding-based similarity computed and saved to {output_jsonl}")




if __name__ == '__main__':
    #this program supposed to get a jsonl with the candidates and the
    arguments = docopt(__doc__, version='Filter JSONL 1.0')
    input_file_candidates = Path(arguments['--in_m'])
    #this is file after we kept only relevan rows
    input_file_all_texts = Path(arguments['--in_t'])
    method=methods[int(arguments['--method'])]
    output_file= Path(arguments['--out'])

    if method=='tf-idf':
        tfidf_matrix, doc_ids=  tfidf(input_file_all_texts)
        tf_idf_cosine(tfidf_matrix, doc_ids,input_file_candidates,output_file)

    elif method=='bertscore':
        bertscore_similarity(input_file_all_texts,input_file_candidates, output_file)

    elif method=='sentencetransformers':
        embedding_similarity(input_file_all_texts,input_file_candidates, output_file)




