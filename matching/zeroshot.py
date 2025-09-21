""" Usage:
    <file-name> --all=ALLFILE  --inf=INFERENCESOFAR --out=OUTFILE[--debug]

"""
import pandas as pd
from together import Together
from tqdm import tqdm
from utils import load_text_and_ids, first_n_words
import ast
import time
from docopt import docopt
from pathlib import Path
import os

PROMPT="Given the following text and summary, answer with 'Yes' if the text relates to the summary, and 'No' if it does not. Do not provide explanations. Only output 'Yes' or 'No'."
# model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

def wrap_text(text_file):
    all_pairs = pd.read_csv(text_file,header=None)
    all_pairs.columns = ['summary','text_candidate','classification']

  # Create a new column 'formatted_output' with the desired format
    all_pairs["formatted_output"] = all_pairs.apply(lambda row: f"""Summary: \n{row['summary']}\nText:\n{first_n_words(row['text_candidate'])}\nRelates: """, axis=1)
    return all_pairs

def wrap_text_from_df(df,summary_col_name='summary',text_candidate_name= 'text_candidate',cut_text = True):
  # Create a new column 'formatted_output' with the desired format
  # df.columns = ['summary', 'text_candidate', 'classification']
  if cut_text:
    df[text_candidate_name] = df[text_candidate_name].apply(lambda x: first_n_words(x,500))

  df["formatted_output"] = df.apply(lambda row: f"""Summary: \n{row[summary_col_name]}\nText:\n{row[text_candidate_name]}\nRelates: """, axis=1)
  return df

def add_texts(documents, ids,df_shuffled):

    main_ids=[]
    documents_ids=[]
    main_texts =[]
    documents_texts =[]
    classifications= []

    for index, row in df_shuffled.iterrows():
        main_text=documents[ids.index(row['summaryID'])]
        for doc in ast.literal_eval(row['text_candidateID'].strip()):
            main_texts.append(main_text)
            main_ids.append(row['summaryID'])
            documents_text = first_n_words(documents[ids.index(doc)])
            documents_texts.append(documents_text)
            documents_ids.append(doc)
            classifications.append(row['Class'])

    new_df=  pd.DataFrame({'summaryID': main_ids, 'summary': main_texts, 'text_candidateID':documents_ids , 'text_candidate': documents_texts, 'classifications':classifications})

    print(new_df.head)
    return new_df


def infer_togetherAI_save_results(df,inferences_so_far,outfile="llama_test.csv"):
    if not os.path.exists(inferences_so_far):
        with open(inferences_so_far, "w") as file:
            file.write("")  # Create an empty file
            inferences=[]
    else:

        with open(inferences_so_far, "r", encoding="utf-8") as file:
            inferences = file.readlines()
            # Remove newline characters
            inferences = [line.strip() for line in inferences]

    start_row=len(inferences)
    client = Together()
    print(df.shape)


    for i, row in tqdm(df.iloc[start_row:].iterrows(), total=len(df) - start_row):
        try:
            current_prompt = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": row['formatted_output']}
                ]
            response = client.chat.completions.create(
            model=model,
            messages=current_prompt,
            )
            inferences.append(response.choices[0].message.content)
        except Exception as e:
            print(type(e).__name__)
            print(i)
            with open(inferences_so_far, "w") as file:
                file.writelines(f"{item}\n" for item in inferences)
            return

    with open(inferences_so_far, "w") as file:
        file.writelines(f"{item}\n" for item in inferences)

    df['inferences2'] = inferences
    df.to_csv(outfile, index=False)


def infer_togetherAI(df,outfile="llama_test.csv"):

    client = Together()
    #taking first 3 rows for quic inference, remove later
    # df = df.head(1)
    # print(df)
    inferences = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        cur=False #we didn't get inference for this example
        while not cur:
            try:
                current_prompt = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": row['formatted_output']}
                    ]
                response = client.chat.completions.create(
                model=model,
                messages=current_prompt,
                )
                #inferences.append(response.choices[0].message.content)
                inferences.append(response.choices[0].message.content)
                cur=True
            except:
                print("Waiting for an hour...")
                time.sleep(3600)
                print("continue...")



    df['inferences'] = inferences
    df.to_csv(outfile, index=False)


def run_on_test_files():
    path_for_texts= 'hadashot_only_relevant.jsonl'
    documents, ids=load_text_and_ids(path_for_texts)
    df1 = pd.read_csv('TestrelatedSample.csv')
    df1.columns = ['summaryID', 'text_candidateID']
    df1['Class'] = 1
    df2 = pd.read_csv('TestunrelatedSample.csv')
    df2.columns = ['summaryID', 'text_candidateID']
    df2['Class'] = 0
    # Combine the DataFrames
    df_combined = pd.concat([df1, df2], ignore_index=True)
    # Shuffle the rows
    df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index()
    print(df_shuffled.columns)
    df = add_texts(documents, ids,df_shuffled)
    print(df.shape)
    df= wrap_text_from_df(df)
    cur_time =time.time()
    infer_togetherAI(df)
    print(f"time it took for 300 pairs is {time.time()-cur_time} seconds")



def run_on_all_data(input_file_candidates,inferences_so_far,outfile,cut_text = True):
    df = pd.read_csv(input_file_candidates)
    print(df.shape)
    df = wrap_text_from_df(df,'main_text','related_text',cut_text)
    print(df.shape)

    #infer_togetherAI(df,outfile)
    infer_togetherAI_save_results(df,inferences_so_far,outfile)



if __name__=="__main__":
    # run_on_test_files()
    arguments = docopt(__doc__, version='Filter JSONL 1.0')
    #this is the the csv file containing all the main texts and the candidate texts and their ids
    input_file_candidates = Path(arguments['--all'])
    outfile = Path(arguments['--out'])
    inferences_so_far = Path(arguments['--inf'])
    #run_on_all_data(input_file_candidates,inferences_so_far,outfile)

    # df =pd.read_csv(input_file_candidates)
    # print(df.head(10))

    #for now:
    run_on_all_data(input_file_candidates,inferences_so_far,outfile,True)
