""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_DIR --timeout=MINS  [--debug]
"""
import json
import logging
from tqdm import tqdm
from docopt import docopt
from pathlib import Path
import requests
import time
import xml.etree.ElementTree as ET

build_issue_url = lambda issue_id: f"https://www.nli.org.il/en/newspapers/?a=d&d={issue_id}&f=XML"


def fetch_webpage_content(url):
    """
    get the contents of a url
    """
    try:
        # Send a GET request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

        response = requests.get(url, headers=headers)
        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Return the content of the webpage
            return response.content
        else:
            err = f"Failed to fetch webpage. HTTP status code: {response.status_code}"
            logging.error(err)
            raise Exception(err)

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred: {e}")
        raise e


def sleep_with_pbar(sleep_secs):
    for _ in tqdm(range(sleep_secs)):
        time.sleep(1)


def build_dataset(inp_fn,out_dir):
    out_fn = inp_fn.with_name(inp_fn.stem + "_with_text.jsonl")
    with inp_fn.open("r", encoding="utf-8") as f_in, \
         out_fn.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                summary_id = obj["summary_id"]
                related_ids = obj["text_ids"]

                sum_text= get_logical_section_text(out_dir / f"{summary_id}.xml")
                articles =[]
                for id in related_ids:
                    articles.append(get_logical_section_text(out_dir / f"{id}.xml"))

                # build output line
                new_obj = {
                    "summary_id": summary_id,
                    "summary_text": sum_text,
                    "related_ids": related_ids,
                    "related_texts": articles
                }

                f_out.write(json.dumps(new_obj, ensure_ascii=False) + "\n")


def get_logical_section_text(xml_path):
    """
    Given a path to xml file, return the section text
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    el = root.find(".//LogicalSectionTextHTML")
    return "".join(el.itertext()).strip() if el is not None else ""



def handle_download(inp_fn,out_dir,timeout_mins):
    """
    download all needed xml not yet in the dir
    :param inp_fn: jsonl with the needed sections
    :param out_dir: dir to download xml to
    :param timeout_mins: wait in case of error
    """
    to_download = get_xml_list(inp_fn,out_dir)
    for id in to_download:
        download_one(id, Path(out_dir), timeout_mins)


def download_one(sect_name,out_dir,timeout_mins):
    """
    gets a name of a single section and downloads it
    :param sect_name: name of a section
    :param timeout_mins: num of minutes to wait before re-trying
    """
    downloaded = False

    #while loop to repeatedly fetch the article until success:
    while not downloaded:
        sect_url = build_issue_url(sect_name)

        try:
            sect_cont = fetch_webpage_content(sect_url)
        except Exception as e:
            # timeouts occur from time to time, sleep to recover
            logging.debug(
                f"Error fetching: {e}\n Sleeping for {timeout_mins}...")
            sleep_with_pbar(timeout_mins)
            continue

        # managed to download, save to file and update progress
        downloaded = True
        out_fn = out_dir / f"{sect_name}.xml"

        try:
            with open(out_fn, "w", encoding="utf8") as fout:
                fout.write(sect_cont.decode("utf8"))
        except IOError:
            logging.debug(f"Io problem in section {sect_name}")
            with open("files_with_errors.txt", "w", encoding="utf8") as feror:
                feror.write(f"{sect_name}")


def get_all_to_download(inp_fn):
    summary_ids = set()
    article_ids = set()

    with open(inp_fn, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            summary_ids.add(obj["summary_id"])
            for tid in obj["text_ids"]:
                article_ids.add(tid)
    
    all= list(summary_ids | article_ids)
    return all

    


def get_xml_list(inp_fn,out_dir):
    all_ids =get_all_to_download(inp_fn)
    downloaded =[p.stem for p in out_dir.glob("*.xml")]
    return [x for x in all_ids if x not in downloaded]


if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = Path(args["--in"]) if args["--in"] else None
    out_dir = Path(args["--out"]) if args["--out"] else None
    timeout_mins = int(args["--timeout"]) * 60 if args["--timeout"] else None

    #Fetch xmls
    handle_download(inp_fn,out_dir,timeout_mins)
    #build the dataset
    build_dataset(inp_fn,out_dir)


