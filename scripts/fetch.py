""" Usage:
    <file-name> [--in=INPUT_FILE] [--out=OUTPUT_FILE] [--debug]

Options:
  --help                           Show this message and exit
  -i INPUT_FILE --in=INPUT_FILE    Input file
                                   [default: infile.tmp]
  -o INPUT_FILE --out=OUTPUT_FILE  Input file
                                   [default: outfile.tmp]
  --debug                          Whether to debug
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
import re

# Local imports


# ----

extract_page_ids = lambda issue_url: extract_ids(issue_url, "PageID")
extract_sect_ids = lambda issue_url: extract_ids(issue_url, "LogicalSectionID")
extract_issu_ids = lambda title_url: extract_ids(title_url, "DocumentID")

build_issue_url = lambda issue_id: f"https://www.nli.org.il/en/newspapers/?a=d&d={issue_id}&f=XML"
build_title_url = lambda title_id: f"https://www.nli.org.il/en/newspapers/?a=cl&cl=CL1&sp={title_id}&f=XML"


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


def extract_ids(issue_url, tag_name):
    """
    get a list of element ids (articles or pages, I think?)
    from an issue url
    """
    issue_xml = fetch_webpage_content(issue_url)
    soup = BeautifulSoup(issue_xml, "xml")
    content_ids = [elem.text for elem in soup.find_all(tag_name)]
    return content_ids


def fetch_title_codes(url):
    """
    returns a list of title codes for publications which can then
    instansiate new roots crawling.
    """
    pattern = re.compile(r"/en/newspapers/([\w]+)\?")
    titles = []

    soup = BeautifulSoup(fetch_webpage_content(url), "lxml")
    # each publication seems to have such div
    title_sections = soup.find_all("div", {"class": "nli-title-browser-row",
                                           "data-language": "Hebrew",
                                           "data-accessible": "1"})

    # and a single url link in it, which contains the publication code
    for title_section in title_sections:
        for link in title_section.find_all("a"):
            href_txt = link.attrs["href"]
            m = pattern.search(href_txt)
            if m:
                titles.append(m.group(1))

    return titles


def extract_issue_content(issue_url, extract_func):
    """
    follow through an issue ids to get all of its content
    returns a list of xmls, extract_func is either extract_sect_ids
    or extract_page_ids
    """
    sect_ids = extract_func(issue_url)
    sect_xmls = []
    for sect_id in tqdm(sect_ids):
        sect_xml = fetch_webpage_content(build_issue_url(sect_id))
        sect_xmls.append((sect_id, sect_xml))
    return sect_xmls


def write_issue_content_to_file(issue_content, output_folder):
    """
    write issue xml contents to a target folder, each in a different file
    """
    base_folder = Path(output_folder)
    for (issue_id, xml) in issue_content:
        out_fn = base_folder / f"{issue_id}.xml"
        logging.debug(f"writing to {out_fn}")
        with open(out_fn, "w", encoding="utf8") as fout:
            fout.write(xml.decode("utf8"))


if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_url = args["--in"]
    out_fn = Path(args["--out"])

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info(f"Input file: {inp_url}, Output file: {out_fn}.")

    # Start computation

    ls = fetch_title_codes(inp_url)

    # issue_ids = extract_issu_ids(inp_url)
    # issue_content = extract_issue_content(inp_url, extract_sect_ids)
    # write_issue_content_to_file(issue_content, out_fn)

    # End
    logging.info("DONE")
