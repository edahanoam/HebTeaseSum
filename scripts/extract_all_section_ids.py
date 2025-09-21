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

# Local imports
from fetch import build_title_url, extract_issu_ids, extract_sect_ids, build_issue_url

# ----


if __name__ == "__main__":

    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = Path(args["--in"])
    out_fn = Path(args["--out"])

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info(f"Input file: {inp_fn}, Output file: {out_fn}.")

    # Start computation
    pub_ids = open(inp_fn, encoding="utf8").read().split("\n")
    pub_ids = list(filter(lambda x: x.strip(), pub_ids))
    sect_ids = []

    with open(out_fn, "w", encoding="utf8") as fout:
        pbar = tqdm(pub_ids, desc=" titles", position=0)
        for pub_id in pbar:
            pbar.set_description(f"Parsing {pub_id}")
            pub_url = build_title_url(pub_id)
            issue_ids = list(reversed(extract_issu_ids(pub_url)))
            for issue_id in tqdm(issue_ids, desc=" issues", position=1, leave=False):
                issue_url = build_issue_url(issue_id)
                cur_sect_ids = extract_sect_ids(issue_url)
                sect_ids += cur_sect_ids
                fout.write("\n".join(cur_sect_ids) + "\n")

    num_of_sects = len(sect_ids)
    logging.info(f"Wrote {num_of_sects} section ids to {out_fn}")

    # End
    logging.info("DONE")
