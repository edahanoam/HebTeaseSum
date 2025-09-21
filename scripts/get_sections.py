""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE --timeout=MINS --threads=THREADS --lst=PATH_TO_LIST_FILE [--debug]

"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from pathlib import Path
from tqdm import tqdm
import json
from glob import glob
import time
import threading
import pandas as pd
import re
import os

# Local imports
from fetch import fetch_webpage_content, build_issue_url
#----

NUM_OF_FILES_PER_SUBDIR = 50000

def sleep_with_pbar(sleep_secs):
    """
    like sleep, but with a progress bar
    """
    for _ in tqdm(range(sleep_secs)):
        time.sleep(1)


def find_path_for_pub(out_dir, pub):
    # get the path pub directory
    pub_dir = os.path.join(out_dir, pub)

    # inside the pub dir there should be numbered dir
    subfolders = [f.path for f in os.scandir(pub_dir) if f.is_dir()]
    print(subfolders)

    #if we already downloaded sections from this pub
    if subfolders:
        numeric_parts = [int(subfolder.split('/')[-1]) for subfolder in subfolders]
        sub_dir_so_far = max(numeric_parts)
        print(sub_dir_so_far)
        # this directory will be completed now:
        destination_dir = os.path.join(pub_dir, str(sub_dir_so_far).zfill(3))
        print(destination_dir)
        # update counter to how many files in dir
        counter_in_dir_so_far = len(os.listdir(destination_dir))
    else:
        sub_dir_so_far = 0
        counter_in_dir_so_far = 0
        # create sub directory
        destination_dir = os.path.join(dir, str(sub_dir_so_far).zfill(3))
        if not os.path.exists(destination_dir):
            os.mkdir(destination_dir)

    return pub_dir,destination_dir,sub_dir_so_far,counter_in_dir_so_far



def update_lst_file(lst_file, name):
    with open(lst_file, 'a') as file:
        file.write('\n' + name + '.xml')

def manage_downloads_by_pub(out_dir, timeout_mins, sect_to_download, num_to_download,path_to_list):
    """"
    return list of secrtions downloaded to append later
    """
    #first check there is somethong to download
    if not sect_to_download:
        return

    sect_ind = 0

    # Regex to match letters up until the first digit which is the pub name
    pattern = r'^[A-Za-z]+(?=\d)'

    #get the name of the first pub to download, which is the pub of first sec in the list
    match = re.search(pattern, sect_to_download[sect_ind])
    #find the first section name (in case the first one had an error)
    while not match and sect_ind<num_to_download-1:
        sect_ind+=1
        match = re.search(pattern, sect_to_download[sect_ind])

    pub=match.group(0)
    #print(current_pub)

    #finding the sub dir of pub, how many files in the directory (so that i dont reacht the 50000 limit and how many sub
    #dirs we have so far in case we want to open a new one
    pub_dir, destination_dir, sub_dir_so_far, counter_in_dir_so_far = find_path_for_pub(out_dir,pub)

    while sect_ind<num_to_download:
        cur_sec= sect_to_download[sect_ind]
        match = re.search(pattern, cur_sec)
        if not match:
            continue
        cur_pub = match.group(0)
        if cur_pub!=pub:
            pub_dir, destination_dir, sub_dir_so_far, counter_in_dir_so_far = find_path_for_pub(out_dir, cur_pub)
            pub=cur_pub

        download_one(sect_to_download[sect_ind],Path(destination_dir), timeout_mins)
        update_lst_file(path_to_list, sect_to_download[sect_ind])

        sect_ind+=1
        counter_in_dir_so_far+=1

        # open a new sub dir if we reached limit
        if counter_in_dir_so_far>=NUM_OF_FILES_PER_SUBDIR:
            sub_dir_so_far+=1
            # create sub dirsctory
            logging.debug(f"CREATE new sub directory ")

            destination_dir = os.path.join(pub_dir, str(sub_dir_so_far).zfill(3))
            if not os.path.exists(destination_dir):
                os.mkdir(destination_dir)
            counter_in_dir_so_far=0

    return


def sort_by_order_file(name_lst):
    """
    re-order the list by the order of my_order.csv
    """
    my_order = pd.read_csv("/cs/snapless/gabis/noam.dahan1/hn/my_order.csv").drop(['usable', 'how_many _pages'], axis=1)
    my_order['our_name'] = my_order['our_name'].apply(lambda x: x + "[\d.-]+")
    regex_list = my_order['our_name'].to_list()

    def custom_sort(name):
        for i, regex_pattern in enumerate(regex_list):
            if re.match(regex_pattern, name):
                return i
        # If the name doesn't match any regex, place it at the end
        return len(regex_list)

    sorted_names = sorted(name_lst, key=custom_sort)
    logging.debug(f"START removal of files with segmentation error length {len(sorted_names)}")

    with open('/cs/snapless/gabis/noam.dahan1/hn/seg_prob_list.txt', 'r') as file:
        names_to_remove = file.read().splitlines()

    # print(sorted_names[0])
    # print(names_to_remove[0])

    # Remove entries from name_list that are in the txt file
    sorted_names = [name for name in sorted_names if re.sub(r'[\d\-.]', '', name) not in names_to_remove]
    logging.debug(f"END removal of files with segmentation error length {len(sorted_names)}")


    return sorted_names


def download_one(sect_name,out_dir,timeout_mins):
    """
    gets a name of a single section and downloads it
    :param sect_name: name of a section
    :param out_dir: output directory
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

        # manage io error by writing problematic section in a seperate file:
        try:
            with open(out_fn, "w", encoding="utf8") as fout:
                fout.write(sect_cont.decode("utf8"))

                logging.debug(f"Success, wrote xml to {out_fn}")
            os.chmod(out_fn, 0o777)

        except IOError:
            logging.debug(f"Io problem in section {sect_name}")
            with open("files_with_errors.txt", "w", encoding="utf8") as feror:
                feror.write(f"{sect_name}")

    return



def get_list_so_far(file_path):
    # Initialize an empty list to hold the filenames
    file_names = []

    # Open the file and read each line
    with open(file_path, 'r') as file:
        for line in file:
            # Strip newline characters and add to list
            file_names.append(line.strip())
    print(len(file_names))
    return file_names


def filter_haaretz(item):
    if not item.startswith("haretz"):
        return False  # Keep items that do not start with "haretz"

    year = int(item[6:10])  # Extract the year part
    return year >= 1966


if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = Path(args["--in"]) if args["--in"] else None
    out_dir = Path(args["--out"]) if args["--out"] else None
    timeout_mins = int(args["--timeout"]) * 60 if args["--timeout"] else None
    num_of_threads = int(args["--threads"]) if args["--threads"] else None
    path_to_list = Path(args["--lst"]) if args["--lst"] else None


    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Start computation

    # Get a list of sections that were already downloaded
    #down_fns = [Path(fn).stem for fn in glob(str(out_dir / "*.xml"))]
    down_fns = get_list_so_far(path_to_list)
    #print(down_fns[0])

    # # Get section names to download
    sect_names = [line.strip() for line in open(inp_fn, encoding="utf8")
               if line.strip()]

    updated_sect_names = [s + '.xml' for s in sect_names]

    #print(sect_names)

    logging.debug(f"START set manipulation to get list of files to download")
    # # get a list of sections that weren't already downloaded
    sect_to_download = list(set(updated_sect_names) - set(down_fns))
    #print(f"len sec_to download {len(sect_to_download)}")
    logging.debug(f"END set manipulation to get list of files to download")

    logging.debug(f"START removal of new haaretz")
    #removal of haaretz file with year larger than 1966 since they contain no info
    sect_to_download_no_new_haaretz = [item for item in sect_to_download if not filter_haaretz(item)]
    logging.debug(f"END removal of new haaretz")

    print(sect_to_download[0])
    updated_sect_to_download = [s[:-4] for s in sect_to_download_no_new_haaretz]
    #print(updated_sect_to_download[0])

    #
    #add sort
    sect_to_download = sort_by_order_file(updated_sect_to_download)
    #print(sect_to_download[0])

    #
    logging.debug(f"FINISH collecting list of files to download")
    num_to_download = len(sect_to_download)


    #to run full
    manage_downloads_by_pub(out_dir, timeout_mins, sect_to_download, num_to_download,path_to_list)
    #manage_downloads_by_pub(out_dir, timeout_mins, sect_to_download[:500], 500,path_to_list)
    logging.debug(f"DONE ")

