import requests
import numpy as np
import sys
import os
from scipy.io import savemat, loadmat

basepath = os.path.dirname(os.path.realpath(__file__))
datasets_store_dir = os.path.join(basepath, ".")

automatic_statistician_download_url = "https://github.com/jamesrobertlloyd/gpss-research/raw/" \
                                      "2a64958a018f1668f7b8eedf33c4076a63af7868/data/tsdlr-renamed/"


def automatic_statistician_process_dataset(filename, descr, url):
    filepath = f"{datasets_store_dir}/{filename}"
    d = loadmat(filepath)
    if 'Y' not in d:
        d['Y'] = d['y']

    nodata = np.full_like((1,), np.nan, dtype=np.double)
    savemat(filepath,
            {'X': d['X'], 'Y': d['Y'], 'tX': nodata, 'tY': nodata,
             'name': filename.split("-")[-1].split(".")[0],
             'description': descr,
             "url": url})


automatic_statistician_datasets = [
    {"filename": "01-airline.mat", "descr": "-", "process_func": automatic_statistician_process_dataset},
    {"filename": "02-solar.mat", "descr": "-", "process_func": automatic_statistician_process_dataset},
    {"filename": "03-mauna.mat", "descr": "-", "process_func": automatic_statistician_process_dataset},
    {"filename": "04-wheat.mat", "descr": "-", "process_func": automatic_statistician_process_dataset},
    {"filename": "05-temperature.mat", "descr": "-", "process_func": automatic_statistician_process_dataset},
    {"filename": "06-internet.mat", "descr": "-", "process_func": automatic_statistician_process_dataset},
    {"filename": "07-call-centre.mat", "descr": "-", "process_func": automatic_statistician_process_dataset},
    {"filename": "08-radio.mat", "descr": "-", "process_func": automatic_statistician_process_dataset}
]


def download_file(url, target_dir='.'):
    local_filename = "%s/%s" % (target_dir, url.split('/')[-1])
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # f.flush() commented by recommendation from J.F.Sebastian
    return local_filename


def setup_datasets():
    print("Downloading files... This may take a while...")
    for ds in automatic_statistician_datasets:
        filename, descr, process_func = ds["filename"], ds["descr"], ds["process_func"]
        filepath = f"{datasets_store_dir}/{filename}"
        url = f"{automatic_statistician_download_url}{filename}"
        if not os.path.exists(filepath):
            print("%-*s" % (40, filename), end="... ")
            sys.stdout.flush()
            download_file(url, datasets_store_dir)
            process_func(filename, descr, url)
        else:
            print("Skipping.", end="")
        print("")
    print("")

    automatic_statistician_process_dataset("snelson1d.mat", "Snelson 1d classic dataset",
                                           "http://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip")

    # print("Processing downloaded files...")
    # process_yearpredmsd()
    # process_kin40k()
    # process_protein()
    # # process_household_electric()
    # process_naval()
    # process_snelson()
    # process_mnist()
