import gdown
import zipfile
import argparse

data_samples = {
    "train_panda2": "https://drive.google.com/u/0/uc?id=1jK-CEqwJjhYIKsHcMlhp2-4lc7ontJfK&export=download",
    "train_voets": "https://drive.google.com/uc?id=1AmcFh1MOOZ6aqKm2eO7XEdgmIEqHKTZ5"
}

def download(data, url):
    # Download dataset
    url = url
    output = "data/{}.zip".format(data)
    gdown.download(url, output, quiet=False)

    # Uncompress dataset
    local_zip = "data/{}.zip".format(data)
    zip_ref = zipfile.ZipFile(local_zip, "r")
    zip_ref.extractall(path="data")
    zip_ref.close()


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset")
    return parser


def main():
    parser = init_argparse()
    args = parser.parse_args()

    UNLABELED = args.dataset

    URL_UNLABELED = data_samples[UNLABELED]
    download(UNLABELED, URL_UNLABELED)

if __name__ == "__main__":
    main()