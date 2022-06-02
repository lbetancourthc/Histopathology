import gdown
import zipfile
import argparse

data_samples = {
    "panda2": ""
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