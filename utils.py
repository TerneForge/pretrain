# note: should implement a avg bitwidth calculation
import datasets
from datasets import load_dataset
def download_pretrain():
    load_dataset("semran1/packed_40B", cache_dir="./data")



if __name__ == "__main__":
    download_pretrain()