from datasets import load_dataset

# should probably find a better way to do this, but copying yulan mini basically after downloading
def download_data(names, dir):
    # downloads the data into specified directory
    for name in names:
        dataset = load_dataset(name, split="train", cache_dir=dir, num_proc=8)
        print(dataset)

if __name__ == "__main__":
    save_dir = "./data"
    # we have a total of 8 datasets to download
    names = ["semran1/full"]
    download_data(names, dir)
