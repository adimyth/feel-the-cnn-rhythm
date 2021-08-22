from pathlib import Path
from shutil import copyfile
import pandas as pd

from tqdm import tqdm  # type:ignore
import random

count_0 = 300893
count_1 = 486
ratio_default = count_1 / (count_1 + count_0)  # ~0.00161, 1/512 ~ 0.00195, 1/64 ~ 0.015625, 1/32 ~ 0.03125
ratio_new = 0.03125
sampling_frac = ratio_default * (count_1 + count_0) - ratio_new * count_1
sampling_frac = sampling_frac / (count_0 * ratio_new)


def create_dataset_v1():
    p = Path("data/heatmaps")
    files = p.glob("*")

    for i, file in tqdm(enumerate(files)):
        try:
            label = file.name.split("_")[0]
            copyfile(str(file), str(p / "v1" / label / file.name))
        except:
            print(f"failed for i: {i}")


def create_dataset_v2():
    p = Path("data/heatmaps")
    files = p.glob("*")

    for i, file in tqdm(list(enumerate(files))):
        try:
            label = file.name.split("_")[0].strip(" ")

            if label == "0":
                if random.random() < sampling_frac:
                    copyfile(str(file), str(p / "v2" / label / file.name))
            if label == "1":
                copyfile(str(file), str(p / "v2" / label / file.name))
                print(f"Found label 1 for index: {i}")
        except:
            continue
    print(f"sampling_frac: {sampling_frac}")
    print(f"Dataset created under {str(p)}")


def create_dataset_kaggle_v1():
    p = Path("data/heatmaps")
    files = p.glob("*")

    train_records = []
    test_records = []
    q = Path("data/kaggle")

    for i, file in tqdm(enumerate(files)):
        try:
            label = file.name.split("_")[0]
            image_id = file.stem.split("_")[1]
            partition = random.sample(
                ["train", "train", "train", "public", "private"],
                # counts=[3, 1, 1], # counts added in 3.9!
                k=1,
            )[0]
            if partition == "train":
                dest = q / "train" / label
                dest.mkdir(parents=True, exist_ok=True)
                copyfile(str(file), str(dest / file.name))
                train_records += [
                    {
                        "image_id": image_id,
                        "count": label,
                    }
                ]
            else:
                dest = q / "test"
                dest.mkdir(parents=True, exist_ok=True)
                copyfile(str(file), str(dest / (image_id + ".png")))
                test_records += [
                    {
                        "image_id": image_id,
                        "count": label,
                        "Usage": partition,
                    }
                ]
        except Exception as e:
            print(f"failed for i: {i}, {e}")
    t = pd.DataFrame(train_records)
    t.to_csv("train.csv", index=False)
    solution = pd.DataFrame(test_records)
    solution.to_csv("test_solution_with_usage.csv", index=False)
    solution = solution.drop("Usage", axis=1)
    solution.to_csv("test_solution_key.csv", index=False)
    solution.loc[:, "count"] = [5] * solution.shape[0]
    solution = solution.drop("Usage", axis=1)
    solution.to_csv("sample_submission.csv", index=False)


if __name__ == "__main__":
    create_dataset_kaggle_v1()
