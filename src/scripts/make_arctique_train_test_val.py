import numpy as np
import os
from pathlib import Path
import shutil
import pandas as pd

orig_train_path = Path("/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/arctique_dataset/original_data//v1-0/train") #Path("C:/Users/cwinklm/Documents/Data/v_review_sample10/train")
target_folder = Path("/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/arctique_dataset/arctique") #Path("C:/Users/cwinklm/Documents/Data/v_review_sample10/train_test_val")
os.makedirs(target_folder, exist_ok=True)
#for split_name in ["train", "test", "val"]: 
os.makedirs(target_folder.joinpath("images"), exist_ok=True)
os.makedirs(target_folder.joinpath("masks", "instance"), exist_ok=True)
os.makedirs(target_folder.joinpath("masks", "semantic"), exist_ok=True)


all_samples = [int(n.split("_")[1].split(".")[0]) for n in os.listdir(orig_train_path.joinpath("images"))]
np.random.shuffle(all_samples)

train_percent = 0.7
n_train = int(len(all_samples)*train_percent)
n_test_val = len(all_samples) - n_train

train_samples = all_samples[:n_train]
val_samples = all_samples[n_train:n_train+n_test_val//2]    
test_samples = all_samples[n_train+n_test_val//2:]    
labels = ["train"]*len(train_samples) + ["val"]*len(val_samples) + ["test"]*len(test_samples)

split_dict = pd.DataFrame({"sample_name":all_samples, "train_test_val_split":labels})
split_dict.to_csv(target_folder.joinpath("train_test_val_split.csv"), index=False)

for sample_idx, sample_name in enumerate(all_samples): 
    shutil.copy(orig_train_path.joinpath("images", f"img_{sample_name}.png"), 
                target_folder.joinpath("images", f"img_{sample_name}.png"))
    
    shutil.copy(orig_train_path.joinpath("masks", "semantic", f"{sample_name}.tif"), 
            target_folder.joinpath("masks", "semantic", f"{sample_name}.png"))

    shutil.copy(orig_train_path.joinpath("masks", "instance", f"{sample_name}.tif"), 
            target_folder.joinpath("masks", "instance", f"{sample_name}.png"))
