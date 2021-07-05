"""
The .csv preperation functions for WSJ0-Mix.

Author
 * Cem Subakan 2020

 """

import os
import csv


def prepare_wsjmix(
    datapath,
    savepath,
    n_spks=4,
    skip_prep=False,
    librimix_addnoise=False,
    fs=8000,
):
    """
    Prepared wsj2mix if n_spks=2 and wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        librimix_addnoise: If True, add whamnoise to librimix datasets
    """

    if skip_prep:
        return

    if "wsj" in datapath:
        assert (
            "4speakers" in datapath
        ), "Inconsistent number of speakers and datapath"
        create_wsj_csv(datapath, savepath)
    else:
        print("Creating a csv file for a custom dataset")
        create_custom_dataset(datapath, savepath)


def create_custom_dataset(
    datapath,
    savepath,
    dataset_name="custom",
    set_types=["train", "valid", "test"],
    folder_names={
        "bass": "bass",
        "drums": "drums",
        "other": "other",
        "vocals": "vocals",
        "mixture": "mixture",
    },
):
    """
    This function creates the csv file for a custom source separation dataset
    """

    for set_type in set_types:
        mix_path = os.path.join(datapath, set_type, folder_names["mixture"])
        s1_path = os.path.join(datapath, set_type, folder_names["bass"])
        s2_path = os.path.join(datapath, set_type, folder_names["drums"])
        s3_path = os.path.join(datapath, set_type, folder_names["other"])
        s4_path = os.path.join(datapath, set_type, folder_names["vocals"])

        print(mix_path)
        print(s1_path)
        print(s2_path)
        print(s3_path)
        print(s4_path)

        files = os.listdir(mix_path)

        mix_fl_paths = [os.path.join(mix_path, fl) for fl in files]
        s1_fl_paths = [os.path.join(s1_path, fl) for fl in files]
        s2_fl_paths = [os.path.join(s2_path, fl) for fl in files]
        s3_fl_paths = [os.path.join(s3_path, fl) for fl in files]
        s4_fl_paths = [os.path.join(s4_path, fl) for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
            "s4_wav",
            "s4_wav_format",
            "s4_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        with open(
            os.path.join(savepath, dataset_name + "_" + set_type + ".csv"), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, s3_path, s4_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths,
                    s2_fl_paths, s3_fl_paths, s4_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                    "s4_wav": s4_path,
                    "s4_wav_format": "wav",
                    "s4_wav_opts": None,
                }
                writer.writerow(row)


def create_wsj_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")
        s3_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s3/")
        s4_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s4/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]
        s4_fl_paths = [s4_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
            "s4_wav",
            "s4_wav_format",
            "s4_wav_opts",
        ]

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, s3_path, s4_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths,
                    s2_fl_paths, s3_fl_paths, s4_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                    "s4_wav": s4_path,
                    "s4_wav_format": "wav",
                    "s4_wav_opts": None,
                }
                writer.writerow(row)
