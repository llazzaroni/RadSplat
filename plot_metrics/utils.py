import re
import os
import pandas as pd
import json

def parse_time_log(file: str)-> list:

    data = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            step = get_step(line)
            if "gaussian-splatting" in step:
                data.append({
                    "step" : int(step.split('-')[-1]),
                    "time" : float(line.split(' - ')[-1])
                    })

        return data

def get_step(line: str) -> str:
    match = re.search(r"\[(.*?)\]", line)
    if match:
        return match.group(1)  
    return ""

def create_exp_df(src_time_logs :str, src_gsplat_stas: str, src_metadata : str) -> pd.DataFrame:

    with open(src_gsplat_stas, "r") as f:
        parsed = json.load(f)
        gs_splt_stats = pd.DataFrame(parsed)

    time_logs = parse_time_log(src_time_logs)
    time_logs_df = pd.DataFrame(time_logs)
    df = pd.concat([gs_splt_stats, time_logs_df], axis=1)

    with open(src_metadata, "r") as f:
        parsed = json.load(f)
        for k,v in parsed.items():
            df[k] = v
    
    return df

def create_complete_df(folder):
    df = pd.DataFrame()
    for i in os.listdir(folder):
        if "SFM" not in i:
            src_time = f"{folder}/{i}/time_logs.txt"
            src_gsplat = f"{folder}/{i}/gsplat_stats.json"
            src_metadata = f"{folder}/{i}/metadata.json"
            df_exp = create_exp_df(src_time, src_gsplat, src_metadata)
            df = pd.concat([df, df_exp])

    return df


def create_sfm_df(folder : str):

    matches = [
    d for d in os.listdir(folder)
    if d.startswith("SFM") and os.path.isdir(os.path.join(folder, d))]
    complete_df = pd.DataFrame()

    for i in matches:
        with open(f"{folder}/{i}/gsplat_stats.json", "r") as f:
            parsed = json.load(f)
            gs_splt_stats = pd.DataFrame(parsed)

        time_logs = parse_time_log(f"{folder}/{i}/time_logs.txt")
        time_logs_df = pd.DataFrame(time_logs)
        sub_df = pd.concat([gs_splt_stats, time_logs_df], axis=1)
        sub_df["scene-name"] = i.split('_')[-1] 
        complete_df = pd.concat([complete_df, sub_df], axis=0)

    return complete_df


# if __name__ == "__main__":
#
#     # pd.set_option("display.max_columns", None)
#     # res = parse_time_log("experiments_results/0/time_logs.txt")
#     # print(res)
#
#     # create_exp_df("experiments_results/0/time_logs.txt", "experiments_results/0/gsplat_stats.json", "experiments_results/0/metadata.json")
#
#     # res = create_sfm_df("experiments_results")
#     res = create_complete_df("experiments_results")
#     print(res)




