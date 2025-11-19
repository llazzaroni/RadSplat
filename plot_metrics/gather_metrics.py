import os
import shutil

FOLDER = "/work/courses/dslab/team20/rbollati/running_env/experiments"
errors = []

# for ind, i in enumerate(os.listdir(FOLDER)):
#
#     print(f"copying: {i}")
#
#     if 'gsplat_stats.json' in os.listdir(f"{FOLDER}/{i}") and 'time_logs.txt' in os.listdir(f"{FOLDER}/{i}") and 'metadata.json' in os.listdir(f"{FOLDER}/{i}"):
#         os.mkdir(f"experiments_results/{ind}")
#         for name in ["gsplat_stats.json","time_logs.txt","metadata.json"]:
#             shutil.copy(f"{FOLDER}/{i}/{name}",f"experiments_results/{ind}/{name}" )
#     else:
#         errors.append(i)

matches = [
    d for d in os.listdir(FOLDER)
    if d.endswith("_SFM") and os.path.isdir(os.path.join(FOLDER, d))
]
for ind, i in enumerate(matches):

    print(f"copying: {i}")

    if 'gsplat_stats.json' in os.listdir(f"{FOLDER}/{i}") and 'time_logs.txt' in os.listdir(f"{FOLDER}/{i}"):
        folder_name = f"experiments_results/SFM_{i.split('_')[2]}"
        os.mkdir(folder_name)
        for name in ["gsplat_stats.json","time_logs.txt"]:
            shutil.copy(f"{FOLDER}/{i}/{name}",f"{folder_name}/{name}" )
    else:
        errors.append(i)

print(errors)
