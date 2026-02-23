import os

# Define the label folder
label_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/labelled_kitti/training/label_yolo"

# Go through all label files and remove lines that start with '7'
removed_counts = {}
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(label_dir, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
        original_len = len(lines)
        filtered_lines = [line for line in lines if not line.strip().startswith("7 ")]
        new_len = len(filtered_lines)

        if new_len < original_len:
            removed_counts[filename] = original_len - new_len
            with open(filepath, "w") as file:
                file.writelines(filtered_lines)

import pandas as pd
import ace_tools as tools

# Show results to user
removed_df = pd.DataFrame(list(removed_counts.items()), columns=["filename", "lines_removed"])
tools.display_dataframe_to_user(name="Files Cleaned of Class 7 (Misc)", dataframe=removed_df)

