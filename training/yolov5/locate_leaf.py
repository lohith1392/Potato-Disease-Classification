import pandas as pd
import os
import shutil
import ast
from sklearn import model_selection
from tqdm import tqdm
import numpy as np

DATA_PATH = "E:/Potato-Disease/training/archive"
OUTPUT_PATH = "E:/Potato-Disease/training/yolov5/leaf/"


def process_data(data, data_type="train"):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row['image_id']
        bounding_boxes = row['bboxes']
        yolo_data = []
        for bbox in bounding_boxes:
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x_center = x + w / 2
            y_center = y + h / 2
            x_center /= float(w)
            y_center /= float(h)
            w /= float(w)
            h /= float(h)
            yolo_data.append([0, x_center, y_center, w, h])
        yolo_data = np.array(yolo_data)
        np.savetxt(
            os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name[0:len(image_name)-4]}.txt"),
            yolo_data,
            fmt=["%d", "%f", "%f", "%f", "%f"]
        )
        shutil.copyfile(
            os.path.join(DATA_PATH, f"train/{image_name}"),
            os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}")
        )


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    df.bbox = df.bbox.apply(ast.literal_eval)
    df = df.groupby("image_id")["bbox"].apply(list).reset_index(name="bboxes")

    df_train, df_valid = model_selection.train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    process_data(df_train,data_type="train")
    process_data(df_valid,data_type="validation")
