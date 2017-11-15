import os
import cv2



# Parameters (train data, which is split into train + val)
location_count = 5
locations = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
data_size_per_location = 36
validation_size_per_location = 5
train_size_per_location = data_size_per_location - validation_size_per_location
raw_extension = ".tif"
processed_extension = ".png"

data_size = location_count*data_size_per_location
validation_size = location_count*validation_size_per_location
train_size = location_count*train_size_per_location

image_channel_count = 3
image_res = 5000
gt_class_count = 2
tile_res = 1000
tile_count = 5

# Root directory of the project
root_dir = os.getcwd()

raw_dir_path = os.path.join(root_dir, "data/AerialImageDataset")

processed_dir_path = os.path.join(root_dir, "data/processed")


def process_image(name, raw_subset, processed_subset):
    image_filename = os.path.join(raw_dir_path, raw_subset, "images", name + raw_extension)
    img = cv2.imread(image_filename)
    if img is not None:
        for i in range(tile_count):
            for j in range(tile_count):
                tile = img[i*tile_res:(i+1)*tile_res, j*tile_res:(j+1)*tile_res, :]
                tile_name = '{}.{:04d}.{:04d}'.format(name, i, j)
                tile_filename = os.path.join(processed_dir_path, processed_subset, "images", tile_name + processed_extension)
                print(tile_filename)
                cv2.imwrite(tile_filename, tile)
    else:
        print("Image {} could not be read".format(image_filename))


def process_gt(name, raw_subset, processed_subset):
    image_filename = os.path.join(raw_dir_path, raw_subset, "gt", name + raw_extension)
    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        for i in range(tile_count):
            for j in range(tile_count):
                tile = img[i * tile_res:(i + 1) * tile_res, j * tile_res:(j + 1) * tile_res]
                tile_name = '{}.{:04d}.{:04d}'.format(name, i, j)
                tile_filename = os.path.join(processed_dir_path, processed_subset, "gt", tile_name + processed_extension)
                print(tile_filename)
                cv2.imwrite(tile_filename, tile)
    else:
        print("Image {} could not be read".format(image_filename))


if __name__ == "__main__":
    print("Processing data...")
    for location_name in locations:
        for i in range(0, train_size_per_location):
            image_name = location_name + str(i + 1)
            print(image_name)
            process_image(image_name, "train", "train")
            process_gt(image_name, "train", "train")

        for i in range(train_size_per_location, data_size_per_location):
            image_name = location_name + str(i + 1)
            print(image_name)
            process_image(image_name, "train", "val")
            process_gt(image_name, "train", "val")


























