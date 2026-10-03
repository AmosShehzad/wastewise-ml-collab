import os


def count_images_by_class(data_dir):
    """Count the number of images in each class folder."""
    class_count = {}

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        if os.path.isdir(class_path):
            class_count[class_name] = len(os.listdir(class_path))

    return class_count
