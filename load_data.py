import os

videos_path = "Sample_LiRoPod"
sample_classes = {}

for dir_name in os.listdir(videos_path):
    class_path = os.path.join(videos_path, dir_name)

    if os.path.isdir(class_path):
        sample_classes[dir_name] = {}

        for split in os.listdir(class_path):
            split_path = os.path.join(class_path, split)

            if os.path.isdir(split_path):
                videos = [
                    video for video in os.listdir(split_path)
                    if os.path.isfile(os.path.join(split_path, video))
                ]

                sample_classes[dir_name][split] = videos

print(sample_classes.keys())
