import os

# if filename is image file, return True
def is_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def make_dataset(dir:str):
    images = []
    assert os.path.isdir(dir), f"{dir} is not a valid directory"

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


