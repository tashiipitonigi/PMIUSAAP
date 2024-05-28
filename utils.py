import os
from pathlib import Path

def load_files_in_directory(directory, load_function):
    if not directory.endswith('/'):
        directory += '/'

    files = os.listdir(directory)

    loaded_files = {}
    for filename in files:
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filepath.find('.mtl') == -1:
            try:
                f = Path(filename).stem
                loaded_data = load_function(filepath)
                loaded_files[f] = loaded_data
                #print(f"Loaded '{filename}' successfully.")
            except Exception as e:
                print(f"Error loading '{filename}': {e}")

    return loaded_files


def lerp(a, b, t):
    return a + (b - a) * t