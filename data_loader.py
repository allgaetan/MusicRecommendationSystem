import os
import h5py
import glob
import hdf5_getters
import numpy as np
import pandas as pd

def get_all_files(basedir, ext='.h5') :
    """
    Récupère tous les fichiers HDF5 d'un répertoire principal
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files :
            allfiles.append(os.path.normpath(f))
    return allfiles

def get_all_metadata(h5_file):
    """
    Récupère toutes les metadonnées d'un fichier HDF5 (donc une chanson) dans un dictionnaire
    """
    metadata = {}
    h5 = hdf5_getters.open_h5_file_read(h5_file)
    numSongs = hdf5_getters.get_num_songs(h5)

    getters = list(filter(lambda x: x[:4] == 'get_', hdf5_getters.__dict__.keys()))
    getters.remove("get_num_songs") # special case
    getters = np.sort(getters)

    for getter in getters:
        res = hdf5_getters.__getattribute__(getter)(h5, 0)
        if res.__class__.__name__ == 'ndarray':
            metadata[getter[4:]] = res
        else:
            metadata[getter[4:]] = res

    h5.close()
    return metadata

### MAIN ###
if __name__ == "__main__":
    # Charger les fichiers HDF5 (Million Song Subset)
    millionsongsubset_path = "./data/MillionSongSubset"
    h5_files = get_all_files(millionsongsubset_path)
    print(f"Nombre de fichiers HDF5 trouvés : {len(h5_files)}")

    # Visualisation du premier fichier HDF5
    h5_file = h5_files[0]
    metadata = get_all_metadata(h5_file)
    print(f"Exemple de métadonnées du fichier {h5_file}:")
    for key, value in list(metadata.items()):  
        print(f"{key}: {value}")
    