from data_loader import *

millionsongsubset_path = "./data/MillionSongSubset"
h5_files = get_all_files(millionsongsubset_path)
metadata_list = [get_all_metadata(file) for file in h5_files] 
metadata_df = pd.DataFrame(metadata_list)
print(metadata_df)
