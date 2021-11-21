SUBSET=$1
export HDF5_USE_FILE_LOCKING=FALSE
python3 -m exp.gen_noun_negatives.cache_neg_features_flickrjp --subset $SUBSET
