echo "Processing train set"
python3 -m exp.gen_noun_negatives.identify_tokens_flickrjp --subset train
echo "Processing val set"
python3 -m exp.gen_noun_negatives.identify_tokens_flickrjp --subset val