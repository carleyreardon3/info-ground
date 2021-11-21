echo "Processing train set"
python3 -m exp.ground.identify_noun_adj_tokens_flickrjp --subset train
echo "Processing val set"
python3 -m exp.ground.identify_noun_adj_tokens_flickrjp --subset val