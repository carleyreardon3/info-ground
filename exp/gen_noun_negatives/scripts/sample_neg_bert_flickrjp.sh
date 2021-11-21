SUBSET=$1
python3 -m exp.gen_noun_negatives.sample_neg_bert_flickrjp \
    --subset $SUBSET \
    --rank 30 \
    --select 25
