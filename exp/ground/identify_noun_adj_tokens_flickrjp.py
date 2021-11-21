import click
import argparse
import os
import nltk
import faulthandler
from tqdm import tqdm
import MeCab
import utils.io as io
from .dataset_flickrjp import FlickrJPDatasetConstants, FlickrJPDataset
from .models.multi_cap_encoder import MultiCapEncoderConstants, MultiCapEncoder
from exp.gen_noun_negatives.identify_tokens import (combine_subtokens, ignore_words_from_pos)
from exp.gen_noun_negatives.identify_tokens_flickrjp import align_pos_tokens_jp
#from .identify_noun_adj_tokens import get_noun_adj_token_ids

faulthandler.enable()
#@click.command()
#@click.option(
#   '--subset',
#    type=click.Choice(['train','val','test']),
#    default='coco subset to identify nouns for')

def get_noun_adj_token_ids_jp(mecab_tokens, pos_tags, alignment):
    token_ids = []
    for i in range(len(mecab_tokens)):
        word = mecab_tokens[i]
        tags = pos_tags[i]
        if tags[0] == "名詞" or tags[0] == "形容詞": #noun or adjective
            #print('ENTERED NOUN OR ADJ CASE')
            #print(len(alignment[i]))
            for idx in alignment[i]:
                token_ids.append(idx)
    #print("TOKEN IDS FOUND = "+str(len(token_ids)))
    return token_ids

def main(args):
    print('Creating Caption Encoder (tokenizer) ...')
    cap_encoder = MultiCapEncoder(MultiCapEncoderConstants())

    #nltk.download('punkt')

    data_const = FlickrJPDatasetConstants(args.subset)
    data_const.read_noun_adj_tokens = False
    data_const.read_neg_noun_samples = False
    dataset = FlickrJPDataset(data_const)
    noun_adj_token_ids = [None]*len(dataset)
    max_item = 0
    last_ids = ""
    for i,data in enumerate(tqdm(dataset)):
    #for i in range((int(len(dataset)/5))):
    #    if i % 1000:
    #        print(i)
    #    data = dataset[i]
        image_id = data['image_id']
#         if image_id == 3652094744:
#             continue
        cap_id = data['cap_id']
        caption = data['caption']
#         if len(caption) == 0:
#             print("Caption is empty")
#             continue
        token_ids, tokens = cap_encoder.tokenize(caption)
        
        #nltk_tokens = nltk.word_tokenize(caption.lower())
        #pos_tags = nltk.pos_tag(nltk_tokens)
        #pos_tags = ignore_words_from_pos(
        #    pos_tags,['is','has','have','had','be'])
        
        tagger = MeCab.Tagger()
        tagger.parse('')
        m = tagger.parseToNode(caption)
        mecab_tokens = []
        pos_tags = []
        while m:
            if m.feature.startswith('BOS/EOS'):
                m = m.next
                continue
            mecab_tokens.append(m.surface) # token
            idx = m.feature.split(',').index('*')
            tags = [t for i, t in enumerate(m.feature.split(',')) if i<idx]
            pos_tags.append(tags) # pos tags
            m = m.next
        
        #print('LENGTH OF MECAB POS TAGS = %d' % len(pos_tags))
        #print('LENGTH OF BERT TOKENS = %d' % len(tokens))
        #print(tokens)
        #alignment = align_pos_tokens_jp(pos_tags,tokens)
        alignment = align_pos_tokens_jp(mecab_tokens,tokens)
        # print('LENGTH OF MECAB TOKENS = %d' % len(mecab_tokens))
        # print('LENGTH OF ALIGNMENT = %d' % len(alignment))
        noun_adj_token_ids_ = get_noun_adj_token_ids_jp(mecab_tokens,pos_tags,alignment)
        
        noun_adj_tokens_ = []
        for k in noun_adj_token_ids_:
            noun_adj_tokens_.append(tokens[k])

        noun_adj_token_ids[i] = {
            'image_id': image_id,
            'cap_id': cap_id,
            'token_ids': noun_adj_token_ids_,
            'tokens': noun_adj_tokens_}
        max_item = i
        last_ids = str(image_id)+", "+str(cap_id)
        if len(noun_adj_token_ids_) == 0:
            print('Issue with noun and adj token idxs for im id %s' % image_id)

#     print(len(dataset))
#     print(max_item)
#     print(last_ids)
#     print(dataset[i])
#     print(dataset[i+1])
    io.dump_json_object(noun_adj_token_ids,data_const.noun_adj_tokens_json)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='to replace kwargs')
    parser.add_argument('--subset', dest='subset', type=str, help='subset to process')
    args = parser.parse_args()
    main(args)