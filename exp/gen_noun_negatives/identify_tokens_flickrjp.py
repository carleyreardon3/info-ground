import click
import argparse
import os
import nltk
import faulthandler
from tqdm import tqdm
import MeCab
import utils.io as io
from .dataset_flickrjp import FlickrJPDatasetConstants, FlickrJPDataset, flickr_jp_paths
from .models.multi_cap_encoder import MultiCapEncoderConstants, MultiCapEncoder
from .identify_tokens import (combine_subtokens, group_token_ids, ignore_words_from_pos)

faulthandler.enable()
#@click.command()
#@click.option(
#    '--subset',
#    type=str,
#    default='train',
#    help='subset to identify nouns for')

def align_pos_tokens_jp(mecab_tokens,tokens):
    alignment = [None]*len(mecab_tokens)
    for i in range(len(alignment)):
        alignment[i] = []
    
    token_len = len(tokens)
    last_match = -1
    skip_until = -1
    for i, word in enumerate(mecab_tokens):
        for j in range(last_match+1,token_len):
            if j < skip_until:
                continue
            
            if j==skip_until:
                skip_until = -1

            token = tokens[j]
            if word==token:
                alignment[i].append(j)
                last_match = j
                break
            elif len(token)>2 and token[:2]=='##':
                combined_token, sub_token_count = combine_subtokens(tokens,j-1)
                skip_until = j-1+sub_token_count
                if word==combined_token:
                    for k in range(sub_token_count):
                        alignment[i].append(k+j-1)
                        last_match = j-1+sub_token_count-1
                 
    return alignment

def get_noun_token_ids_jp(mecab_tokens, pos_tags, alignment):
    noun_words = set()
    token_ids = []
    for i in range(len(mecab_tokens)):
        word = mecab_tokens[i]
        tags = pos_tags[i]
        if tags[0] == "名詞": #noun
            noun_words.add(word)
            for idx in alignment[i]:
                token_ids.append(idx)
    return token_ids, noun_words

def main(args):
    print('Creating Caption Encoder (tokenizer) ...')
    cap_encoder = MultiCapEncoder(MultiCapEncoderConstants())

    #nltk.download('punkt')
    
    # check kwargs
    print(args.subset)
    
    data_const = FlickrJPDatasetConstants(args.subset)
    data_const.read_noun_token_ids = False
    dataset = FlickrJPDataset(data_const)
    noun_token_ids = [None]*len(dataset)
    noun_vocab = set()
    num_human_captions = 0
    num_noun_captions = 0
    for i,data in enumerate(tqdm(dataset)):
        image_id = data['image_id']
        cap_id = data['cap_id']
        caption = data['caption']
        
        if caption is None:
            print(image_id, cap_id)
            continue
        
        token_ids, tokens = cap_encoder.tokenize(caption)
        
        #nltk_tokens = nltk.word_tokenize(caption.lower())
        #pos_tags = nltk.pos_tag(nltk_tokens)
        #pos_tags = ignore_words_from_pos(
        #    pos_tags,['is','has','have','had','be'])
        
        #if image_id == "3652094744":
        #    print(cap_id, caption)
        
        tagger = MeCab.Tagger()
        tagger.parse('')
        #if caption is None:
        #    print("Null caption: "+str(image_id)+", "+str(cap_id))
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

        alignment = align_pos_tokens_jp(mecab_tokens,tokens)
        noun_token_ids_, noun_words = get_noun_token_ids_jp(mecab_tokens, pos_tags, alignment)
        noun_token_ids_ = group_token_ids(noun_token_ids_, tokens)
        if len(noun_token_ids_) > 0:
            num_noun_captions += 1
        
        noun_token_ids[i] = {
            'image_id': image_id,
            'cap_id': cap_id,
            'token_ids': noun_token_ids_,
            'words': list(noun_words)}
        
        noun_vocab.update(noun_words)
        
        ##  japanese for: man, person, human, woman, boy, girl, people, child, children
        for human_word in ['男','人','人間','女性','男子','女子','人々','子','子供']:
            if human_word in tokens:
                num_human_captions += 1
                break

    io.mkdir_if_not_exists(os.path.join(flickr_jp_paths['proc_dir'],'annotations'))
    io.dump_json_object(noun_token_ids,data_const.noun_tokens_json)
    io.dump_json_object(sorted(list(noun_vocab)),data_const.noun_vocab_json)
    print('Number of human captions:',num_human_captions)
    print('Number of noun captions:',num_noun_captions)
    print('Total number of captions:',len(dataset))
    print('Size of noun vocabulary:',len(noun_vocab))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='to replace kwargs')
    parser.add_argument('--subset', dest='subset', type=str, help='subset to process')
    args = parser.parse_args()
    main(args)
