import xml.etree.ElementTree as ET
import os
import re
import MeCab
from global_constants import flickr_jp_paths
from global_constants import flickr_paths

def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to

    """
    # read Japanese data
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        
        if not sentence or sentence == '' or sentence == ' ' or sentence is None:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []

        #preprocess japanese data
        itr = re.finditer("[0-9]\:", sentence) # split by number
        indices = [m.start(0) for m in itr]
        substrs = [sentence[indices[i]+2:indices[i+1]] if i+1 < len(indices) else sentence[indices[i]+2:] for i in range(len(indices))]
        phrase_nums = [sentence[indices[i]] for i in range(len(indices))]
        tokenizer = MeCab.Tagger("-Owakati")
        for n, substr in enumerate(substrs):
            start_itr = re.finditer("\[", substr)
            end_itr = re.finditer("\]", substr)
            phrase_start = [m.start(0) for m in start_itr]
            phrase_end = [m.start(0) for m in end_itr]
            phrase_data = substr[phrase_start[0]+1:phrase_end[0]]
            p_data = phrase_data.split()
            parts = p_data[0].split('/')
            # phrase id
            phrase_id.append(parts[1])
            # phrase type
            phrase_type.append(parts[2:])
            # add phrase to phrases and words
            phrases.append(p_data[1])
            # concat text and tokenize
            text = p_data[1] + substr[phrase_end[0]+1:]
            tok = tokenizer.parse(text).split()
            # first word index of this phrase is len of all words before (+1 for this word, -1 because 0 index)
            first_word.append(len(words))
            # add text to words
            words.append(tok)
            # then flatten out into one list of tokens so we can keep track of word indices
            words = [word for sublist in words for word in sublist]
        curr_sent = ' '.join(words)
        tok = tokenizer.parse(curr_sent).split()
        sentence_data = {'sentence' : ' '.join(words), 'phrases' : []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index' : index,
                                             'phrase' : phrase,
                                             'phrase_id' : p_id,
                                             'phrase_type' : p_type})

        annotations.append(sentence_data)
    return annotations

def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the 
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes' : {}, 'scene' : [], 'nobox' : []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info


def test():
    fn = '2426047791'
    anno_dir = flickr_jp_paths['anno_dir']
    sent_dir = flickr_jp_paths['sent_dir']
    anno_info = get_annotations(f'{anno_dir}/{fn}.xml')
    sent_info = get_sentence_data(f'{sent_dir}/{fn}.txt')
    import pdb; pdb.set_trace()


if __name__=='__main__':
    test()
