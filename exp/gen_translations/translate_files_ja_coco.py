# -*- coding: utf-8 -*-
from googletrans import Translator
import os, sys
import nltk
import string
import json

import time



source_language = 'en'

path = '/research/donhk/multi_language/dataset/coco_japanese/'
dirs = os.listdir(path)



with open("/research/donhk/multi_language/dataset/jp_image2caption.josn", "r") as write_file:
    jp_image2caption = json.load(write_file)


image_list = list(jp_image2caption)



source_language = 'ja'
target_language = 'en'

batch = 100


max_iter = len(image_list)


for iteration in range(3959, max_iter):

    print('image idx: {}'.format(iteration))

    translator = Translator(service_urls=[
         'translate.google.com',
         'translate.google.co.kr',
         'translate.google.de',
         'translate.google.fr',
         'translate.google.cz',
         'translate.google.co.jp',
        'translate.google.ch',

    ])

    time.sleep(1.0)
    lines = []
    trans_line = []
    image_name = []

    img = image_list[iteration]

    f = jp_image2caption[img]

    for idx_line, line in enumerate(f):

        line = line.strip()
        image_name.append(img)
        tran = translator.translate(line,  src=source_language, dest=target_language)
        trans_line.append(tran.text)
        # file.write(tran.text + '\n')

    # if len(trans_line)== 100:
    with open(path + 'google_trans_{}_to_{}_translated'.format(source_language, target_language), 'a') as file:
        for trans_idx, trans in enumerate(trans_line):
            file.write(img.encode('utf-8') + '\t' + trans.encode('utf-8') + '\n')

