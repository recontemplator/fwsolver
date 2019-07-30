#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:19:55 2016

@author: zy
"""
import xml.etree.ElementTree as ET
from tqdm import tqdm
def build_dict(src_file='dict.opcorpora.xml'):
    i = 0
    is_sing = is_nomn = is_noun = is_name = is_abbr_form = is_abbr_lemma = False
    t = None
    res_list = []

    name_parts = {'Name', 'Surn', 'Patr', 'Orgn', 'Trad', 'Geox'}
    pbar = tqdm(ET.iterparse(src_file, events=("start", "end")), total=42e6, unit_scale=1)
    for event, elem in pbar:
        if event == 'start' and elem.tag == 'lemma':
            is_noun = False
            is_name = False
            is_abbr_lemma = False
            t = set()
        elif event == 'end' and elem.tag == 'l':
            is_abbr_lemma = is_abbr_form
        elif event == 'start' and elem.tag == 'f':
            is_sing = is_nomn = is_abbr_form = False
        elif event == 'end' and elem.tag == 'g':
            is_sing = is_sing or elem.attrib['v'] == 'sing'
            is_nomn = is_nomn or elem.attrib['v'] == 'nomn'
            is_noun = is_noun or elem.attrib['v'] == 'NOUN'
            is_name = is_name or elem.attrib['v'] in name_parts
            is_abbr_form = is_abbr_form or elem.attrib['v'] == 'Abbr'
        elif event == 'end' and elem.tag == 'f' and is_sing and is_nomn and not (is_abbr_form):
            t.add(elem.attrib['t'])
        elif event == 'end' and elem.tag == 'lemma' and is_noun \
                and not (is_name) and not (is_abbr_lemma) and len(t) > 0:
            res_list.extend(t)
            i += 1
            pbar.set_description(f'{i:7} - {str(list(t)[0])[:15]:15}')
    return res_list

if __name__ == "__main__":
    with open('dict.out', 'w') as out_file:
            print('\n'.join(build_dict()), file=out_file)

# tree = ET.parse('dict.opcorpora.xml')
# root = tree.getroot()
# print len(root.findall('./grammemes'))

# import libxml2
# doc = libxml2.parseFile('books.xml')
# for url in doc.xpathEval('//bookstore/book/@category'):
#  print url.content
# """
# tree = ET.parse('books.xml')
# root = tree.getroot()
# print len(root.findall('./book'))
# for e in root.findall('./book'):
#     print e.find('title').text
# """
