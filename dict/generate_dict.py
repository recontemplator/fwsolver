t#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:19:55 2016

@author: zy
"""
import xml.etree.ElementTree as ET


"""
tree = ET.parse('books.xml')
root = tree.getroot()
print len(root.findall('./book'))
for e in root.findall('./book'):
    print e.find('title').text
"""
i=0
is_sing=is_nomn=is_noun=is_name=is_abbr_form=is_abbr_lemma=False
t=prev_t=None
res_list=[]
for event, elem in ET.iterparse('dict.opcorpora.xml', events=("start", "end")):
    if event=='start' and elem.tag=='lemma':
        is_noun=False
        is_name=False
        is_abbr_lemma=False
        t=set()
    elif event=='end' and elem.tag=='l':
        is_abbr_lemma=is_abbr_form
    elif event=='start' and elem.tag=='f':
        is_sing=is_nomn=is_abbr_form=False
    elif event=='end' and elem.tag=='g':
        is_sing=is_sing or elem.attrib['v']=='sing'
        is_nomn=is_nomn or elem.attrib['v']=='nomn'
        is_noun=is_noun or elem.attrib['v']=='NOUN'
        is_name=is_name or elem.attrib['v']=='Name' 
        is_name=is_name or elem.attrib['v']=='Surn' 
        is_name=is_name or elem.attrib['v']=='Patr' 
        is_name=is_name or elem.attrib['v']=='Orgn' 
        is_name=is_name or elem.attrib['v']=='Trad' 
        is_name=is_name or elem.attrib['v']=='Geox' 
        is_abbr_form=is_abbr_form or elem.attrib['v']=='Abbr' 
    elif event=='end' and elem.tag=='f' and is_sing and is_nomn and not(is_abbr_form):
        t.add(elem.attrib['t'])
    elif event=='end' and elem.tag=='lemma'and is_noun\
    and not(is_name) and not(is_abbr_lemma) and len(t)>0:
            #print t
            res_list.extend(t)
            #prev_t=t
            i+=1
            if i%1000==0:print i,repr(t).decode("unicode-escape")

#print ["%s\n" % item  for item in res_list]        
with open('dict.out','w') as out_file:
    for item in res_list:
        print>>out_file,item.encode('utf8')
#    out_file.writelines(["%s\n" % item  for item in res_list])

#tree = ET.parse('dict.opcorpora.xml')
#root = tree.getroot()
#print len(root.findall('./grammemes'))

print 'Done'

#import libxml2
#doc = libxml2.parseFile('books.xml')
#for url in doc.xpathEval('//bookstore/book/@category'):
#  print url.content
