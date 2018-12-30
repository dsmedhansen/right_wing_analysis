### Dit bestand prepareert de youtube captions voor gebruik. Het heeft twee aparte delen, die aan -en uitgezet
#   kunnen worden met onderstaande booleans. De onderdelen zijn:
#  -- Scoonmaken en dedupliceren van de captions (alles staat er drie keer in)
#     Dit is redelijk straightforward en efficiënt, duurt minder dan tien minuten bij mij.
#  -- Lemmatiseren van de captions (terugbrengen naar woordenboekvorm)
#     Dit duurt langer, bij mij ca 70 minuten. Gebruik dus waar mogelijk dat databestand in plaats van het script.
#
#  Timing is obv rechtse captions.
#  Beide werken parallel, by default op N-1 cores. Pas N_JOBS aan om dit te veranderen. Tokeniseren/pos-taggen
#  gaat met de defaults van nltk, lemmatiseren met de WordNetLemmatizer obv de POS-tags van nltk.
#
#   --Max

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from joblib import Parallel, delayed
from csv import DictReader, DictWriter
import csv
import re
import os

DATA_PATH = 'transcripts_right.csv'
OUT_FILE_CLEAN = 'transcripts_right_clean.csv'
OUT_FILE_LEMMA = 'transcripts_right_lemma.csv'
N_JOBS = os.cpu_count() - 1

CLEAN_TRANSCRIPTS = True
LEMMATIZE_TRANSCRIPTS = True

def clean_caption(caption_string):
    ''' Clean caption from list format in string to a string, lines separated by \n '''
    segments = re.split(',',re.sub(r'[\[\]]','',caption_string))
    caption = []
    for s in segments:
        try:
            seg = eval(s)
            caption.append(seg.strip())
        except:
            caption.append(s.strip())
            
    caption = [newline for oldline in caption for newline in oldline.split('\n') if newline != ' '] #split lines by newline,
                #unnest the result and keep if line is not empty (in this case just a space)\
    caption = [re.sub(r'(^\'|\'$)','',c) for c in caption]
    caption = [c for c in caption if not c.replace(' ','') == '']
    result = []  #Initialise empty list to store non-duplicates in. Does not use set because lines can be identical,
                #so the criterion is to only drop duplicates if they follow eachother
        
    prevline = ''
    for line in caption:
        if line == prevline:
            continue
        result.append(line)
        prevline = line
    return r'\n'.join(result)

def clean_document(document):
    ''' Clean a document -- worker function for joblib Parallel '''
    newdoc = dict()
    for k,v in document.items():
        if k == 'transcript':
            newdoc['transcript'] = clean_caption(v)
        elif k:
            newdoc[k] = v
        else:
            newdoc = None
            break
    return newdoc


def get_wordnet_pos(treebank_tag):
    ''' Convert NLTK POS tag to WordNet tags. If not in list, return NOUN (default for WordNet) '''

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_document(document, wnl = WordNetLemmatizer()):
    ''' Convert document string to a list of lemmas using nltk tokenizer, pos tagger and WordNetLemmatizer'''
    text = document['transcript'].replace('\\n',' ')
    videoid = document['videoId']
    return {'videoId':videoid,
        'transcript':' '.join([wnl.lemmatize(word, pos = get_wordnet_pos(tag)) for word,tag in pos_tag(word_tokenize(text))])}

if __name__ == '__main__':
    csv.field_size_limit(10**7)

    if CLEAN_TRANSCRIPTS:
        with open(DATA_PATH, encoding = 'utf8') as f:
            clean_file = open(OUT_FILE_CLEAN, 'w+', encoding = 'utf8')

            reader = DictReader(f, delimiter = '¶', quoting=csv.QUOTE_NONE)
            writer_cleaned = DictWriter(clean_file, fieldnames = reader.fieldnames, lineterminator = '\n')
            writer_cleaned.writeheader()

            data = Parallel(N_JOBS, verbose = 2)(delayed(clean_document)(line) for line in reader)

            for row in data:
                if row:
                    writer_cleaned.writerow(row)
            clean_file.close()

    if LEMMATIZE_TRANSCRIPTS: 
        with open(OUT_FILE_CLEAN, encoding = 'utf8') as f:
            lemma_file = open(OUT_FILE_LEMMA, 'w+', encoding = 'utf8')
            writer_lemma = DictWriter(lemma_file, fieldnames = ['videoId','transcript'], lineterminator = '\n')
            writer_lemma.writeheader()
            wnl = WordNetLemmatizer()

            data = Parallel(n_jobs = N_JOBS, verbose = 2)(delayed(lemmatize_document)(row, wnl) for row in DictReader(f))

            for row in data:
                writer_lemma.writerow(row)
            lemma_file.close()
