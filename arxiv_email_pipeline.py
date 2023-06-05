import urllib, urllib.request    
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.adapters import TimeoutSauce
from datetime import datetime
import os
import feedparser
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import PyPDF2
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import re

start = time.time()

date = "06-04-23"
directory = f".//data//arxiv emails//{date}//" # change manually

# read in email from arxiv
# currently a txt file i copied email contents into but could change to read email directly(?)
with open(f"{directory}email.txt", 'r') as f:
    # file_content = f.read()
    arxiv_ids = [line.strip() for line in f]
# arxiv_ids = re.findall(r'^arXiv:\d{4}\.\d{5}', file_content) # looks like 'arXiv:2303.11347'
print(f"Checking {len(arxiv_ids)} papers")

# download pdfs from arxiv ids
headers = {
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36',
        }
session = requests.Session()
retry = Retry(connect=3, backoff_factor=10)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

files = np.unique(os.listdir(directory))

for arxiv_id in arxiv_ids:
    id_ = arxiv_id[6:]

    # pdf_link = f"http://export.arxiv.org/pdf/{id_[6:]}"
    pdf_link = f"http://arxiv.org/pdf/{arxiv_id}"
    filename = f'{directory}{id_}.pdf'

    # print(pdf_link)

    if f"{id_}.pdf" in files:
        file_size = os.path.getsize(filename)
        if file_size < 80000:
            pdf_link = f"http://arxiv.org/pdf/{arxiv_id}"
            pass
        else:
            continue

    response = session.get(pdf_link, headers=headers)
    with open(filename, 'wb') as f:
        f.write(response.content)

files = np.unique(os.listdir(directory))

bad_papers = []
good_papers = []
good_titles = []
paper_texts = []
for filename in files: 
    # filename looks like 1202.5809.pdf
    if ("pdf" not in filename) or (filename in bad_papers):
        continue

    try:
        reader = PyPDF2.PdfReader(directory+filename) # EOF error when file is corrupt
    except:
        if filename not in bad_papers:
            bad_papers.append(filename)
        continue

    paper_text = ""
    paper_good = True
    for page in reader.pages:
        if not paper_good:
            continue
            
        try:
            text = page.extract_text() # grabs file name if text is nonselectable
        except:
            paper_good = False
            if filename not in bad_papers:
                bad_papers.append(filename)
            continue

#                 if text in filename: # triggers when the pdf contains a blank page (4 in feb set), change in future versions of Convert PDFs to Test notebook (ie future re-training)
#                     paper_good = False
#                     if filename not in bad_papers:
#                         bad_papers.append(filename)
#                     continue

        paper_text += text

    if len(paper_text) > 1000:
        paper_text = paper_text.replace('\n', '').replace("\r", "")
        for punct in string.punctuation:
            paper_text = paper_text.replace(punct, "")
        good_papers.append(filename)
        paper_texts.append(paper_text)
    else:
        paper_good = False
        if filename not in bad_papers:
            bad_papers.append(filename)
        continue 

def tokenize_text(text):
    stopWords = set(stopwords.words('english'))

    tokens = []
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if len(word) < 2 or word in stopWords:
                continue
            tokens.append(word.lower())
    return tokens

titles = []
links = []
for filename in good_papers:
    arxiv_id = filename[:-4]
    query_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = urlopen(query_url).read()
    feed = feedparser.parse(response)
    title = [e.title_detail["value"] for e in feed.entries][0]
    titles.append(title)
    link = [e.link for e in feed.entries][0]
    links.append(link)

doc2vec_path=".//models//doc2vec//model_dbow_vs_600_n_5_s_0.bin"
doc2vec_model = Doc2Vec.load(doc2vec_path)

print(f"Number of good papers:", len(paper_texts))
X = [doc2vec_model.infer_vector(tokenize_text(paper_text)) for paper_text in paper_texts]

classifier_path=".//models//classifiers//lr.sav"
classifier = pickle.load(open(classifier_path, 'rb'))

probs = classifier.predict_proba(X) # need this if inferring one at a time: X.reshape(1, -1) also [0] at end
ea_probs = [prob[1] for prob in probs]

sorted_ea_probs = sorted(ea_probs, reverse=True)
sorted_good_papers = [paper for prob, paper in sorted(zip(ea_probs, good_papers), reverse=True)]
sorted_titles = [title.replace("\n", "") for prob, title in sorted(zip(ea_probs, titles), reverse=True)]
sorted_links = [link for prob, link in sorted(zip(ea_probs, links), reverse=True)]

with open(f'{directory}results.txt', 'w', encoding="utf-8") as f:
    f.write(f"ML Predictions for the arXiv email sent on {date}\n")
    for prob, title, link in zip(sorted_ea_probs, sorted_titles, sorted_links):
        # f.write(f"{round(prob,3)} | <a href='{link}'>{title}</a>\n")
        f.write(f"{round(prob,3)} | {title} | {link}\n")

with open(f'{directory}bad_papers.txt', 'w') as f:
    for paper in bad_papers:
        f.write(f"{paper}\n")

duration = time.time() - start
print("Duration (min):", duration/60)