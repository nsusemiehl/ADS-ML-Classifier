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
import ads
import urllib3
from urllib.parse import urlencode, quote_plus
import argparse
# from progress.bar import Bar

class ml_doc_classifier:
    def __init__(self, start_day, start_month, start_year, end_day, end_month, end_year, output_type, rerun=False):
        """ Add doc string
        """
        # user gives start + end dates
        # note to user: api calls at 12:00am (00:00:00 HH:MM:ss) of days
        # allow user to change hour?
        self.start_day = start_day
        self.start_month = start_month
        self.start_year = start_year

        self.end_day = end_day
        self.end_month = end_month
        self.end_year = end_year

        # if type(self.start_day) == int:
        #     if int(self.start_day) < 10:
        #         self.start_day = f"0{self.start_day}"
        #     if int(self.start_month) < 10:
        #         self.start_month = f"0{self.start_month}"
        #     if int(self.end_day) < 10:
        #         self.end_day = f"0{self.end_day}"
        #     if int(self.end_month) < 10:
        #         self.end_month = f"0{self.end_month}"

        # elif type(self.start_day) == str:
        #     if len(self.start_day) == 1:
        #         self.start_day = f"0{self.start_day}"
        #     if len(self.start_month) == 1:
        #         self.start_month = f"0{self.start_month}"
        #     if len(self.end_day) == 1:
        #         self.end_day = f"0{self.end_day}"
        #     if len(self.end_month) == 1:
        #         self.end_month = f"0{self.end_month}"

        # prepare directories to store paper ids and pdfs (ids need to be saved long term, pdfs do not)
        self.root_directory = f".//data//api_pipeline//"
        if not os.path.exists(self.root_directory):
            os.makedirs(self.root_directory)
        self.sub_directory = f"{self.root_directory}{start_year[2:]}{start_month}{start_day}_{end_year[2:]}{end_month}{end_day}//"
        if not os.path.exists(self.sub_directory):
            os.makedirs(self.sub_directory)

        # check for existing ids (files might not exist yet)
        # doing it this way would ensure no duplicate papers are checked but would prevent running this program on the same date (unless the rerun flag is used)
        self.rerun = rerun # user sets this to true if re-running predictions on a date that was already queried (if true existing ids are ignored, if false new ids are checked against existing ids)

        self.existing_arxiv_ids = [] # like all_titles, these lists are only used to check for global presence of ids
        self.existing_ads_bibs = []

        # try:
        with open(f'{self.root_directory}all_existing_results.txt', 'a+') as f:
            for line in f:
                # line should be "id,prob\n"
                line = line.strip()
                id_ = line.split(",")[0]
                if re.search("^\d+\.\d+$", id_) is not None:
                    self.existing_arxiv_ids.append(id_)
                else:
                    self.existing_ads_bibs.append(id_)
        # except:
        #     pass

        # need to have title lists that are the same length as the id lists
        self.arxiv_ids = []
        self.ads_bibs = []
        self.arxiv_titles = [] 
        self.ads_titles = []

        self.all_titles = [] # all_titles wont have the same length as the other title lists, just used to check global presence of titles

        self.downloaded_ids = []
        self.downloaded_titles = []

        self.converted_ids = []
        self.converted_titles = []

        self.bad_ids = []
        self.bad_titles = []

        self.ADS_DEV_KEY = "FEbkvybqpxpJQdeF5E8btv7KvfFM4Eh5ezHV7Z8S"

        self.output_type = output_type

    def arxiv_api_call(self):
        """ Add doc string
        """

        start =  f"{self.start_year}{self.start_month}{self.start_day}" #start_date.strftime("%Y%m%d")
        end = f"{self.end_year}{self.end_month}{self.end_day}" #end_date.strftime("%Y%m%d")
        range_str = f"[{start}000000+TO+{end}000000]" # single digit dates must be 0-padded
        range_query = f"lastUpdatedDate:{range_str}"
        # range_query = f"submittedDate:{range_str}"

        base_url = "http://export.arxiv.org/api/query?"
        full_query = f"search_query=%28astro-ph.GA+OR+astro-ph.CO+OR+astro-ph.EP+OR+astro-ph.HE+OR+astro-ph.IM+OR+astro-ph.SR%29+AND+{range_query}&max_results={9999}"
        url = base_url + full_query

        response = urlopen(url).read()
        feed = feedparser.parse(response)

        api_results = []

        for e in feed.entries:
            arxiv_id = re.sub("v\d+$", "", e.id.split("/")[-1]) # e.id looks like 'http://arxiv.org/abs/2301.05335v2'
            api_results.append(arxiv_id)
            if (arxiv_id not in self.arxiv_ids) and (e.title not in self.all_titles):
                if self.rerun: 
                    self.arxiv_ids.append(arxiv_id)
                    self.arxiv_titles.append(e.title.strip("\n"))
                    self.all_titles.append(e.title.strip("\n"))
                elif not self.rerun and (arxiv_id not in self.existing_arxiv_ids):
                    self.arxiv_ids.append(arxiv_id)
                    self.arxiv_titles.append(e.title.strip("\n"))
                    self.all_titles.append(e.title.strip("\n"))

        return api_results

    def ads_api_call(self, date_query):
        """ Add doc string
        """

        # single digit dates must be 0-padded
        if date_query == "entdate":
            encoded_query = urlencode({"q": f"entdate:[{self.start_year}-{self.start_month}-{self.start_day} TO {self.end_year}-{self.end_month}-{self.end_day}]", "fl": "bibcode, identifier, title", "fq": "database:astronomy, property:article, property:(refereed OR eprint_openaccess), abs:(planet OR exoplanet OR extrasolar OR brown OR Jupiter OR Neptune OR TESS OR K2 OR Kepler OR TOI OR KOI OR OGLE OR KMT OR EPIC OR MOA)", "rows": 9999})
        elif date_query == "metadata_mtime":
            encoded_query = urlencode({"q": f"metadata_mtime:[{self.start_year}-{self.start_month}-{self.start_day}T00\:00\:00.000Z TO {self.end_year}-{self.end_month}-{self.end_day}T00\:00\:00.000Z]", "fl": "bibcode, identifier, title", "fq": "database:astronomy, property:article, property:(refereed OR eprint_openaccess), abs:(planet OR exoplanet OR extrasolar OR brown OR Jupiter OR Neptune OR TESS OR K2 OR Kepler OR TOI OR KOI OR OGLE OR KMT OR EPIC OR MOA)", "rows": 9999})


        results = requests.get("https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query), headers={'Authorization': 'Bearer ' + self.ADS_DEV_KEY}).json()["response"]["docs"]

        if len(results) == 2000:
            print("Maximum number of ADS results exceeded")

        api_results = []

        for paper in results:
            identifiers = paper["identifier"]
            
            found_arxiv_id = False
            for id_ in identifiers:
                if id_[:6] == "arXiv:": # in ads, arxiv ids look like 'arXiv:2302.07880'; this might not work as intended if a paper has multiple arxiv ids but i dont think thats possible
                    found_arxiv_id = True
                    api_results.append(id_[6:])
                    if (id_[6:] not in self.arxiv_ids) and (paper["title"] not in self.all_titles):
                        if self.rerun:
                            self.arxiv_ids.append(id_[6:])
                            self.arxiv_titles.append(paper["title"][0])
                            self.all_titles.append(paper["title"][0])
                        elif not self.rerun and (id_[6:] not in self.existing_arxiv_ids):
                            self.arxiv_ids.append(id_[6:])
                            self.arxiv_titles.append(paper["title"][0])
                            self.all_titles.append(paper["title"][0])
                    
            if not found_arxiv_id:
                if (paper["bibcode"] not in self.ads_bibs) and (paper["title"] not in self.all_titles):
                    if self.rerun:
                        self.ads_bibs.append(paper["bibcode"])
                        self.ads_titles.append(paper["title"][0])
                        self.all_titles.append(paper["title"][0])
                    elif not self.rerun and (paper["bibcode"] not in self.existing_ads_bibs):
                        self.ads_bibs.append(paper["bibcode"])
                        self.ads_titles.append(paper["title"][0])
                        self.all_titles.append(paper["title"][0])

            api_results.append(paper["bibcode"])

        return api_results

    def download_pdfs(self, source, ids, titles):
        """ Add doc string
        """
        for id_, title in zip(ids, titles):
            # print(id_)
            if source == "arxiv":
                headers = {
                'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36',
                }
                pdf_link1 = f"http://export.arxiv.org/pdf/{id_}"
                pdf_link2 = f"http://arxiv.org/pdf/{id_}"
            elif source == "ads":
                headers = {
                'Authorization': f'Bearer {self.ADS_DEV_KEY}',
                'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36',
                }
                pdf_link1 = f"https://ui.adsabs.harvard.edu/link_gateway/{id_}/pub_pdf"
                pdf_link2 = f"https://ui.adsabs.harvard.edu/link_gateway/{id_}/eprint_pdf"

            session = requests.Session()
            retry = Retry(connect=3, backoff_factor=10)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            filename = f'{id_}.pdf'.replace("/", "")
            full_filename = self.sub_directory + filename

            # i think i hit a rate limit after downloading ~2000 pdfs

            # make sure pdfs are not downloaded twice
            files = os.listdir(self.sub_directory) # could go outside loop but if i keep it in it can triple check that a duplicate file isnt downloaded
            file_size = None

            if filename in files: # file was already downloaded
                file_size = os.path.getsize(full_filename)
                if file_size > 40000: # file was downloaded successfully
                    self.downloaded_ids.append(id_)
                    self.downloaded_titles.append(title)
                    continue
                else:
                    pass

            # download the pdfs
            try:
                response = session.get(pdf_link1, headers=headers)
                with open(full_filename, 'wb') as f:
                    f.write(response.content)
            except:
                pass

            if os.path.isfile(full_filename):
                file_size = os.path.getsize(full_filename) 
                # sometimes calls to the arxiv api using "export" in the url fail but succeed if "export" is not in the url
                # in this case a corrupt pdf with a small file size is downloaded instead
                if file_size < 40000:
                    try:
                        response = session.get(pdf_link2, headers=headers)
                        with open(full_filename, 'wb') as f:
                            f.write(response.content)
                    except:
                        pass
            else:
                try:
                    response = session.get(pdf_link2, headers=headers)
                    with open(full_filename, 'wb') as f:
                        f.write(response.content)
                except:
                    pass
                if os.path.isfile(full_filename):
                    file_size = os.path.getsize(full_filename)

            # final check to see if paper downloaded successfully 
            if file_size is not None:
                if file_size > 40000:
                    self.downloaded_ids.append(id_)
                    self.downloaded_titles.append(title)
                else:
                    self.bad_ids.append(id_)
                    self.bad_titles.append(title)
            else:
                self.bad_ids.append(id_)
                self.bad_titles.append(title)

                # bar.next()

    def convert_pdfs_to_text(self):
        """ Add doc string
        """
        paper_texts = []
        for id_, title in zip(self.downloaded_ids, self.downloaded_titles): 
            # filename could be arxiv id or ads bibcode
            filename = f'{self.sub_directory}{id_.replace("/", "")}.pdf'

            try:
                reader = PyPDF2.PdfReader(filename) # EOF error when file is corrupt
            except:
                if id_ not in self.bad_ids:
                    self.bad_ids.append(id_)
                    self.bad_titles.append(title)
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
                    if id_ not in self.bad_ids:
                        self.bad_ids.append(id_)
                        self.bad_titles.append(title)
                    continue

        # i dont think this is needed anymore but ill keep it here just in case
        #                 if text in filename: # triggers when the pdf contains a blank page (4 in feb set), change in future versions of Convert PDFs to Test notebook (ie future re-training)
        #                     paper_good = False
        #                     if filename not in bad_papers:
        #                         bad_papers.append(filename)
        #                     continue

                paper_text += text

            if len(paper_text) > 1000: # make sure text was actually read
                paper_text = paper_text.replace('\n', '').replace("\r", "")
                for punct in string.punctuation:
                    paper_text = paper_text.replace(punct, "")
                self.converted_ids.append(id_)
                self.converted_titles.append(title)
                paper_texts.append(paper_text)
            else:
                paper_good = False
                if id_ not in self.bad_ids:
                    self.bad_ids.append(id_)
                    self.bad_titles.append(title)
                continue

        return paper_texts

    def doc2vec(self, paper_texts):
        """ Add doc string
        """
        # doc2vec time
        def tokenize_text(text):
            stopWords = set(stopwords.words('english'))

            tokens = []
            for sentence in nltk.sent_tokenize(text):
                for word in nltk.word_tokenize(sentence):
                    if len(word) < 2 or word in stopWords:
                        continue
                    tokens.append(word.lower())
            return tokens

        doc2vec_path=".//models//doc2vec//model_dbow_vs_600_n_5_s_0.bin"
        doc2vec_model = Doc2Vec.load(doc2vec_path)
        X = [doc2vec_model.infer_vector(tokenize_text(paper_text)) for paper_text in paper_texts] # this takes a long time

        return X

    def classifier(self, X):
        """ Add doc string
        """
        # classifier
        classifier_path=".//models//classifiers//lr.sav"
        classifier = pickle.load(open(classifier_path, 'rb'))

        probs = classifier.predict_proba(X) # need this if inferring one at a time: X.reshape(1, -1) also [0] at end
        ea_probs = [prob[1] for prob in probs]

        return ea_probs

    def return_results(self, ea_probs):
        """ Add doc string
        """
        # bad_ids should be printed out alongside titles
        # sort results
        sorted_ea_probs = sorted(ea_probs, reverse=True)
        sorted_ids = [paper for prob, paper in sorted(zip(ea_probs, self.converted_ids), reverse=True)]
        sorted_titles = [title.replace("\n", "") for prob, title in sorted(zip(ea_probs, self.converted_titles), reverse=True)]

        sorted_links = []
        for id_ in sorted_ids:
            if re.search("^\d+\.\d+$", id_) is not None:
                sorted_links.append(f"https://arxiv.org/abs/{id_}")
            else:
                sorted_links.append(f"https://ui.adsabs.harvard.edu/abs/{id_}/abstract")

        sorted_html_links = []
        for link, title in zip(sorted_links, sorted_titles):
            html_link = f"<a href='{link}'>{title}</a>"
            sorted_html_links.append(html_link)

        bad_paper_html_links = []
        for title, id_ in zip(self.bad_titles, self.bad_ids):
            if re.search("^\d+\.\d+$", id_) is not None:
                link = f"https://arxiv.org/abs/{id_}"
                html_link = f"<a href='{link}'>{title}</a>"
                bad_paper_html_links.append(f"NA | {html_link}\n")
            else:
                link = f"https://ui.adsabs.harvard.edu/abs/{id_}/abstract"
                html_link = f"<a href='{link}'>{title}</a>"
                bad_paper_html_links.append(f"NA | {html_link}\n")

        if self.output_type == "text":
            # store these for future retraining
            with open(f'{self.root_directory}all_existing_results.txt', 'a', encoding="utf-8") as f:
                for id_, prob in zip(sorted_ids, sorted_ea_probs):
                    if id_ not in self.existing_arxiv_ids and id_ not in self.existing_ads_bibs:
                        f.write(f"{id_},{prob}\n")

            # output to user
            with open(f'{self.sub_directory}results.txt', 'w', encoding="utf-8") as f:
                for title, prob, html_link in zip(sorted_titles, sorted_ea_probs, sorted_html_links):
                    # f.write(f"{round(prob,3)} | {title} | {link}\n")
                    f.write(f"{round(prob,3)} | {html_link}\n")
                for link in bad_paper_html_links:
                    f.write(f"NA | {link}\n")

            # with open(f'{self.sub_directory}bad_ids.txt', 'w', encoding="utf-8") as f:
            #     for id_ in self.bad_ids:
            #         f.write(f"{id_}\n")

        elif self.output_type == 'html':
            df1 = pd.DataFrame({"EA Prob":sorted_ea_probs, "Paper Title/Link":sorted_html_links})
            df2 = pd.DataFrame({"EA Prob": np.repeat("NA", len(bad_paper_html_links)), "Paper Title/Link":bad_paper_html_links})
            df = pd.concat([df1, df2])
            
            return df.to_html()
        
        elif self.output_type == 'struct':
            struct_string = f'[struct stat="OK", num_papers="{len(sorted_ea_probs)}", references_json="/work/TMP_FJk12o_17209/TransitView/2023.05.18_14.22.23_013043/references.json"]'

            return struct_string

def main():
    """ Add doc string
    """
    # python api_pipeline.py --startdate 230521 --enddate 230522 --verbose 1
    parser = argparse.ArgumentParser(description="Input start & end dates")

    parser.add_argument("--startdate",
                            metavar="Start Date",
                            type=str,
                            nargs=1,
                            help="Set start date like 'YYMMDD'")

    parser.add_argument("--enddate",
                            metavar="End Date",
                            type=str,
                            nargs=1,
                            help="Set end date like 'YYMMDD'")

    parser.add_argument("--output_type",
                        metavar="Output Type",
                        type=str,
                        nargs=1,
                        default='text',
                        help="Whether to output results as 'html', 'struct', or 'text'")
    
    parser.add_argument("--rerun",
                            metavar="Rerun Mode",
                            type=bool,
                            nargs=1,
                            default=True,
                            help="Run converting/inference on scraped papers that were previously predicted")

    parser.add_argument("--debug",
                            metavar="Debug Mode",
                            type=bool,
                            nargs=1,
                            default=False,
                            help="Turn off scraping/converting/inference to test outputs")

    parser.add_argument("--verbose",
                            metavar="Verbose Mode",
                            type=bool,
                            nargs=1,
                            default=False,
                            help="Print what's happening ...",
                            dest="verbose")

    args = parser.parse_args()
    print(args)

    # Parse dates but keep as strings
    start_date = args.startdate[0]
    start_year = start_date[0:2]
    start_year = "20" + start_year
    start_month = start_date[2:4]
    start_day = start_date[4:6]

    end_date = args.enddate[0]
    end_year = end_date[0:2]
    end_year = "20" + end_year
    end_month = end_date[2:4]
    end_day = end_date[4:6]

    if type(args.rerun) == list:
        rerun = args.rerun[0]
    else:
        rerun = args.rerun

    if type(args.debug) == list:
        debug = args.debug[0]
    else:
        debug = args.debug

    if type(args.output_type) == list:
        output_type = args.output_type[0]
    else:
        output_type = args.output_type

    if type(args.verbose) == list:
        verbose = args.verbose[0]
    else:
        verbose = args.verbose


    if not debug:
        if verbose:
            print(f"Range: {start_year}/{start_month}/{start_day} - {end_year}/{end_month}/{end_day}")

        ml_clf = ml_doc_classifier(start_day, start_month, start_year, end_day, end_month, end_year, output_type=output_type, rerun=rerun)

        all_start = time.time()

        ml_clf.arxiv_api_call()
        if verbose:
            print("Number of results after first arXiv query:", len(ml_clf.arxiv_ids))

        ml_clf.ads_api_call("entdate")
        if verbose:
            print("Number of results after first ADS query:", len(ml_clf.arxiv_ids)+len(ml_clf.ads_bibs))

        ml_clf.ads_api_call("metadata_mtime")
        if verbose:
            print("Number of results after second ADS query:", len(ml_clf.arxiv_ids)+len(ml_clf.ads_bibs))

        download_start = time.time()
        ml_clf.download_pdfs("arxiv", ml_clf.arxiv_ids, ml_clf.arxiv_titles)
        ml_clf.download_pdfs("ads", ml_clf.ads_bibs, ml_clf.ads_titles)
        download_duration = (time.time() - download_start)/60 
        if verbose:
            print("download duration (mins):", download_duration)
            print("Number of PDFs successfully downloaded:", len(ml_clf.downloaded_ids))

        convert_start = time.time()
        paper_texts = ml_clf.convert_pdfs_to_text()
        convert_duration = (time.time() - convert_start)/60
        if verbose:
            print("convert duration (mins):", convert_duration)
            print("Number of PDFs successfully converted to text:", len(ml_clf.converted_ids))

        if len(ml_clf.converted_ids) > 0:
            doc2vec_start = time.time()
            X = ml_clf.doc2vec(paper_texts)
            doc2vec_duration = (time.time() - doc2vec_start)/60
            if verbose:
                print("doc2vec duration (mins):", doc2vec_duration)

            inference_start = time.time()
            ea_probs = ml_clf.classifier(X)
            inference_duration = (time.time() - inference_start)/60
            if verbose:
                print("inference duration (mins):", inference_duration)
            
            output = ml_clf.return_results(ea_probs)

        all_duration = (time.time() - all_start)/60
        print("total duration (mins):", all_duration)

    else: # debug

        sorted_ids = [1,2,3]
        sorted_titles = ["a", "b", "c"]
        sorted_links = ["a1", "b1", "c1"]
        sorted_ea_probs = [0.99, 0.99, 0.99]

        if output_type == 'html':
            df = pd.DataFrame({"Paper ID":sorted_ids, "Paper Title":sorted_titles, "Paper Link":sorted_links, "EA Prob":sorted_ea_probs})
            output = df.to_html()
            print(output)

        elif output_type == 'struct':
            output = '[struct stat="OK", query_type="GET_REFERENCES", num_rows="17", references_json="/work/TMP_FJk12o_17209/TransitView/2023.05.18_14.22.23_013043/references.json"]'
            print(output)


    return output


if __name__ == "__main__":
    main()

# python api_pipeline.py --startdate 230301 --enddate 230302 --verbose 1
# Number of results after first arXiv query: 82
# Number of results after first ADS query: 84
# Number of results after second ADS query: 235
# download duration (mins): 13.801348316669465
# Number of PDFs successfully downloaded: 228
# convert duration (mins): 13.493611121177674
# Number of PDFs successfully converted to text: 224
# doc2vec duration (mins): 1.451873509089152
# inference duration (mins): 0.00011666218439737956
# 28.8

# python api_pipeline.py --startdate 230402 --enddate 230403 --verbose 1
# Number of results after first arXiv query: 24
# Number of results after first ADS query: 33
# Number of results after second ADS query: 296
# download duration (mins): 16.318849154313405
# Number of PDFs successfully downloaded: 272
# FloatObject (b'0.00-42771595') invalid; use 0.0 instead
# convert duration (mins): 22.463578081130983
# Number of PDFs successfully converted to text: 271
# doc2vec duration (mins): 1.91994602282842
# inference duration (mins): 0.00024393796920776367
# total duration (mins): 40.76192534764608

# python api_pipeline.py --startdate 230320 --enddate 230321 --verbose 1
# Number of results after first arXiv query: 85
# Number of results after first ADS query: 87
# Number of results after second ADS query: 690
# download duration (mins): 7.026944736639659
# Number of PDFs successfully downloaded: 551
#  impossible to decode XFormObject /Im101
# Multiple definitions in dictionary at byte 0x2b8cbc for key /Rotate
# Multiple definitions in dictionary at byte 0x2b8d86 for key /Rotate
# Multiple definitions in dictionary at byte 0x2b8e50 for key /Rotate
# Multiple definitions in dictionary at byte 0x2b8f1a for key /Rotate
# Multiple definitions in dictionary at byte 0x2b8fe4 for key /Rotate
# Multiple definitions in dictionary at byte 0x2b90ae for key /Rotate
# FloatObject (b'0.00000000000-11368684') invalid; use 0.0 instead
# FloatObject (b'0.00000000000-11368684') invalid; use 0.0 instead
# convert duration (mins): 37.67474456230799
# Number of PDFs successfully converted to text: 549
# doc2vec duration (mins): 3.6725465099016827
# inference duration (mins): 0.0005276242891947429
# total duration (mins): 48.44829435348511