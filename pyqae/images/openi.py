"""
A simple interface for pulling data from the NIH OpenI site

Getting Started
from pyqae.images.openi import pull_collection_as_df, LUNG_STUDY_COLLECTON
lung_study_df = pull_collection_as_df(LUNG_STUDY_COLLECTON, 100, verbose = False)
"""

import json
import requests

import pandas as pd

OPENI_IMG_URL = "https://openi.nlm.nih.gov"
REST_BASE_URL = "https://openi.nlm.nih.gov/retrieve.php?coll={collection}&it=xg&m={m}&n={n}&lic=byncnd"
LUNG_STUDY_COLLECTON = "cxr"

def _proc_req(cxn_str, m, count = 100):
    response = requests.get(REST_BASE_URL.format(m=m, n=m + count, collection=cxn_str))
    if response.ok:
        c_out_json = json.loads(response.content.decode("utf-8"))
        return c_out_json['count'], c_out_json['total'], c_out_json['list']
    return 0, -1, None


def _fetch_db(cxn_str, max_results = None, verbose = True):
    _, total, _ = _proc_req(cxn_str, 1, 2)
    print('Total Images', total)
    all_results = []
    cur_m = 1
    step_size = 50
    cnt = 1
    while cnt > 0:
        cnt, ctot, lreq = _proc_req(cxn_str, cur_m, step_size)
        if cnt >= 0: all_results += lreq
        cur_m += step_size
        if verbose: print(cur_m, cnt)
        if max_results is not None:
            if cur_m>max_results: break
    return all_results


def _db_to_df(all_results):
    summary_df = pd.DataFrame([{
                                   'uid': ie['uid'],
                                   'major': ";".join(ie['MeSH']['major']),
                                   'minor': ";".join(ie['MeSH']['minor']),
                                   'problem': ie['Problems'],
                                   'abstract': ie['abstract'],
                                   'caption': ie['image']['caption'],
                                   'image_id': ie['image']['id'],
                                   'url': ie['imgLarge']
                               } for ie in all_results])
    print('Rows:', len(summary_df))
    return summary_df

def pull_collection_as_df(cxn_str, max_results = None, verbose = False):
    res_list = _fetch_db(cxn_str, max_results = max_results, verbose = verbose)
    return _db_to_df(res_list)
    