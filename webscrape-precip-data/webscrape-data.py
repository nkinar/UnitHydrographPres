#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
from constants import *
import json
from collections import OrderedDict
import datetime
from datetime import timedelta
from datetime import datetime
import pytz
import pprint
from pytz import timezone
import time
import pickle
import copy
import glob
import os
import pandas as pd
import sys


def get_station_page_at_link(url):
    """
    Obtain the station info at the given link
    :param sn:  as the name of the station
    :return: None if the page could not be found
    """
    page = requests.get(url)
    if page.status_code != 200:
        return None
    soup = BeautifulSoup(page.text, "lxml")
    return soup.prettify()


def extract_json_dict_from_page(text):
    """
    Extract the initial JSON
    :param text:
    :return:
    """
    search_first = '<script id="app-root-state" type="application/json">'
    search_second = '</script>'
    removestr = '&q;'

    find_len = len(search_first)
    find_first = text.find(search_first)
    find_second = text.find(search_second)

    substr = text[find_first+find_len+1:]
    find_second = substr.find(search_second)
    finalstr = substr[:find_second].strip()
    finalstr = finalstr.replace(removestr, "\"")

    if not finalstr.startswith('{'):
        return EMPTY_STRING
    if not finalstr.endswith('}'):
        return EMPTY_STRING

    j = json.loads(finalstr, object_pairs_hook=OrderedDict)
    env = j["wu-next-state-key"]
    keys = []
    n = []
    for k, v in env.items():
        val = v.get("value", None)
        if val is None:
            continue
        ob = val.get("observations", None)
        if ob is not None:
            keys.append(k)
            n.append(len(ob))
    dnew = {}
    if len(n) == 0:
        return dnew
    max_value = max(n)
    print('obs: {}'.format(max_value))
    if max_value == 0:  # don't download missing data
        return dnew
    max_index = n.index(max_value)
    key_max = keys[max_index]
    obj = env[key_max]["value"]["observations"]
    for o in obj:
        tz = o.get("tz")
        id = o.get("stationID")
        lat = o.get("lat")
        lng = o.get("lon")
        dat = o.get("imperial")
        e = int(o.get("epoch"))
        if tz is None:
            t = datetime.fromtimestamp(e)
        else:
            t = datetime.fromtimestamp(e, timezone(tz))
        dnew[t] = dat
        dnew[t]["tz"] = tz
        dnew[t]["id"] = id
        dnew[t]["lat"] = lat
        dnew[t]["lon"] = lng
        dnew[t]["epoch"] = e
    return dnew


def download_in_range(station_name, start_date, end_date):
    """
    Download bewteen range
    :param station_name:    as the station name from a file
    :param start_date:      as the starting date in datetime(year, month, day) format
    :param end_date:        as the ending date in datetime(year, month, day) format
    :return:
    """
    base_url = 'https://www.wunderground.com/dashboard/pws/' + station_name + '/table/'
    d = copy.deepcopy(start_date)
    station_dict = {}
    print('Station Name: ', station_name)
    while d < end_date:
        year = d.year
        month = d.month
        day = d.day
        daystr = "{}-{:02}-{:02}".format(year, month, day)
        url = base_url + daystr + '/' + daystr + '/daily'
        print(url)
        page = get_station_page_at_link(url)
        if page is None:
            print('Could not access link')
            continue
        month_dict = extract_json_dict_from_page(page)
        if not month_dict:  # return if there is no data
            break
        station_dict.update(month_dict)
        time.sleep(WAIT_TIME_DOWNLOAD)
        d += timedelta(days=1)
    return station_dict


def download_from_filelist(fn, start_date, end_date):
    lines = None
    with open(fn, "r") as f:
        lines = f.readlines()
    if lines is None:
        return
    n = len(lines)
    cnt = 1
    for line in lines:
        if not line:
            continue
        station_name = line.strip()
        print("[{}/{}] {}".format(cnt, n, station_name))
        d = download_in_range(station_name, start_date, end_date)
        if d:
            with open(station_name + '.pickle', 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        cnt += 1
    # DONE


def download_data():
    fn = 'stations.txt'
    start_date = datetime(2021, 5, 20)  # year, month, day
    end_date = datetime(2021, 6, 7)     # year, month, day
    download_from_filelist(fn, start_date, end_date)


def get_all_files_in_dir_with_ext(dir):
    os.chdir(dir)
    out = []
    for file in glob.glob("*.pickle"):
        out.append(file)
    return out


def assemble_tables():
    files = get_all_files_in_dir_with_ext(DOWNLOAD_DIR)
    dataframe_dict = {}
    print('Reading and appending')
    for file in files:
        print(file)
        table = pd.read_pickle(DOWNLOAD_DIR + file)
        df = pd.DataFrame.from_dict(table, orient='index')
        obsname = file.replace('.pickle', '')
        dataframe_dict[obsname] = df
    print('done')
    with open(PRECIP_RAW_FN, 'wb') as handle:
        pickle.dump(dataframe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done writing to pickle')


def main():
    download_data()
    assemble_tables()


if __name__ == '__main__':
    main()



"""
Index(['tempHigh', 'tempLow', 'tempAvg', 'windspeedHigh', 'windspeedLow',
       'windspeedAvg', 'windgustHigh', 'windgustLow', 'windgustAvg',
       'dewptHigh', 'dewptLow', 'dewptAvg', 'windchillHigh', 'windchillLow',
       'windchillAvg', 'heatindexHigh', 'heatindexLow', 'heatindexAvg',
       'pressureMax', 'pressureMin', 'pressureTrend', 'precipRate',
       'precipTotal', 'tz', 'id', 'lat', 'lon', 'epoch'],
      dtype='object')
"""



