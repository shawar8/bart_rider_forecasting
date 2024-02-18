import pandas as pd
import numpy as np
import os
import holidays
import nfl_data_py as nfl
from variables import config
from minio import Minio
import pickle
import logging
logging.basicConfig(level=logging.INFO)


def get_temporal_features(date):
    year = date.year
    month = date.month
    day = date.day
    which_day = date.day_of_week
    is_weekend = 1 if which_day in ['Saturday', 'Sunday'] else 0
    return year, month, day, which_day, is_weekend


def get_days_count(df, nm, date_list):
    idx_req = list(df.loc[df['date'].isin(date_list)].index)
    max_date = df.loc[max(idx_req), 'date']
    min_date = df.loc[min(idx_req), 'date']
    df.loc[idx_req, f'is_{nm}'] = 1
    df.loc[pd.isnull(df[f'is_{nm}']), f'is_{nm}'] = 0

    for idx in idx_req:
        curr_date = df.loc[idx, 'date']
        # next_dt = list(filter(lambda x: x > curr_date, date_list))[0]
        next_dt = list(filter(lambda x: x > curr_date, date_list))
        if not next_dt:
            next_dt = pd.to_datetime(f'{curr_date.year}-09-15')
        else:
            next_dt = next_dt[0]
        prev_dt = list(filter(lambda x: x < curr_date, date_list))[-1]
        # print (str(curr_date.date()), ' || ', next_hol, ' || ', prev_hol)
        df.loc[idx, f'next_{nm}'] = next_dt
        df.loc[idx, f'prev_{nm}'] = prev_dt

    df[f'next_{nm}'] = df[f'next_{nm}'].ffill()
    df[f'prev_{nm}'] = df[f'prev_{nm}'].backfill()

    df.loc[pd.isnull(df[f'prev_{nm}']), f'prev_{nm}'] = max_date
    df.loc[pd.isnull(df[f'next_{nm}']), f'next_{nm}'] = min_date

    df[f'days_since_{nm}'] = (df['date'] - df[f'prev_{nm}']).apply(lambda x: x.days)
    df[f'days_to_next_{nm}'] = (df[f'next_{nm}'] - df['date']).apply(lambda x: x.days)
    df.loc[df[f'is_{nm}'] == 1, [f'days_since_{nm}', f'days_to_next_{nm}']] = [0,0]

    return df


def get_num_days(dt, nfl_dates):
    req_dt = max(filter(lambda x: x < dt, nfl_dates))
    td = (dt - req_dt).days
    return td

def dump_data(data, dest_filname: str, source_file: str, is_index:bool=False):
    logging.info('Creating Client')
    minio_client = Minio(config["minio_endpoint"], secure= False,
                   access_key=config["minio_username"],
                   secret_key=config["minio_password"])
    logging.info('Created Client')
    bucket_name=config['dest_bucket']
    logging.info('Checking to see if bucket exists')
    found=minio_client.bucket_exists(bucket_name)
    if not found:
        minio_client.make_bucket(bucket_name)
        logging.info('Bucket Created !')
    else:
        logging.info('Bucket Exists!')
    logging.info('saving df to storage')
    ext = source_file[-3:]
    if ext == 'csv':
        data.to_csv(source_file, index= is_index)
    elif ext == 'pkl':
        with open(source_file, 'wb') as f:
            pickle.dump(data, f)
    minio_client.fput_object(
        bucket_name, dest_filname, source_file
    )
    os.remove(source_file)


def get_data_from_minio(filename):
    logging.info('Creating Client')
    minio_client = Minio(config["minio_endpoint"], secure=False,
                         access_key=config["minio_username"],
                         secret_key=config["minio_password"])
    logging.info('Created Client')
    bucket_name = config['dest_bucket']
    logging.info('Downloading Data')
    ext = filename[-3:]
    minio_client.fget_object(bucket_name, filename, filename)
    if ext == 'csv':
        data = pd.read_csv(filename)
    elif ext == 'pkl':
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    os.remove(filename)
    return data
