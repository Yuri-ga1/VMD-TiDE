import psycopg2
import psycopg2.extras
from dotenv import dotenv_values
from psycopg2 import sql
from contextlib import closing
import sqlalchemy
import os
import numpy as np
import pandas as pd
from psycopg2.extensions import register_adapter, AsIs

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

def addapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))

register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)
register_adapter(np.float32, addapt_numpy_float32)
register_adapter(np.int32, addapt_numpy_int32)
register_adapter(np.ndarray, addapt_numpy_array)

def get_row(query):
    config = dotenv_values("/home/user/projects/f107/.env")
    with closing(psycopg2.connect(dbname=config['DB_NAME'], user=config['DB_USER'],
                                  password=config['DB_PASSWORD'], host=config['DB_HOST'])) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor) as cursor:
            cursor.execute(sql.SQL('SET search_path TO "'+str(config['DB_SCHEMA'])+'"'))
            cursor.execute(query)
            row = cursor.fetchone()
    return row


def get_rows(query):
    config = dotenv_values("/home/user/projects/f107/.env")

    with closing(psycopg2.connect(dbname=config['DB_NAME'], user=config['DB_USER'],
                                  password=config['DB_PASSWORD'], host=config['DB_HOST'])) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor) as cursor:
            cursor.execute(sql.SQL('SET search_path TO "'+str(config['DB_SCHEMA'])+'"'))
            cursor.execute(query)
            row = cursor.fetchall()
    return row


def select_rows(query):
    config = dotenv_values("/home/user/projects/f107/.env")

    conn = psycopg2.connect("host='{}' dbname='{}' user={} password={}".format(
        config['DB_HOST'], config['DB_NAME'], config['DB_USER'], config['DB_PASSWORD']))
    dat = pd.read_sql_query('SET search_path TO "'+str(config['DB_SCHEMA'])+'"; '+query, conn)
    conn = None

    return dat


def update_row(query):
    config = dotenv_values("/home/user/projects/f107/.env")
    with closing(psycopg2.connect(dbname=config['DB_NAME'], user=config['DB_USER'],
                                  password=config['DB_PASSWORD'], host=config['DB_HOST'])) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor) as cursor:
            conn.autocommit = True
            cursor.execute(sql.SQL('SET search_path TO "'+str(config['DB_SCHEMA'])+'"'))
            cursor.execute(query)



def insert_df_rows(name, df):

    if len(df) > 0:
        config = dotenv_values("/home/user/projects/f107/.env")
        df_columns = list(df)
        # create (col1,col2,...)
        columns = ",".join(df_columns)

        # create VALUES('%s', '%s",...) one '%s' per column
        values = "VALUES({})".format(",".join(["%s" for _ in df_columns]))

        #create INSERT INTO table (columns) VALUES('%s',...)
        insert_stmt = "INSERT INTO {} ({}) {}".format(name,columns,values)

        with closing(psycopg2.connect(dbname=config['DB_NAME'], user=config['DB_USER'],
                                  password=config['DB_PASSWORD'], host=config['DB_HOST'])) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql.SQL('SET search_path TO "'+str(config['DB_SCHEMA'])+'"'))
                psycopg2.extras.execute_batch(cursor, insert_stmt, df.values)
                conn.commit()
                cursor.close()

def insert_df(name, df):
    config = dotenv_values("/home/user/projects/f107/.env")
    DB_USER = config["DB_USER"]
    DB_PASSWORD = config["DB_PASSWORD"]
    DB_HOST = config["DB_HOST"]
    DB_PORT = config["DB_PORT"]  # Default PostgreSQL port
    DB_NAME = config["DB_NAME"]
    DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    engine = sqlalchemy.create_engine(DB_URL,connect_args={'options': '-csearch_path={}'.format('f107')})
    # write the DataFrame to the database
    df.to_sql(name, engine, if_exists='replace', index=True)