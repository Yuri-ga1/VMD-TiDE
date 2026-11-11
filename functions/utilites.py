from sunpy.coordinates.sun import carrington_rotation_number as crn
import functions.db as db
import pandas as pd
import requests
from datetime import timedelta
from scipy.signal import savgol_filter
from datetime import datetime

def get_train_data(db_name='data3',smooth=0, cutoff = datetime.today().strftime('%Y-%m-%d')):
    df = db.select_rows(f"SELECT * from data WHERE date<='{cutoff}' order by date")
    df.date = df.date.astype('datetime64[ns]')
    df.index = df.date
    if smooth>0:
        df['real'] = savgol_filter(df['real'], smooth, 3)
    df = df[df.date<pd.to_datetime(cutoff)]
    return df

def convert_date_to_crn(date):
    return crn(date)

def convert_dates_to_cycle_day(dates):
    res = []
    for date in dates:
        res.append(convert_date_to_cycle_day(date))
    return res

def convert_dates_to_cycle_day_with_cycle_number(dates):
    res = []
    for date in dates:
        res.append(convert_date_to_cycle_day_with_cycle_number(date))
    return res

def convert_date_to_cycle_day(date):
    df =db.select_rows(f"select * from solar_cycles where starts<='{date}' and ends>'{date}'")
    df['number_of_day_in_cycle'] = df.apply(lambda row: (row['ends'] - row['starts']).days, axis=1)
    df['current_date'] = pd.to_datetime(date)
    df['current_date_number'] = df.apply(lambda row: (row['current_date'] - pd.Timestamp(row['starts'])).days, axis=1)
    df['solar_date_number'] = df.apply(lambda row: (row['current_date_number']/row['number_of_day_in_cycle']), axis=1)
    return df['solar_date_number'][0]

def convert_date_to_cycle_day_with_cycle_number(date):
    df =db.select_rows(f"select * from solar_cycles where starts<='{date}' and ends>'{date}'")
    df['number_of_day_in_cycle'] = df.apply(lambda row: (row['ends'] - row['starts']).days, axis=1)
    df['current_date'] = pd.to_datetime(date)
    df['current_date_number'] = df.apply(lambda row: (row['current_date'] - pd.Timestamp(row['starts'])).days, axis=1)
    df['solar_date_number'] = df.apply(lambda row: (row['current_date_number']/row['number_of_day_in_cycle']+row['name']), axis=1)
    return df['solar_date_number'][0]

def prepare_data_table():
    df = get_train_data()
    df.index = range(1, len(df) + 1)
    df['crn'] = df['date'].apply(convert_date_to_crn)
    df['cycle_day_number'] = df['date'].apply(convert_date_to_cycle_day)
    df['sdn'] = df['date'].apply(convert_date_to_cycle_day_with_cycle_number)
    return df


def get_request(date):
        url = ('https://services.simurg.space/swis/F107/get_data?'
            f'start_date={date}T00:00&end_date={date}T23:00&stage=nowcast')
        try:
            response = requests.get(url, timeout=0.5)
            response.raise_for_status()
            data = response.json()
            if len(data) == 0:
                return False
            else:
                return data[-1]['f']
        except (requests.exceptions.RequestException, StopIteration):
            return False

def check_new_real():
    date = db.get_row("SELECT * from data ORDER BY date DESC LIMIT 1").date
    next_day = date + timedelta(days=1)
    date = next_day.strftime('%Y-%m-%d')
    real = get_request(date)
    if real:
        crn = float(convert_date_to_crn(date)),
        cdn = convert_date_to_cycle_day(date)
        sdn = convert_date_to_cycle_day_with_cycle_number(date)
        db.update_row(f"INSERT INTO data (date,real,crn,cycle_day_number,sdn) VALUES ('{date}', '{real}', '{float(crn[0])}', '{cdn}', '{sdn}')")
        return True
    else:
        return False

def log_message(text):
    # Get current time formatted as YYYY-MM-DD HH:MM
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open('/home/user/projects/f107/regular_job.log', 'a') as file:
        file.write(f"{now} {text}\n")
