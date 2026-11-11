from functions.utilites import check_new_real, log_message
from classes.ForecastModel import ForecastModel

if check_new_real():
    fm = ForecastModel(7)
    fm.prepare_data_decomposed()
    log_message('Data is updated')
else:
    log_message('No new data')







