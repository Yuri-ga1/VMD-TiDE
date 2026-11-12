import numpy as np
import pandas as pd
import logging
import warnings
from dotenv import dotenv_values
from darts import TimeSeries, concatenate
from darts.utils.callbacks import TFMProgressBar
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from sklearn.preprocessing import MinMaxScaler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from darts.metrics.metrics import mape, mae, rmse, smape
import functions.db as db
import joblib

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
# a period of 5 epochs (`patience`)
my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=0.0001,
    mode='min',
)

class TideModel:
    def __init__(
        self,
        df,
        context_length,
        prediction_length,
        model_name='adj',
        tuned='default',
        target = 'IMF1',
        mode = 'train',
        device='cuda',
        split_date='2008-12-01',
        past_covariates=['sdn'],
        future_covariates=['crn'],
        **kwargs
    ):
        self.split_date = split_date
        self.df = df
        self.mode = mode
        self.tuned = tuned
        self.device = device
        self.target = target
        self.past_covs = past_covariates
        self.future_covs = future_covariates
        self.model_name = model_name
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.work_dir = kwargs.get('work_dir')
        self.force_reset = kwargs.get('force_reset', True)
        self.extra_callbacks = kwargs.get('extra_callbacks', [])
        env_name = 'default' if self.tuned == 'default' else f"p{self.prediction_length}_{self.target}"
        params = dotenv_values(f"hyperparams/{env_name}.env")
        self.params = params

    def generate_torch_kwargs(self):
        return {
            "pl_trainer_kwargs": {
                "accelerator": self.device,
                "callbacks": [TFMProgressBar(enable_train_bar_only=True), *self.extra_callbacks],
            }
        }

    def get_model_name(self):
        return f'TIDE_h{self.context_length}_p{self.prediction_length}_{self.model_name}'

    def get_model_path(self):
        name = self.get_model_name()
        return f'models/{name}.pt'

    def prepare_time_series(self, df, target = 'IMF1'):

        filler = MissingValuesFiller()
        series = filler.transform(
            TimeSeries.from_dataframe(
                df, "date", [target]
            )
        ).astype(np.float32)

        if self.mode == "train":
            train, val = series.split_after(pd.Timestamp(self.split_date))
            scaler = Scaler(scaler=MinMaxScaler((-1,1)))
            train_scaled = scaler.fit_transform(train)
            val_scaled = scaler.transform(val)
            series_scaled = scaler.transform(series)
            joblib.dump(scaler, f'save/scaler_{target}.save')
        else:
            train = series
            val = series
            scaler = joblib.load(f'save/scaler_{target}.save')
            train_scaled = scaler.transform(train)
            val_scaled = scaler.transform(val)
            series_scaled = scaler.transform(series)

        return train_scaled, val_scaled, series_scaled, scaler

    def prepare_covs_ts(self, df, target):
        # df = db.select_rows("SELECT * from data_decomposed")
        scaler = MinMaxScaler((-1,1))
        df[target] = df[target] % 1
        arr = df[target]
        df = pd.DataFrame(arr)
        df[target] = np.cos(scaler.fit_transform(df)*np.pi)
        df['date'] = df.index
        series = TimeSeries.from_dataframe(
                    df, "date", [target]
                ).astype(np.float32)
        if self.mode == "train":
            train, val = series.split_after(pd.Timestamp(self.split_date))
        else:
            train = series
            val = series
        return train, val, series, scaler
    def prepare_covs(self, df, covs):
        for cov in covs:
            if 'train_past' not in locals():
                train_past, val_past, series_past, scaler = self.prepare_covs_ts(df, target=cov)
            else:
                train, val, series, scaler = self.prepare_covs_ts(df, target=cov)
                train_past = train_past.stack(train)
                val_past = val_past.stack(val)
                series_past = series_past.stack(series)
        return train_past, val_past, series_past

    def process_data(self):
        df = self.df
        params = self.params
        context_length=int(params['input_chunk_length']) if self.context_length==0 else self.context_length
        # find integer number of data sets
        limit = self.prediction_length*((df.shape[0]-context_length)//self.prediction_length)+context_length
        df = df.tail(limit)

        # train_past, val_past, series_past = self.prepare_covs(df, self.past_covs)
        # train_future, val_future, series_future = self.prepare_covs(df, self.future_covs)

        data = {
            'future': {},
            'past': {},
            'target': {},
        }
        # data['past']={
        #     'train': train_past,
        #     'val': val_past,
        #     'series': series_past,
        # }
        # data['future']={
        #     'train': train_future,
        #     'val': val_future,
        #     'series': series_future,
        # }
        train, val, series, scaler = self.prepare_time_series(df, target=self.target)
        data['target']={
            'train': train,
            'val': val,
            'series': series,
        }
        return data, scaler

    def prepare_model(self):
        model_name = self.get_model_name()
        params = self.params
        model = TiDEModel(
            input_chunk_length=int(params['input_chunk_length']) if self.context_length==0 else self.context_length,
            output_chunk_length=self.prediction_length,
            num_encoder_layers = int(params['num_encoder_layers']),
            num_decoder_layers = int(params['num_decoder_layers']),
            decoder_output_dim = int(params['decoder_output_dim']),
            hidden_size = int(params['hidden_size']),
            temporal_width_past= int(params['temporal_width_past']),
            temporal_width_future= int(params['temporal_width_future']),
            temporal_decoder_hidden= int(params['temporal_decoder_hidden']),
            dropout= float(params['dropout']),
            n_epochs=50,
            nr_epochs_val_period=1,
            use_reversible_instance_norm=True,
            use_layer_norm=False,
            batch_size=512,
            random_state=42,
            model_name=model_name,
            work_dir=self.work_dir,
            save_checkpoints=True,
            force_reset=self.force_reset,
            **self.generate_torch_kwargs(),
        )
        return model

    def load_best_model(self):
        name = self.get_model_name()
        return TiDEModel.load_from_checkpoint(model_name=name, work_dir=self.work_dir, best=True)

    def load_model(self):
        model = self.prepare_model()
        model.load_weights(self.get_model_path())
        return model

    def train(self, save_path=None, resume=False):
        data, scaler = self.process_data()
        if resume:
            model = TiDEModel.load_from_checkpoint(
                model_name=self.get_model_name(), work_dir=self.work_dir, best=False
            )
        else:
            model = self.prepare_model()
        model.fit(
            series=data['target']['train'],
            val_series=data['target']['val'],
            # past_covariates=data['past']['train'],
            # val_past_covariates=data['past']['val'],
            # future_covariates=data['future']['train'],
            # val_future_covariates=data['future']['val'],
            verbose=True,
            load_best=True,
        )
        model.save(save_path or self.get_model_path())
        return model, scaler

    def predict_all(self):
        model = self.load_model()
        data, scaler = self.process_data()
        pred_series = model.historical_forecasts(
            data["target"]["series"],
            # future_covariates = data["future"]["series"],
            # past_covariates = data["past"]["series"],
            forecast_horizon=self.prediction_length,
            # stride=self.prediction_length,
            stride=1,
            # last_points_only=False,
            last_points_only=True,
            retrain=False,
            verbose=True,
        )
        # pred_series = concatenate(pred_series)
        return scaler.inverse_transform(pred_series), scaler.inverse_transform(data["target"]["series"])

    def predict(self):
        model = self.load_model()
        data, scaler = self.process_data()
        pred_series = model.predict(
            self.prediction_length,
            data["target"]["series"].head(-self.prediction_length),
            # future_covariates = data["future"]["series"],
            # past_covariates = data["past"]["series"].head(-self.prediction_length),
        )
        return scaler.inverse_transform(pred_series)

    def evaluate_todb(self, cycle_number):
        predictions, answers = self.predict_all()
        db.add_row('comparison', [self.target, self.context_length, self.prediction_length, rmse(answers, predictions),mape(answers, predictions),mae(answers, predictions),smape(answers, predictions), cycle_number])
