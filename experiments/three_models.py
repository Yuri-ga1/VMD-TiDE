from __future__ import annotations
import json
import hashlib
import random
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch._utils
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TiDEModel
from darts.models.forecasting import torch_forecasting_model as tfm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD

MODES = [f"{name}_{i}" for name in ("IMF1", "IMF2", "IMF3", "R") for i in (1,2,3)]

TRAINING_PROTOCOL = {
    'version': 'val-loss-best-v1',
    'early_stopping': {'monitor': 'val_loss', 'patience': 10,
                       'min_delta': 0.0001, 'mode': 'min', 'verbose': True},
    'validation_vmd': 'prefix-through-fusion-end; unchanged-mode-order',
}

@dataclass
class Config:
    csv_path: str = "dataset.csv"
    output_dir: str = "models/VMD_TiDE_ABC_h7_train1954_test2009"
    target: str = "F10.7adj"
    start: str = "1954-01-02"
    train_end: str = "1986-08-31"
    fusion_end: str = "2008-12-31"
    test_end: str | None = None
    horizon: int = 7
    context: int = 126
    epochs: int = 50
    batch_size: int = 256
    stride: int = 27
    max_origins: int | None = None
    vmd_history: int | None = 2048
    random_seed: int = 42
    save_checkpoints: bool = True
    alpha: int = 2626
    hidden_size: int = 128
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    decoder_output_dim: int = 16
    temporal_width_past: int = 4
    temporal_width_future: int = 4
    temporal_decoder_hidden: int = 32
    dropout: float = 0.1

def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)
    required = {"Year", "Month", "Day", "SN", cfg.target}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    date = pd.to_datetime(df[["Year", "Month", "Day"]].rename(columns={"Year":"year", "Month":"month", "Day":"day"}), errors="coerce")
    df = df.assign(date=date).set_index("date").sort_index()
    df = df.loc[cfg.start:cfg.test_end, [cfg.target, "SN"]].copy()
    if df.index.duplicated().any():
        raise ValueError("Duplicate dates in CSV")
    df = df.asfreq("D").rename(columns={cfg.target:"f107", "SN":"sn"})
    df = df.apply(pd.to_numeric, errors="coerce")
    df.loc[df["sn"] < 0, "sn"] = np.nan
    df.loc[df["f107"] <= 0, "f107"] = np.nan
    joint = df[["f107", "sn"]].notna().all(axis=1)
    if not joint.any():
        raise ValueError("No joint F10.7/SN observations")
    last_joint = df.index[joint][-1]
    if cfg.test_end is None:
        cfg.test_end = last_joint.strftime('%Y-%m-%d')
    if pd.Timestamp(cfg.test_end) > last_joint:
        raise ValueError("test_end exceeds the last joint CSV observation")
    df = df.loc[:cfg.test_end]
    if pd.Timestamp(cfg.train_end) >= pd.Timestamp(cfg.fusion_end) or pd.Timestamp(cfg.fusion_end) >= pd.Timestamp(cfg.test_end):
        raise ValueError("Require train_end < fusion_end < test_end")
    train = df.loc[:cfg.train_end]
    if train["f107"].notna().sum() < cfg.context + cfg.horizon + 20:
        raise ValueError("Not enough training observations")
    filled = df.copy()
    filled.loc[train.index, "f107"] = train["f107"].interpolate(method="time", limit=5, limit_area="inside")
    filled.loc[train.index, "sn"] = train["sn"].interpolate(method="time", limit=5, limit_area="inside")
    filled["f107"] = filled["f107"].ffill()
    filled["sn"] = filled["sn"].ffill()
    filled["f107_original"] = df["f107"]
    filled["sn_original"] = df["sn"]
    if filled[["f107", "sn"]].isna().any().any():
        raise ValueError("Missing leading values; change start date")
    return filled

def vmd12(values: np.ndarray, alpha=2626) -> np.ndarray:
    """Original two-stage 3-mode VMD and residual branch; 12 components."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 32:
        raise ValueError("Too few samples for VMD")
    x = values if n % 2 == 0 else np.r_[values, values[-1]]
    def split(y):
        u, _, _ = VMD(y, alpha, 0, 3, 0, 1, 1e-7)
        return np.asarray(u)[:, :len(y)]
    top = split(x)
    resid = x - top.sum(axis=0)
    second = [split(signal) for signal in [top[0], top[1], top[2], resid]]
    result = np.vstack(second).T[:n]
    if result.shape != (n,12):
        raise RuntimeError(f"Unexpected VMD output {result.shape}")
    return result.astype(np.float32)

def ts(values, index, name):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    index = pd.DatetimeIndex(index)
    if len(values) != len(index) or not np.isfinite(values).all():
        raise ValueError(f"Invalid values/index for {name}")
    return TimeSeries.from_times_and_values(index, values, columns=[name], freq="D")

def make_model(cfg: Config, name: str):
    EarlyStopping = tfm.pl.callbacks.EarlyStopping
    stopper = EarlyStopping(**TRAINING_PROTOCOL['early_stopping'])
    checkpoint_work_dir = Path(cfg.output_dir) / '.darts_checkpoints'
    if (checkpoint_work_dir / name).exists():
        checkpoint_work_dir = checkpoint_work_dir / f'retry-{uuid.uuid4().hex}'
    params = {
        'input_chunk_length':cfg.context,
        'output_chunk_length':cfg.horizon,
        'num_encoder_layers':cfg.num_encoder_layers,
        'num_decoder_layers':cfg.num_decoder_layers,
        'decoder_output_dim':cfg.decoder_output_dim,
        'hidden_size':cfg.hidden_size,
        'temporal_width_past':cfg.temporal_width_past,
        'temporal_width_future':cfg.temporal_width_future,
        'temporal_decoder_hidden':cfg.temporal_decoder_hidden,
        'dropout':cfg.dropout,
        'use_reversible_instance_norm':True,
        'use_layer_norm':False,
        'n_epochs':cfg.epochs,
        'batch_size':cfg.batch_size,
        'random_state':cfg.random_seed,
        'model_name':name,
        'save_checkpoints':True,
        'work_dir':str(checkpoint_work_dir),
        'nr_epochs_val_period':1,
        'force_reset':False,
        'pl_trainer_kwargs': {'accelerator':'gpu' if torch.cuda.is_available() else 'cpu', 'devices':1, 'enable_progress_bar':True, 'logger':False, 'callbacks':[stopper]},
    }
    return TiDEModel(**params)

def _future_cov(sn: np.ndarray, dates: pd.DatetimeIndex, horizon: int, name="sn_lag"):
    shifted = pd.Series(sn, index=dates).shift(horizon)
    return ts(shifted.to_numpy()[horizon:], dates[horizon:], name)

def _fingerprint(cfg):
    settings = asdict(cfg)
    for key in ('output_dir', 'csv_path', 'save_checkpoints'):
        settings.pop(key)
    settings['csv_sha256'] = hashlib.sha256(Path(cfg.csv_path).read_bytes()).hexdigest()
    settings['training_protocol'] = TRAINING_PROTOCOL
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

def _check_config(cfg, *, initialize=False, force=False):
    """Refuse legacy/mismatched artifacts before writing anything or training."""
    if cfg.test_end is None:
        load_data(cfg)
    out = Path(cfg.output_dir)
    manifest = out / 'training_fingerprint.txt'
    occupied = out.exists() and any(out.iterdir())
    digest = _fingerprint(cfg)
    if occupied and (not manifest.exists() or manifest.read_text().strip() != digest):
        raise RuntimeError('Existing results are legacy or incompatible with val-loss-best-v1. '
                           'They were preserved. Choose a NEW output_dir; no automatic retraining.')
    if force and occupied:
        raise RuntimeError('force_train cannot overwrite existing results. Choose a NEW output_dir.')
    if not initialize and not manifest.exists():
        raise FileNotFoundError(f'No compatible training manifest: {manifest}')
    if initialize:
        out.mkdir(parents=True, exist_ok=True)
        if not manifest.exists():
            manifest.write_text(digest)
            (out / 'config.json').write_text(json.dumps(
                {**asdict(cfg), 'training_protocol': TRAINING_PROTOCOL}, indent=2), encoding='utf-8')

def _component_complete(path):
    return all(p.is_file() for p in (path, Path(str(path) + '.ckpt'),
                                    path.with_suffix('.scaler.joblib'),
                                    path.with_suffix('.training.json')))

def _save_best(model, cfg, path, scaler):
    """Reload Darts' best val_loss checkpoint before exporting each component."""
    trainer = model.trainer
    checkpoint = trainer.checkpoint_callback
    if not checkpoint.best_model_path or checkpoint.best_model_score is None:
        raise RuntimeError('No best val_loss checkpoint; component will not be saved')
    best = type(model).load_from_checkpoint(
        model_name=model.model_name, work_dir=model.work_dir, best=True,
        map_location='cpu')
    state = torch.load(checkpoint.best_model_path, map_location='cpu', weights_only=False)
    best_epoch = int(state['epoch']) + 1
    stopped_epoch = int(trainer.current_epoch)
    metadata = {'training_protocol': TRAINING_PROTOCOL, 'best_epoch': best_epoch,
                'best_val_loss': float(checkpoint.best_model_score),
                'epochs_completed': int(model.epochs_trained),
                'trainer_stop_epoch': stopped_epoch, 'max_epochs': cfg.epochs}
    best.save(str(path))
    joblib.dump(scaler, path.with_suffix('.scaler.joblib'))
    path.with_suffix('.training.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'{model.model_name}: completed {metadata["epochs_completed"]}/{cfg.epochs} epochs; '
          f'Trainer stop epoch={stopped_epoch}; saved best epoch={best_epoch}, '
          f'val_loss={metadata["best_val_loss"]:.6g}')

def _mark_complete(out, variant, names):
    if all(_component_complete(out / 'weights' / variant / f'{name}.pt') for name in names):
        marker = out / f'{variant}_complete.json'
        if not marker.exists():
            marker.write_text(json.dumps(
                {'variant': variant, 'components': len(names), 'status': 'saved',
                 'training_protocol': TRAINING_PROTOCOL}, indent=2), encoding='utf-8')
        print(f'Variant {variant}: all {len(names)} components saved')

def _component_frequencies(values):
    """Diagnostic only: spectral centroids, without reordering/modifying VMD modes."""
    power = np.abs(np.fft.rfft(values - values.mean(axis=0), axis=0)) ** 2
    return (np.fft.rfftfreq(len(values)) @ power / np.maximum(power.sum(axis=0), 1e-30)).tolist()

def train_all(cfg: Config, df: pd.DataFrame, force=False):
    """Train A (12 TiDE), B (12 TiDE with lagged-SN future covariate), and SN (1 TiDE)."""
    out = Path(cfg.output_dir)
    _check_config(cfg, initialize=True, force=force)
    all_paths = [out / 'weights' / variant / f'{name}.pt'
                 for variant, names in (('A', MODES), ('B', MODES), ('SN', ['sn']))
                 for name in names]
    if all(_component_complete(path) for path in all_paths):
        for variant, names in (('A', MODES), ('B', MODES), ('SN', ['sn'])):
            _mark_complete(out, variant, names)
        return {(variant, path.stem): (str(path), str(path.with_suffix('.scaler.joblib')))
                for variant in ('A', 'B', 'SN') for path in all_paths if path.parent.name == variant}
    train_df = df.loc[:cfg.train_end]
    if len(train_df) < cfg.context + cfg.horizon + 30:
        raise ValueError("Training segment too short")
    components = vmd12(train_df.f107.to_numpy(), cfg.alpha)
    val_prefix = df.loc[:cfg.fusion_end]
    val_df = val_prefix.loc[pd.Timestamp(cfg.train_end) + pd.Timedelta(days=1):]
    if len(val_df) < cfg.context + cfg.horizon:
        raise ValueError('Validation segment too short')
    val_components = vmd12(val_prefix.f107.to_numpy(), cfg.alpha)[-len(val_df):]
    if components.shape[1] != len(MODES) or val_components.shape[1] != len(MODES):
        raise ValueError('Train/validation VMD component mismatch')
    diagnostics = {'modes': MODES, 'train_end': cfg.train_end, 'validation_end': cfg.fusion_end,
                   'train_centroids': _component_frequencies(components),
                   'validation_centroids': _component_frequencies(val_components)}
    diagnostic_path = out / 'vmd_validation.json'
    if not diagnostic_path.exists():
        diagnostic_path.write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    print('VMD component centroids (cycles/day):\n', pd.DataFrame(
        {'train': diagnostics['train_centroids'], 'validation': diagnostics['validation_centroids']}, index=MODES))
    sn = train_df.sn.to_numpy(np.float32)
    dates = train_df.index
    sn_cov_values = pd.Series(sn, index=dates).shift(cfg.horizon)
    cut = cfg.horizon
    cov = ts(sn_cov_values.to_numpy()[cut:], dates[cut:], 'sn_lag')
    val_cov_values = df.loc[:cfg.fusion_end, 'sn'].shift(cfg.horizon).loc[val_df.index]
    val_cov = ts(val_cov_values.to_numpy(), val_df.index, 'sn_lag')
    saved = {}
    for variant in ('A','B'):
        for i, name in enumerate(MODES):
            path = out / 'weights' / variant / f'{name}.pt'
            scaler_path = path.with_suffix('.scaler.joblib')
            path.parent.mkdir(parents=True, exist_ok=True)
            if not force and _component_complete(path):
                saved[(variant, name)] = (str(path), str(scaler_path))
                continue
            if path.exists() or scaler_path.exists() or Path(str(path) + '.ckpt').exists():
                raise RuntimeError(f'Incomplete component preserved: {path}. Choose a new output_dir.')
            scaler = MinMaxScaler(feature_range=(-1,1))
            scaled = scaler.fit_transform(components[:,i:i+1]).ravel()
            series = ts(scaled, dates, name)
            model = make_model(cfg, f'{variant}_{name}_{cfg.horizon}')
            val_scaled = scaler.transform(val_components[:, i:i+1]).ravel()
            kwargs = {'series':series, 'val_series':ts(val_scaled, val_df.index, name)}
            if variant == 'B':
                kwargs['series'] = ts(scaled[cut:], dates[cut:], name)
                kwargs['future_covariates'] = cov
                kwargs['val_future_covariates'] = val_cov
            print(f'Training {variant} {name} ({i+1}/12)')
            model.fit(**kwargs, verbose=False)
            _save_best(model, cfg, path, scaler)
            saved[(variant, name)] = (str(path), str(scaler_path))
            del model
        _mark_complete(out, variant, MODES)
    path = out / 'weights' / 'SN' / 'sn.pt'
    scaler_path = path.with_suffix('.scaler.joblib')
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or not _component_complete(path):
        if path.exists() or scaler_path.exists() or Path(str(path) + '.ckpt').exists():
            raise RuntimeError(f'Incomplete component preserved: {path}. Choose a new output_dir.')
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaled = scaler.fit_transform(sn.reshape(-1,1)).ravel()
        model = make_model(cfg, f'SN_{cfg.horizon}')
        val_scaled = scaler.transform(val_df.sn.to_numpy(np.float32).reshape(-1, 1)).ravel()
        model.fit(series=ts(scaled, dates, 'sn'),
                  val_series=ts(val_scaled, val_df.index, 'sn'), verbose=False)
        _save_best(model, cfg, path, scaler)
        del model
    saved[('SN','sn')] = (str(path),str(scaler_path))
    _mark_complete(out, 'SN', ['sn'])
    return saved

def load_models(cfg: Config):
    out = Path(cfg.output_dir)
    _check_config(cfg)
    loaded={}
    for variant in ('A','B'):
        for name in MODES:
            path = out/'weights'/variant/f'{name}.pt'
            if not _component_complete(path): raise FileNotFoundError(f'Incomplete component: {path}')
            loaded[(variant,name)] = (TiDEModel.load(str(path), map_location='cpu'), joblib.load(path.with_suffix('.scaler.joblib')))
    path=out/'weights'/'SN'/'sn.pt'
    if not _component_complete(path): raise FileNotFoundError(f'Incomplete component: {path}')
    loaded[('SN','sn')] = (TiDEModel.load(str(path), map_location='cpu'),joblib.load(path.with_suffix('.scaler.joblib')))
    return loaded

def forecast_at(cfg: Config, df:pd.DataFrame, models:dict, origin:pd.Timestamp):
    """Predict a full horizon using data no later than `origin`."""
    origin = pd.Timestamp(origin)
    if origin not in df.index:
        raise ValueError('origin must be an available observation date')
    history = df.loc[:origin]
    comp_input = history.f107.to_numpy()
    if cfg.vmd_history is not None:
        comp_input = comp_input[-cfg.vmd_history:]
        history = history.tail(cfg.vmd_history)
    comps=vmd12(comp_input, cfg.alpha)
    dates = history.index
    horizon=cfg.horizon
    cov_date = pd.date_range(dates[0], origin + pd.Timedelta(days=horizon), freq='D')
    source_dates = cov_date - pd.Timedelta(days=horizon)
    observed_sn = df.loc[:origin, 'sn']
    cov_values = observed_sn.reindex(source_dates).to_numpy(dtype=np.float32)
    leading = source_dates < df.index.min()
    cov_values[leading] = observed_sn.iloc[0]
    if not np.isfinite(cov_values).all():
        raise ValueError('NaN in forecast SN covariates')
    cov = ts(cov_values, cov_date, 'sn_lag')
    forecast = {}
    for variant in ('A','B'):
        p = np.zeros(horizon, dtype=np.float64)
        for i, name in enumerate(MODES):
            m, scaler = models[(variant,name)]
            scaled = scaler.transform(comps[:,i:i+1]).ravel()
            series = ts(scaled, dates, name)
            kwargs = {'n':horizon, 'series':series}
            if variant=='B': kwargs['future_covariates']=cov
            yhat=m.predict(**kwargs, verbose=False).values().reshape(-1,1)
            p += scaler.inverse_transform(yhat).ravel()
        forecast[variant]=p
    m, scaler = models[('SN','sn')]
    sn_history = df.loc[:origin,'sn']
    sn_scaled = scaler.transform(sn_history.to_numpy().reshape(-1,1)).ravel()
    sn_pred = m.predict(n=horizon,series=ts(sn_scaled,sn_history.index,'sn'),verbose=False).values().reshape(-1,1)
    forecast['SN'] = scaler.inverse_transform(sn_pred).ravel()
    return forecast

def origins_for(df:pd.DataFrame, cfg:Config, part:str):
    if part == 'fusion':
        start = pd.Timestamp(cfg.train_end) + pd.Timedelta(days=1)
        end = pd.Timestamp(cfg.fusion_end)
    elif part=='test':
        start = pd.Timestamp(cfg.fusion_end) + pd.Timedelta(days=1)
        end = pd.Timestamp(cfg.test_end)
    else: raise ValueError(part)
    first = max(start-pd.Timedelta(days=1), df.index.min()+pd.Timedelta(days=cfg.context + cfg.horizon + 1))
    last = end-pd.Timedelta(days=cfg.horizon)
    dates = pd.date_range(first, last, freq=f'{cfg.stride}D')
    if cfg.max_origins is not None:
        dates = dates[:cfg.max_origins]
    return dates

def backtest(cfg: Config, df:pd.DataFrame, models:dict, part:str):
    rows=[]
    origins=origins_for(df,cfg,part)
    for count,origin in enumerate(origins,1):
        print(f'[{part}] {count}/{len(origins)} origin={origin.date()}')
        preds=forecast_at(cfg,df,models,origin)
        target_dates=pd.date_range(origin+pd.Timedelta(days=1), periods=cfg.horizon, freq='D')
        y=df['f107_original'].reindex(target_dates).to_numpy()
        for i,date in enumerate(target_dates):
            if not np.isfinite(y[i]): continue
            rows.append({'date':date,'origin':origin,'lead':i+1,'actual':float(y[i]),
                        'A':float(preds['A'][i]),'B':float(preds['B'][i]), 'SN_prediction':float(preds['SN'][i])})
    frame=pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f'No evaluable predictions for {part}')
    return frame

def merge_c(cfg:Config, fusion:pd.DataFrame, test:pd.DataFrame):
    Xf=fusion[['A','SN_prediction']].values
    Xt=test[['A','SN_prediction']].values
    out=Path(cfg.output_dir)
    _check_config(cfg)
    ridge_path = out / 'fusion_ridge.joblib'
    if ridge_path.exists():
        ridge = joblib.load(ridge_path)
    else:
        ridge=RidgeCV(alphas=np.logspace(-3, 5, 17))
        ridge.fit(Xf,fusion.actual.values)
        joblib.dump(ridge, ridge_path)
    fusion=fusion.copy();test=test.copy()
    fusion['C']=ridge.predict(Xf)
    test['C']=ridge.predict(Xt)
    marker = out / 'C_complete.json'
    if not marker.exists():
        marker.write_text(json.dumps(
            {'variant': 'C', 'status': 'saved', 'model': 'Ridge fusion',
             'training_protocol': TRAINING_PROTOCOL}, indent=2), encoding='utf-8')
    return fusion,test,ridge

def evaluate(frame:pd.DataFrame):
    rows=[]
    for key in ('A','B','C'):
        truth=frame.actual.values
        pred=frame[key].values
        rows.append({'model':key,'N':len(truth), 'MAE':mean_absolute_error(truth,pred),
                     'RMSE':np.sqrt(mean_squared_error(truth,pred)),
                     'R2':r2_score(truth,pred)})
    result=pd.DataFrame(rows).set_index('model')
    baseline=result.loc['A','MAE']
    result['MAE_improvement_vs_A_pct']=100*(baseline-result.MAE)/baseline
    return result

def plot_results(cfg:Config, test:pd.DataFrame, metrics:pd.DataFrame):
    out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True)
    def save_figure(fig, name):
        if not (out / name).exists():
            fig.savefig(out / name, dpi=160)
    fig,ax=plt.subplots(figsize=(13,5))
    ax.plot(test.date,test.actual,label='Observed F10.7',lw=1.4)
    for key in 'ABC':ax.plot(test.date,test[key],label=f'Model {key}',alpha=.8)
    ax.set(xlabel='Date',ylabel='F10.7 (sfu)',title=f'{cfg.horizon}-day forecasts on unseen test origins')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();save_figure(fig, 'predictions.png');plt.show()
    fig,ax=plt.subplots(figsize=(10,4))
    for key in 'ABC': ax.plot(test.date,np.abs(test.actual-test[key]),label=f'{key} absolute error',alpha=.7)
    ax.set(xlabel='Date',ylabel='Absolute error (sfu)',title='Absolute errors');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();save_figure(fig, 'errors.png');plt.show()
    fig,ax=plt.subplots(figsize=(8,4));metrics[['MAE','RMSE']].plot.bar(ax=ax,rot=0)
    ax.set(ylabel='sfu',title='Test errors (lower is better)');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save_figure(fig, 'metrics.png');plt.show()
    fig,ax=plt.subplots(figsize=(8,4));metrics['MAE_improvement_vs_A_pct'].plot.bar(ax=ax,rot=0)
    ax.axhline(0,color='black',lw=.8);ax.set(ylabel='%',title='MAE improvement versus A')
    fig.tight_layout();save_figure(fig, 'improvement.png');plt.show()

def run(cfg:Config, force_train=False):
    random.seed(cfg.random_seed);np.random.seed(cfg.random_seed)
    df=load_data(cfg)
    _check_config(cfg, initialize=True, force=force_train)
    out=Path(cfg.output_dir)
    if (out / 'metrics.csv').exists() and (out / 'test_predictions.csv').exists():
        required = [out / f'{v}_complete.json' for v in ('A', 'B', 'SN', 'C')]
        if not all(p.exists() for p in required):
            raise RuntimeError('Existing results lack completion markers; preserved without overwriting')
        print('Loading compatible completed results; no training or result overwrites')
        return pd.read_csv(out / 'metrics.csv', index_col='model'), pd.read_csv(
            out / 'test_predictions.csv', parse_dates=['date', 'origin'])
    print('Observed gaps in F10.7:',df.f107_original.isna().sum())
    print('Observed gaps in SN:',df.sn_original.isna().sum())
    print('Training all 3 variants (A and B = 12 TiDE each, C reuses A + 1 SN TiDE + Ridge)')
    train_all(cfg,df)
    models=load_models(cfg)
    def predictions(part):
        path = out / f'{part}_predictions_without_C.csv'
        if path.exists():
            return pd.read_csv(path, parse_dates=['date', 'origin'])
        frame = backtest(cfg, df, models, part)
        frame.to_csv(path, index=False)
        return frame
    fusion=predictions('fusion')
    test=predictions('test')
    fusion,test,ridge=merge_c(cfg,fusion,test)
    for name, frame in (('fusion_predictions.csv', fusion), ('test_predictions.csv', test)):
        if not (out / name).exists():
            frame.to_csv(out / name, index=False)
    metrics=evaluate(test)
    if not (out / 'metrics.csv').exists():
        metrics.to_csv(out/'metrics.csv')
    print('\nTEST METRICS\n',metrics.round(4).to_string())
    print('Fusion coefficients:',ridge.coef_,'intercept:',ridge.intercept_,'alpha:',ridge.alpha_)
    plot_results(cfg,test,metrics)
    return metrics,test
