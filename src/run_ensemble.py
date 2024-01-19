import argparse
import os
import torch
import time
import numpy as np
import pandas as pd
import shutil

from data_utils import g_node_col, g_date_col, process_cdc_truth_from_csse, process_cdc_loc, get_all_cdc_label, read_cdc_forecast
from base_task import load_json_from


# exp_dir_template = '../Exp_us_{}_{}'  # level, forecast_date
cdc_forecast_dir = '../covid19-forecast-hub/data-processed'


def load_exp_res(exp_dir, extra_columns=None):
    task_dirs = os.listdir(exp_dir)

    test_results = []
    for task_dir in task_dirs:
        task_items = task_dir.split('_')
        target, horizon, model, seed = task_items[:4]
        horizon = int(horizon)
        if len(task_items) == 4:
            seed = int(seed.lstrip('seed'))
        else:
            seed = '_'.join([seed.lstrip('seed')] + task_items[4:])

        if model == 'gbm':
            gbm_out = pd.read_csv(os.path.join(exp_dir, task_dir, 'test_out.csv'), parse_dates=[g_date_col])
            test_res = gbm_out[[g_date_col, g_node_col, 'pred', 'label']].fillna(0)
        else:
            try:
                nn_out = torch.load(os.path.join(exp_dir, task_dir, 'Output/test.out.cpt'))
            except:
                print(f'Warning: {os.path.join(exp_dir, task_dir)} is an incomplete task directory! ...skip...')
                continue
            if 'y_scale' in nn_out and nn_out['y_scale'] == 'linear':
                log_scale = False
            else:
                log_scale = True
            nn_pred = nn_out['pred'].reset_index(drop=False)
            nn_pred['pred'] = np.expm1(nn_pred['val']) if log_scale else nn_pred['val']
            nn_pred[g_date_col] = nn_out['dates']
            nn_pred[g_node_col] = nn_out['countries'] if 'countries' in nn_out else nn_out['nodes']
            nn_label = nn_out['label'].reset_index(drop=False)
            nn_label['label'] = np.expm1(nn_label['val']) if log_scale else nn_label['val']
            nn_label[g_date_col] = nn_out['dates']
            nn_label[g_node_col] = nn_out['countries'] if 'countries' in nn_out else nn_out['nodes']
            test_res = pd.merge(nn_pred, nn_label, on=[g_date_col, g_node_col])[[g_date_col, g_node_col, 'pred', 'label']]

            if extra_columns is not None:
                cfg = load_json_from(os.path.join(exp_dir, task_dir, 'config.json'))
                for extra_col in extra_columns:
                    if extra_col == 'best_epoch':
                        test_res[extra_col] = nn_out['epoch']
                    else:
                        test_res[extra_col] = cfg[extra_col]

        test_res['target'] = target
        test_res['horizon'] = horizon
        test_res['model'] = model
        test_res['seed'] = seed
        test_results.append(test_res)

    exp_res = pd.concat(test_results, axis=0).sort_values(['target', 'horizon', 'model', 'seed', g_node_col]).reset_index(drop=True)

    return exp_res


def merge_cdc_loc(raw_pred):
    # ensure the order
    raw_pred = raw_pred.sort_values([g_date_col, g_node_col, 'target', 'horizon'])
    # align g_node_col with cdc location
    locs = process_cdc_loc()
    node2loc = dict(zip(locs[g_node_col], locs['location']))
    raw_pred['location'] = raw_pred[g_node_col].map(lambda x: node2loc.get(x, pd.NA))

    return raw_pred


def merge_last_cum_truth(raw_pred, forecast_date, cdc_cum_truth=None):
    if 'location' not in raw_pred.columns:
        raw_pred = merge_cdc_loc(raw_pred)
    if cdc_cum_truth is None:
        cdc_confirmed_cum_truth = process_cdc_truth_from_csse('confirmed', stat_type='cum')
        cdc_deaths_cum_truth = process_cdc_truth_from_csse('deaths', stat_type='cum')
        cdc_confirmed_cum_truth['target'] = 'confirmed'
        cdc_deaths_cum_truth['target'] = 'deaths'
        cdc_cum_truth = pd.concat([cdc_confirmed_cum_truth, cdc_deaths_cum_truth], axis=0, ignore_index=True)

    # merge cdc cumulative info into forecasting results
    last_date = pd.to_datetime(forecast_date) + pd.Timedelta(-1, unit='day')
    last_cum_truth = cdc_cum_truth[cdc_cum_truth['date'] == last_date]
    raw_pred = pd.merge(raw_pred, last_cum_truth[['location', 'target', 'value']].rename(columns={'value': 'cum_sum'}),
                        on=['location', 'target'], how='left')

    # remove useless nodes that do not have a cdc location
    # TODO: do this when training our models
    useless_nodes = raw_pred[raw_pred['location'].isnull()][g_node_col].unique()
    if useless_nodes.size > 0:
        print(f'# useless nodes in our models {useless_nodes.size}, ...removed...')
    raw_pred = raw_pred.dropna(subset=['location', 'cum_sum']).reset_index(drop=True)

    return raw_pred


def transform_to_cdc_format(raw_pred, forecast_date):
    if 'cum_sum' not in raw_pred.columns:
        raise Exception('You should run merge_last_cum_truth before this function')
        # raw_pred = merge_last_cum_truth(raw_pred, forecast_date)

    # transform into CDC formats
    target2tag = {
        'confirmed': 'case',
        'deaths': 'death',
    }
    cdc_results = []
    for target in ['confirmed', 'deaths']:
        tag = target2tag[target]
        for n_week in [1, 2, 3, 4]:
            horizon = n_week * 7
            for stat_type in ['inc', 'cum']:
                cdc_target = f'{n_week} wk ahead {stat_type} {tag}'
                cdc_target_end_date = pd.to_datetime(forecast_date) + pd.Timedelta(horizon-1, unit='day')
                # print(cdc_target)
                cdc_res = raw_pred[(raw_pred['target'] == target) & (raw_pred['horizon'] == horizon)].reset_index(drop=True).copy()
                if stat_type == 'inc':
                    if n_week == 1:
                        cdc_res['value'] = cdc_res['pred']
                    else:
                        cdc_res['value'] = cdc_res['pred'] - raw_pred[(raw_pred['target'] == target) & (raw_pred['horizon'] == horizon-7)].reset_index(drop=True)['pred']
                else:
                    cdc_res['value'] = cdc_res['cum_sum'] + cdc_res['pred']
                cdc_res = cdc_res.rename(columns={g_date_col: 'forecast_date', 'target': 'model_target'})
                cdc_res['target'] = cdc_target
                cdc_res['target_end_date'] = cdc_target_end_date
                cdc_res['type'] = 'point'
                cdc_res['quantile'] = pd.NA
                cdc_results.append(cdc_res[[
                    'forecast_date', 'target', 'target_end_date', 'location', 'type', 'quantile', 'value'
                ]])
    all_cdc_res = pd.concat(cdc_results, axis=0, ignore_index=True)

    return all_cdc_res


def eval_cdc_pred(model_pred_pairs, all_cdc_label, forecast_date,
                  dropna=False, pre_loc_set=None,
                  level='county', n_week=1, stat_type='inc', value_type='case'):
    target = f'{n_week} wk ahead {stat_type} {value_type}'
    target_end_date = pd.to_datetime(forecast_date) + pd.Timedelta(n_week*7-1, unit='day')
    locs = process_cdc_loc()
    if pre_loc_set is None:
        if level == 'us':
            loc_set = set(locs['location'][:1])
        elif level == 'state':
            loc_set = set(locs['location'][1:58])
        elif level == 'county':
            loc_set = set(locs['location'][58:])
        else:
            loc_set = set(locs['location'])
    else:
        loc_set = pre_loc_set

    eval_res = all_cdc_label[(all_cdc_label['target'] == target) &
                             (all_cdc_label['target_end_date'] == target_end_date) &
                             all_cdc_label['location'].isin(loc_set)].copy()

    # location includes the cdc code, while g_node_col presents the node name
    eval_res = pd.merge(eval_res, locs[['location', g_node_col]], on='location', how='left')[[
        'forecast_date', 'target', 'target_end_date', 'location', g_node_col, 'label'
    ]]

    value_cols = ['label']
    mae_cols = []
    mape_cols = []
    for model_name, pred_df in model_pred_pairs:
        pred_col = f'Pred_{model_name}'
        mae_col = f'MAE_{model_name}'
        mape_col = f'MAPE_{model_name}'
        value_cols.append(pred_col)
        mae_cols.append(mae_col)
        mape_cols.append(mape_col)

        eval_res = pd.merge(eval_res,
                            pred_df[pred_df['type'] == 'point'][['target', 'target_end_date', 'location', 'value']].\
                                rename(columns={'value': pred_col}),
                            on=['target', 'target_end_date', 'location'], how='left')
        eval_res[mae_col] = np.abs(eval_res['label'] - eval_res[pred_col])
        eval_res[mape_col] = np.abs(eval_res['label'] - eval_res[pred_col]) / (eval_res['label'] + 1)
        non_pred_node_num = eval_res[pred_col].isnull().sum()
        if non_pred_node_num > 3:
            print(f'{model_name} drops {non_pred_node_num} nodes, such as')
            print(eval_res[eval_res[pred_col].isnull()][g_node_col].unique()[:10])

    if dropna:
        eval_res = eval_res.dropna(axis=0)

    print('-' * 30)
    print(f'forecast date: {forecast_date}')
    print(f'{level}-level {target} {target_end_date}')
    print('Label & Forecasting')
    print(eval_res[value_cols].mean().sort_values())
    print('MAE (Sorted)')
    print(eval_res[mae_cols].mean().sort_values())
    print('MAPE (Sorted)')
    print(eval_res[mape_cols].mean().sort_values())

    return eval_res


def get_model_seed_sort_by_mae(exp_res, cdc_label, forecast_date, level='county', n_week=1, stat_type='inc', value_type='case'):
    assert 'location' in exp_res.columns and 'cum_sum' in exp_res.columns
    exp_res = exp_res.set_index(['model', 'seed']).sort_index()
    model_pred_pairs = []
    for model, seed in exp_res.index.unique():
        model_tag = f'{model}~{seed}'
        cur_cdc_res = transform_to_cdc_format(exp_res.loc[(model, seed)], forecast_date)
        model_pred_pairs.append((model_tag, cur_cdc_res))
    eval_res = eval_cdc_pred(model_pred_pairs, cdc_label, forecast_date,
                             level=level, n_week=n_week, stat_type=stat_type, value_type=value_type)
    mae_cols = [col for col in eval_res.columns if col.startswith('MAE_')]
    sorted_model_seed_pairs = []
    for mae_col in eval_res[mae_cols].mean().sort_values().index:
        model, seed = mae_col[4:].split('~')
        if '_' in seed:
            sorted_model_seed_pairs.append((model, seed))
        else:
            sorted_model_seed_pairs.append((model, int(seed)))

    return sorted_model_seed_pairs, eval_res


def load_all_cdc_forecasts(min_date='2020-09-01'):
    cdc_model_pred_pairs = []
    for sub_dir in os.listdir(cdc_forecast_dir):
        sub_dir_path = os.path.join(cdc_forecast_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            model = sub_dir
            print(time.asctime(), f'Load forecast results for {model}')
            preds = []
            for idx, csv_fn in enumerate(os.listdir(sub_dir_path)):
                if not csv_fn.endswith('.csv'):
                    continue

                csv_date = '-'.join(csv_fn.split('-')[:3])
                if pd.to_datetime(csv_date) < pd.to_datetime(min_date):
                    continue

                csv_fp = os.path.join(sub_dir_path, csv_fn)
                print(f'--[{idx}] Load {csv_fp}')
                pred = read_cdc_forecast(csv_fp)
                preds.append(pred)

            if len(preds) > 0:
                all_pred = pd.concat(preds, axis=0, ignore_index=True)
                cdc_model_pred_pairs.append((model, all_pred))
            else:
                print('--Skip because no csv')

    return cdc_model_pred_pairs


def extract_cdc_forecast(raw_pred, level, target, target_end_date):
    raw_pred = raw_pred[(raw_pred['target'] == target) & (raw_pred['target_end_date'] == pd.to_datetime(target_end_date))]
    if level == 'county':
        raw_pred = raw_pred[raw_pred['location'].map(lambda x: len(x) == 5)]
    else:
        raw_pred = raw_pred[raw_pred['location'].map(lambda x: len(x) == 2)]

    return raw_pred


def filter_valid_cdc_forecasts(model_pred_pairs, forecast_date, min_loc_num, fixed_model_set=None, dup_keep='first', level='county', n_week=1, stat_type='inc', value_type='case'):
    target = f'{n_week} wk ahead {stat_type} {value_type}'
    target_end_date = pd.to_datetime(forecast_date) + pd.Timedelta(n_week*7-1, unit='day')
    print('-'*30)
    print(f'{level}-level {target} {target_end_date}')

    valid_model_pred_pairs = []
    for model, raw_pred in model_pred_pairs:
        if fixed_model_set is not None and model not in fixed_model_set:
            continue
        raw_valid_pred = extract_cdc_forecast(raw_pred, level, target, target_end_date)
        valid_pred = raw_valid_pred.drop_duplicates(subset=['location', 'target', 'type', 'quantile', 'target_end_date'],
                                                    keep=dup_keep, ignore_index=True)
        if valid_pred.shape[0] > 0 and valid_pred['location'].unique().size > min_loc_num:
            print(f'Add {model} as a candidate [{len(valid_model_pred_pairs)}] (drop {raw_valid_pred.shape[0]-valid_pred.shape[0]} duplicated forecasts)')
            valid_model_pred_pairs.append((model, valid_pred))

    return valid_model_pred_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Generation')
    parser.add_argument('--forecast_date', type=str, required=True)
    parser.add_argument('--level', type=str, default='all')
    parser.add_argument('--exp_dir_template', type=str, default='../Exp_us_{}')
    parser.add_argument('--output_dir', type=str, default='../outputs')
    parser.add_argument('--ens_num', type=int, default=50)
    # the following arguments are for evaluation purposes only
    # we will produce ensemble outputs for all kinds of targets
    parser.add_argument('--n_week', type=int, default=1)
    parser.add_argument('--stat_type', type=str, default='inc')
    parser.add_argument('--value_type', type=str, default='case')
    parser.add_argument('--use_last_ens', action='store_true')
    parser.add_argument('--do_eval', action='store_true')

    args = parser.parse_args()
    forecast_date = args.forecast_date
    level = args.level  # state or county
    exp_dir_template = args.exp_dir_template
    output_dir = args.output_dir
    n_week = args.n_week  # 1, 2, 3, 4
    stat_type = args.stat_type  # inc or cum
    value_type = args.value_type  # case or death
    ens_num = args.ens_num  # how many models to be included into the ensemble one

    exp_dir = exp_dir_template.format(forecast_date)
    forecast_dt = pd.to_datetime(forecast_date)
    next_dt = forecast_dt + pd.Timedelta(days=1)
    next_date = next_dt.strftime('%Y-%m-%d')

    print(time.asctime(), f'Load experimental results from {exp_dir}')
    exp_res = load_exp_res(exp_dir)
    print(time.asctime(), 'Merge last cumulative truths')
    exp_res = merge_last_cum_truth(exp_res, forecast_date)
    sel_exp_res = exp_res

    # transform to the cdc format per model and seed
    sel_exp_res = sel_exp_res.set_index(['model', 'seed']).sort_index()
    cdc_res_list = []
    for (m, s) in sel_exp_res.index.unique():
        cur_cdc_res = transform_to_cdc_format(sel_exp_res.loc[(m, s)], forecast_date)
        cur_cdc_res['model'] = m
        cur_cdc_res['seed'] = s
        cdc_res_list.append(cur_cdc_res)
    cdc_res = pd.concat(cdc_res_list, axis=0)
    cdc_res.loc[cdc_res['value'] < 0, 'value'] = 0
    cdc_res = cdc_res.dropna(subset=['value'])

    # point estimation
    point_cdc_res = cdc_res.groupby([
        'forecast_date', 'target', 'target_end_date', 'location', 'type',
    ])[['value']].mean().reset_index(drop=False)

    # quantile estimation
    fine_quantile_list = [
        0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
        0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990
    ]
    coarse_quantile_list = [0.025, 0.100, 0.250, 0.500, 0.750, 0.900, 0.975]
    # only calculate quantiles for state and country
    quant_cdc_res = cdc_res[cdc_res.location.map(lambda x: len(x) == 2)]
    quant_cdc_res.loc[quant_cdc_res.index, 'type'] = 'quantile'
    # for 'inc case', cdc requires coarse-grained quantiles
    coarse_cond = quant_cdc_res.target.map(lambda x: x.endswith(' inc case'))
    coarse_quant_cdc_res = quant_cdc_res[coarse_cond]
    fine_quant_cdc_res = quant_cdc_res[~coarse_cond]
    fine_quant_cdc_res = fine_quant_cdc_res.groupby([
        'forecast_date', 'target', 'target_end_date', 'location', 'type'
    ])['value'].quantile(fine_quantile_list).reset_index(drop=False).rename(columns={'level_5': 'quantile'})
    coarse_quant_cdc_res = coarse_quant_cdc_res.groupby([
        'forecast_date', 'target', 'target_end_date', 'location', 'type'
    ])['value'].quantile(coarse_quantile_list).reset_index(drop=False).rename(columns={'level_5': 'quantile'})

    os.makedirs(output_dir, exist_ok=True)
    cdc_ens_pred = pd.concat([fine_quant_cdc_res, coarse_quant_cdc_res, point_cdc_res], axis=0)\
        .sort_values(['target', 'target_end_date', 'location', 'type', 'quantile'])
    # change date to be compatible with cdc
    out_fp = os.path.join(output_dir, f'{forecast_date}_forecasts.csv')
    print(time.asctime(), 'Dump cdc results to', out_fp)
    # cdc does not allow 'cum case' for all targets and only allows 'inc case' for county-level forecasts
    cdc_ens_pred[
        cdc_ens_pred.apply(
            lambda x:
                (not x['target'].endswith('cum case')) and
                (len(x['location']) == 2 or x['target'].endswith('inc case')) and
                (x['location'] != '02063') and (x['location'] != '02066'),
            axis=1
        )
    ].groupby(
        ['forecast_date', 'target', 'target_end_date', 'location', 'type', 'quantile'],
        dropna=False
    ).mean().reset_index(drop=False).to_csv(out_fp, index=False)