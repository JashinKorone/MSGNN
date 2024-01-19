import json
import pandas as pd
import numpy as np
import datetime
import torch
from torch import nn
import torch.nn.functional as F
import os
import time
import geopy.distance
from sklearn.preprocessing import StandardScaler
from pandarallel import pandarallel
num_threads = 24
pandarallel.initialize(nb_workers=num_threads)


# From https://github.com/reichlab/covid19-forecast-hub
url_cdc_locations = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv'
path_cdc_locations = '../data/locations.csv'
url_cdc_inc_case = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Cases.csv'
url_cdc_inc_death = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Deaths.csv'
url_cdc_cum_case = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-truth/truth-Cumulative%20Cases.csv'
url_cdc_cum_death = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-truth/truth-Cumulative%20Deaths.csv'
# From https://github.com/CSSEGISandData/COVID-19
url_csse_us_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
path_csse_us_confirmed = '../data/time_series_covid19_confirmed_US.csv'
url_csse_us_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
path_csse_us_death = '../data/time_series_covid19_deaths_US.csv'
csse_meta_columns = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population']
csse_feas = [
    'confirmed', 'deaths', 'recovered',
    'confirmed_rolling', 'deaths_rolling', 'recovered_rolling',
    'weekday',
]
csse_targets = [
    'confirmed_target', 'deaths_target', 'recovered_target',
]
# From https://www.google.com/covid19/mobility/
url_google_mobility = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
google_mobility_feas = [
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
]
# From https://github.com/GoogleCloudPlatform/covid-19-open-data/blob/main/docs/table-government-response.md
url_google_gov_index = 'https://storage.googleapis.com/covid19-open-data/v2/index.csv'
url_google_gov = 'https://storage.googleapis.com/covid19-open-data/v2/oxford-government-response.csv'
google_gov_feas = [
    'school_closing', 'workplace_closing', 'cancel_public_events',
    'restrictions_on_gatherings', 'public_transport_closing', 'stay_at_home_requirements',
    'restrictions_on_internal_movement', 'international_travel_controls',
    'income_support', 'debt_relief', 'fiscal_measures', 'international_support',
    'public_information_campaigns', 'testing_policy', 'contact_tracing',
    'emergency_investment_in_healthcare', 'investment_in_vaccines', 'stringency_index'
]
# Define global column names
g_node_col = 'Node'
g_date_col = 'Date'


def process_cdc_loc():
    locs = pd.read_csv(path_cdc_locations, dtype={'location': str})
    csse_deaths = pd.read_csv(path_csse_us_death)

    csse_deaths = csse_deaths.dropna(subset=['FIPS'])
    csse_deaths[g_node_col] = csse_deaths.apply(lambda x: f"{x['Province_State']} ~ {x['Admin2']}", axis=1)
    csse_deaths['location'] = csse_deaths['FIPS'].map(lambda x: '{0:0>5}'.format(int(x)))
    ct_loc_node = csse_deaths[['location', g_node_col]].drop_duplicates()

    locs = pd.merge(locs, ct_loc_node, on='location', how='left')
    locs[g_node_col] = locs.apply(lambda x: x['location_name'] if pd.isna(x[g_node_col]) and len(x['location']) == 2 else x[g_node_col], axis=1)
    locs.dropna(subset=[g_node_col])

    return locs


def get_valid_node_set(level):
    if level == 'state':
        valid_node_set = {
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
            'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
            'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas',
            'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
            'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana',
            'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
            'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
            'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
            'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'District of Columbia',
            'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'US',
        }
    else:
        locs = process_cdc_loc()
        valid_node_set = set(locs[locs['location'].map(lambda x: len(x) == 5)][g_node_col].values)

    return valid_node_set


def process_cdc_truth(target, stat_type='inc'):
    if target == 'confirmed':
        if stat_type == 'inc':
            cdc_truth = pd.read_csv(url_cdc_inc_case, parse_dates=['date'], dtype={'location': str})
        elif stat_type == 'cum':
            cdc_truth = pd.read_csv(url_cdc_cum_case, parse_dates=['date'], dtype={'location': str})
        else:
            raise Exception(f'Unsupported stat_type {stat_type}')
    elif target == 'deaths':
        if stat_type == 'inc':
            cdc_truth = pd.read_csv(url_cdc_inc_death, parse_dates=['date'], dtype={'location': str})
        elif stat_type == 'cum':
            cdc_truth = pd.read_csv(url_cdc_cum_death, parse_dates=['date'], dtype={'location': str})
        else:
            raise Exception(f'Unsupported stat_type {stat_type}')
    else:
        raise Exception(f'Unsupported target {target}')

    return cdc_truth


def process_cdc_truth_from_csse(target, stat_type='inc'):
    if target == 'confirmed':
        daily_ts = pd.read_csv(path_csse_us_confirmed)
    elif target == 'deaths':
        daily_ts = pd.read_csv(path_csse_us_death)
    else:
        raise Exception(f'Unknown target {target}')
    daily_feas = [item for item in daily_ts.columns if item not in csse_meta_columns]

    if stat_type == 'inc':
        # get daily incremental values
        daily_ts[daily_feas] = daily_ts[daily_feas].diff(axis=1)

    def reindex_daily_fea(df, level, name):
        # name the node col
        if level == 'state':
            df[g_node_col] = df['Province_State']
        else:
            df[g_node_col] = df.apply(lambda x: '{} ~ {}'.format(x['Province_State'], x['Admin2']), axis=1)
        # get daily features and reindex
        df = df[[g_node_col] + daily_feas].set_index(g_node_col).stack().reset_index().\
            rename({'level_1': g_date_col, 0: name}, axis=1).groupby([g_node_col, g_date_col]).sum().reset_index()
        # add a global node
        if level == 'state':
            df_us = df.groupby(g_date_col).sum().reset_index()
            df_us[g_node_col] = 'US'
            df = pd.concat([df, df_us], axis=0, ignore_index=True)
        # fix abnormal values
        df.loc[df[name] < 0, name] = 0
        # sort by node and date
        df[g_date_col] = pd.to_datetime(df[g_date_col])
        df = df.sort_values([g_node_col, g_date_col])
        return df
    daily_county_ts = reindex_daily_fea(daily_ts, 'county', 'value')
    daily_state_ts = reindex_daily_fea(daily_ts, 'state', 'value')
    daily_ts = pd.concat([daily_county_ts, daily_state_ts], axis=0, ignore_index=True)

    locs = process_cdc_loc()
    daily_ts = pd.merge(daily_ts, locs, on=[g_node_col], how='inner').rename(columns={g_date_col: 'date'})

    cdc_truth = daily_ts[['date', 'location', 'location_name', 'value']].reset_index(drop=True)

    return cdc_truth


def read_cdc_forecast(csv_path):
    cdc_fore = pd.read_csv(csv_path,
                           parse_dates=['forecast_date', 'target_end_date'],
                           dtype={'location': str})

    return cdc_fore


def get_all_cdc_label(forecast_dates, from_csse=True):
    # 'forecast_date' should be the sunday of the epiweek (included)
    # collect cdc ground truths
    if from_csse:
        # CSSE always produces the latest results
        cdc_confirmed_inc_truth = process_cdc_truth_from_csse('confirmed', stat_type='inc')
        cdc_confirmed_cum_truth = process_cdc_truth_from_csse('confirmed', stat_type='cum')
        cdc_deaths_inc_truth = process_cdc_truth_from_csse('deaths', stat_type='inc')
        cdc_deaths_cum_truth = process_cdc_truth_from_csse('deaths', stat_type='cum')
    else:
        # CDC aggregates results with one-week delay
        cdc_confirmed_inc_truth = process_cdc_truth('confirmed', stat_type='inc')
        cdc_confirmed_cum_truth = process_cdc_truth('confirmed', stat_type='cum')
        cdc_deaths_inc_truth = process_cdc_truth('deaths', stat_type='inc')
        cdc_deaths_cum_truth = process_cdc_truth('deaths', stat_type='cum')

    forecast_date2all_cdc_label = {}
    for forecast_date in forecast_dates:
        # we only need data behind 'forecast_date' to calculate ground-truth labels
        cdc_key2truth = {
            ('confirmed', 'inc'): cdc_confirmed_inc_truth[cdc_confirmed_inc_truth['date'] >= pd.to_datetime(forecast_date)],
            ('confirmed', 'cum'): cdc_confirmed_cum_truth[cdc_confirmed_cum_truth['date'] >= pd.to_datetime(forecast_date)],
            ('deaths', 'inc'): cdc_deaths_inc_truth[cdc_deaths_inc_truth['date'] >= pd.to_datetime(forecast_date)],
            ('deaths', 'cum'): cdc_deaths_cum_truth[cdc_deaths_cum_truth['date'] >= pd.to_datetime(forecast_date)],
        }
        target2tag = {
            'confirmed': 'case',
            'deaths': 'death',
        }

        all_cdc_labels = []
        for n_week in [1, 2, 3, 4]:
            cdc_target_end_date = pd.to_datetime(forecast_date) + pd.Timedelta(n_week*7-1, unit='day')
            for stat_type in ['inc', 'cum']:
                for target in ['confirmed', 'deaths']:
                    tag = target2tag[target]
                    cdc_target = f'{n_week} wk ahead {stat_type} {tag}'
                    # print(cdc_target_end_date, cdc_target)

                    # organize the format to be similar to forecasting results in https://github.com/reichlab/covid19-forecast-hub/blob/master/data-processed/
                    cdc_truth = cdc_key2truth[(target, stat_type)]
                    if stat_type == 'inc':
                        # This label is the incident (weekly) number of cases/deaths during the week that is N weeks after forecast_date
                        cdc_truth['label'] = cdc_truth.groupby('location')['value'].rolling(7).sum().shift(1-7*n_week).reset_index(0, drop=True)
                        cdc_label = cdc_truth[cdc_truth['date'] == pd.to_datetime(forecast_date)].copy()
                    else:
                        # This label is the cumulative number of cases/deaths up to and including N weeks after forecast_date
                        cdc_truth['label'] = cdc_truth.groupby('location')['value'].shift(1-7*n_week)
                        cdc_label = cdc_truth[cdc_truth['date'] == pd.to_datetime(forecast_date)].copy()
                    cdc_label['forecast_date'] = cdc_label['date']
                    cdc_label['target'] = cdc_target
                    cdc_label['target_end_date'] = cdc_target_end_date
                    all_cdc_labels.append(cdc_label[[
                        'forecast_date', 'target', 'target_end_date', 'location', 'label',
                    ]])
        all_cdc_label = pd.concat(all_cdc_labels, axis=0, ignore_index=True)
        forecast_date2all_cdc_label[forecast_date] = all_cdc_label

    return forecast_date2all_cdc_label


def process_csse_us(forecast_days=7, level='county', do_log1p=True):
    daily_confirmed = pd.read_csv(url_csse_us_confirmed)
    daily_deaths = pd.read_csv(url_csse_us_deaths)
    daily_recovered = pd.read_csv(url_csse_us_deaths)
    # Note: us_confirmed does not have the 'Population' column, but us_deaths has
    daily_feas = [item for item in daily_confirmed.columns if item not in csse_meta_columns]
    daily_feas = [item for item in daily_feas if item in daily_deaths.columns]

    # calculate daily incremental number
    daily_confirmed[daily_feas] = daily_confirmed[daily_feas].diff(axis=1)
    daily_deaths[daily_feas] = daily_deaths[daily_feas].diff(axis=1)
    daily_recovered[daily_feas] = daily_recovered[daily_feas].diff(axis=1)

    # Define the node column (state or county)
    for df in [daily_confirmed, daily_deaths, daily_recovered]:
        if level == 'county':
            df[g_node_col] = df.apply(lambda x: '{} ~ {}'.format(x['Province_State'], x['Admin2']), axis=1)
        else:
            df[g_node_col] = df['Province_State']

    # Separate metadata and time-series
    csse_meta = daily_deaths[[g_node_col] + csse_meta_columns]

    def reindex_daily_fea(df, name):
        # get daily features and reindex
        df = df[[g_node_col] + daily_feas].set_index(g_node_col).stack().reset_index().\
            rename({'level_1': g_date_col, 0: name}, axis=1).groupby([g_node_col, g_date_col]).sum().reset_index()
        # fix abnormal values
        df.loc[df[name] < 0, name] = 0
        # sort by node and date
        df[g_date_col] = pd.to_datetime(df[g_date_col])
        df = df.sort_values([g_node_col, g_date_col])
        # add a global node
        if level == 'state':
            df_us = df.groupby(g_date_col).sum().reset_index()
            df_us[g_node_col] = 'US'
            df_us = df_us[df.columns]
            df = pd.concat([df_us, df], axis=0, ignore_index=True)
        return df
    daily_confirmed = reindex_daily_fea(daily_confirmed, 'confirmed')
    daily_deaths = reindex_daily_fea(daily_deaths, 'deaths')
    daily_recovered = reindex_daily_fea(daily_recovered, 'recovered')

    # Add rolling features and targets
    def add_rolling_fea(df, name):
        df['{}_rolling'.format(name)] = np.log1p(df.groupby(g_node_col)[name].rolling(forecast_days).sum().reset_index(0, drop=True))
        df['{}_target'.format(name)] = np.log1p(df.groupby(g_node_col)[name].rolling(forecast_days).sum().shift(1-forecast_days).reset_index(0, drop=True))
        df[name] = np.log1p(df[name])
    add_rolling_fea(daily_confirmed, 'confirmed')
    add_rolling_fea(daily_deaths, 'deaths')
    add_rolling_fea(daily_recovered, 'recovered')

    # merge confirm, death, and recover time series, and add another 'weekday' feature
    daily_ts = pd.merge(daily_confirmed, daily_deaths, how='left', on=[g_node_col, g_date_col])
    daily_ts = pd.merge(daily_ts, daily_recovered, how='left', on=[g_node_col, g_date_col])
    daily_ts['weekday'] = daily_ts[g_date_col].map(lambda x: x.weekday())

    valid_node_set = get_valid_node_set(level)
    daily_ts = daily_ts[daily_ts[g_node_col].isin(valid_node_set)]

    return daily_ts


def process_mobility_us(level='county'):
    mobility = pd.read_csv(url_google_mobility)

    # filter to get interested nodes
    if level == 'county':
        filter_cond = (mobility['country_region'] == 'United States') & mobility['sub_region_1'].notnull() & mobility['sub_region_2'].notnull()
        mobility = mobility[filter_cond]
        # TODO: not sure whether these county names align with CSSE counterparts
        mobility[g_node_col] = mobility.apply(lambda x: '{} ~ {}'.format(x['sub_region_1'], x['sub_region_2'].rstrip(' County')), axis=1)
    else:
        filter_cond = (mobility['country_region'] == 'United States') & mobility['sub_region_1'].notnull() & mobility['sub_region_2'].isnull()
        mobility = mobility[filter_cond]
        mobility[g_node_col] = mobility['sub_region_1']
    mobility[g_date_col] = pd.to_datetime(mobility['date'])
    mobility = mobility[[g_node_col, g_date_col] + google_mobility_feas]
    mobility = mobility.sort_values([g_node_col, g_date_col])
    mobility[google_mobility_feas] = mobility[google_mobility_feas] / 100.0

    # add a global node
    if level == 'state':
        mobility_us = mobility.groupby(g_date_col)[google_mobility_feas].mean().reset_index()
        mobility_us[g_node_col] = 'US'
        mobility = pd.concat([mobility_us, mobility], axis=0, ignore_index=True)
    mobility = mobility.drop_duplicates([g_node_col, g_date_col])

    return mobility


def process_gov_us(level='county'):
    gov_index = pd.read_csv(url_google_gov_index)
    gov_raw = pd.read_csv(url_google_gov)

    # filter to get interested nodes
    if level == 'county':
        filter_cond = (gov_index['aggregation_level'] == 2) & (gov_index['country_code'] == 'US')
        gov_index = gov_index[filter_cond]
        # TODO: not sure whether these county names align with CSSE counterparts
        gov_index[g_node_col] = gov_index.apply(lambda x: '{} ~ {}'.format(x['subregion1_name'], x['subregion2_name'].rstrip(' County')), axis=1)
    else:
        filter_cond = (gov_index['aggregation_level'] == 1) & (gov_index['country_code'] == 'US')
        gov_index = gov_index[filter_cond]
        gov_index[g_node_col] = gov_index['subregion1_name']
    gov_index = gov_index[['key', g_node_col]]
    # TODO: gov_raw does not have county-level data yet
    gov = pd.merge(gov_index, gov_raw, how='left', on='key').fillna(0.0)
    gov[g_date_col] = pd.to_datetime(gov['date'])
    gov[google_gov_feas] = np.log1p(gov[google_gov_feas])

    # TODO: we do not have aggregated values for a global node, US
    gov = gov[[g_node_col, g_date_col] + google_gov_feas].drop_duplicates([g_node_col, g_date_col])

    return gov


def generate_dataset_us(dump_fp, forecast_days, level='county', dump_flag=True):
    print('='*30)
    print(time.asctime(), 'Generate dataset')

    print(time.asctime(), 'Fetch and process CSSE daily time series')
    daily_ts = process_csse_us(forecast_days=forecast_days, level=level)
    ts_max_dt = daily_ts[g_date_col].max().date()
    print('Shape:', daily_ts.shape,
          '# Nodes:', daily_ts[g_node_col].unique().size,
          '# Dates:', daily_ts[g_date_col].unique().size,
          'Max Date:', ts_max_dt,)

    def rolling_shift_mobility(df, days):
        for col in df.columns:
            if col == g_node_col or col == g_date_col:
                continue
            df[col] = df.groupby(g_node_col)[col]\
                .rolling(days).mean().shift(days).reset_index(0, drop=True)

    print(time.asctime(), 'Fetch and process Google mobility time series')
    mobility = process_mobility_us(level=level)
    mob_max_dt = mobility[g_date_col].max().date()
    print('Shape:', mobility.shape,
          '# Nodes:', mobility[g_node_col].unique().size,
          '# Dates:', mobility[g_date_col].unique().size,
          'Max Date:', mob_max_dt,)
    rolling_shift_mobility(mobility, 7)
    if (ts_max_dt - mob_max_dt).days > 7:
        raise Exception('Mobility delays more than 7 days')

    print(time.asctime(), 'Fetch and process gov policy time series')
    gov = process_gov_us(level=level)
    print('Shape:', gov.shape,
          '# Nodes:', gov[g_node_col].unique().size,
          '# Dates:', gov[g_date_col].unique().size,
          'Max Date:', gov[g_date_col].max().date(),)
    # TODO: add updating-delay processing logics

    data = pd.merge(daily_ts, mobility, how='left', on=[g_node_col, g_date_col]).fillna(0.0)
    data = pd.merge(data, gov, how='left', on=[g_node_col, g_date_col]).fillna(0.0)
    if dump_flag:
        print(time.asctime(), f'Dump dataset into {dump_fp}')
        data.to_csv(dump_fp, index=False)

    return data


def process_us_geo_distance(level='county', max_state_neighbor=5):
    assert level in {'county', 'state'}
    daily_deaths = pd.read_csv(url_csse_us_deaths)
    isolated_states = {
        'Alaska', 'American Samoa', 'Guam', 'Hawaii', 'Northern Mariana Islands',
        # the following two are very close to the mainland
        # 'Puerto Rico', 'Virgin Islands',
    }

    # 1. Calculate pairwise distances at state level

    daily_deaths['State'] = daily_deaths['Province_State']
    state_geo = daily_deaths[['State', 'Lat', 'Long_']].groupby('State')[['Lat', 'Long_']].median().reset_index(drop=False)
    state_index = pd.MultiIndex.from_product([state_geo['State'].values, state_geo['State'].values],
                                             names=['State_1', 'State_2']).to_frame(index=False)
    state_index = state_index[(state_index['State_1'] != state_index['State_2']) &
                              state_index['State_1'].map(lambda x: x not in isolated_states) &
                              state_index['State_2'].map(lambda x: x not in isolated_states)].reset_index(drop=True)
    state_pair = pd.merge(state_index, state_geo.rename(columns={'State': 'State_1', 'Lat': 'Lat_1', 'Long_': 'Long_1'}),
                          how='left', on='State_1')
    state_pair = pd.merge(state_pair, state_geo.rename(columns={'State': 'State_2', 'Lat': 'Lat_2', 'Long_': 'Long_2'}),
                          how='left', on='State_2')
    state_pair['distance'] = state_pair.apply(lambda x: geopy.distance.distance((x['Lat_1'], x['Long_1']), (x['Lat_2'], x['Long_2'])).km, axis=1)
    state_pair['rank'] = state_pair.groupby('State_1')['distance'].rank()

    if level == 'state':
        return state_pair[['State_1', 'State_2', 'distance', 'rank']].\
            rename(columns={'State_1': f'{g_node_col}', 'State_2': f'{g_node_col}_1'})

    # 2. Calculate approximate pairwise distances at county level

    # Since it is computationally prohibitive to compute pairwise distances at county level,
    # we only compute county-level distances on filtered state pairs
    filtered_state_pair = state_pair[state_pair['rank'] <= max_state_neighbor][['State_1', 'State_2']]
    self_state_pair = pd.concat([state_geo[['State']].rename(columns={'State': 'State_1'}),
                                 state_geo[['State']].rename(columns={'State': 'State_2'})], axis=1)
    filtered_state_pair = pd.concat([filtered_state_pair, self_state_pair], axis=0, ignore_index=True)
    # Calculate pairwise distances among counties within filtered state pairs
    daily_deaths['County'] = daily_deaths.apply(lambda x: '{} ~ {}'.format(x['Province_State'], x['Admin2']), axis=1)
    county_geo = daily_deaths[['County', 'Lat', 'Long_']].groupby('County')[['Lat', 'Long_']].median().reset_index(drop=False)
    county_geo['State'] = county_geo['County'].map(lambda x: x.split(' ~ ')[0])
    county_pair = pd.merge(filtered_state_pair, county_geo.rename(columns={'County': 'County_1', 'State': 'State_1', 'Lat': 'Lat_1', 'Long_': 'Long_1'}),
                           how='left', on='State_1')
    county_pair = pd.merge(county_pair, county_geo.rename(columns={'County': 'County_2', 'State': 'State_2', 'Lat': 'Lat_2', 'Long_': 'Long_2'}),
                           how='left', on='State_2')
    county_pair = county_pair[county_pair['County_1'] != county_pair['County_2']].sort_values(['County_1', 'County_2']).reset_index(drop=True)
    county_pair['distance'] = county_pair.parallel_apply(lambda x: geopy.distance.distance((x['Lat_1'], x['Long_1']), (x['Lat_2'], x['Long_2'])).km, axis=1)
    county_pair['rank'] = county_pair.groupby('County_1')['distance'].rank()

    return county_pair[['County_1', 'County_2', 'distance', 'rank']].\
        rename(columns={'County_1': f'{g_node_col}', 'County_2': f'{g_node_col}_1'})


def generate_us_graph(dump_fp, max_neighbor_num=50, dump_flag=True):
    daily_state = generate_dataset_us('', 7, level='state', dump_flag=False)
    daily_county = generate_dataset_us('', 7, level='county', dump_flag=False)
    geo_state = process_us_geo_distance(level='state')
    geo_county = process_us_geo_distance(level='county')

    state_nodes = list(daily_state[g_node_col].unique())
    assert state_nodes[0] == 'US'
    county_nodes = list(daily_county[g_node_col].unique())

    all_nodes = state_nodes + county_nodes
    node2idx = {node: idx for idx, node in enumerate(all_nodes)}
    node_levels = [0] + [1]*(len(state_nodes)-1) + [2]*len(county_nodes)

    geo_state = geo_state[geo_state[g_node_col].isin(node2idx) & geo_state[f'{g_node_col}_1'].isin(node2idx) & (geo_state['rank'] <= max_neighbor_num)]
    geo_county = geo_county[geo_county[g_node_col].isin(node2idx) & geo_county[f'{g_node_col}_1'].isin(node2idx) & (geo_county['rank'] <= max_neighbor_num)]
    geo_all = pd.concat([geo_state, geo_county], axis=0, ignore_index=True)

    # graph defined by geographical information
    geo_index_src = list(geo_all[f'{g_node_col}_1'].map(node2idx).values)
    geo_index_tgt = list(geo_all[g_node_col].map(node2idx).values)
    geo_weight = list(geo_all['distance'].map(lambda x: 1.0/np.sqrt(x)).values)

    # graph defined by government topology
    state_ids = [node2idx[node] for node in state_nodes[1:]]
    state_parent_ids = [node2idx[state_nodes[0]] for _ in state_ids]
    county_ids = [node2idx[node] for node in county_nodes if node.split(' ~ ')[0] in node2idx]
    county_parent_ids = [node2idx[node.split(' ~ ')[0]] for node in county_nodes if node.split(' ~ ')[0] in node2idx]
    parent_ids = state_parent_ids + county_parent_ids
    child_ids = state_ids + county_ids
    topo_index_src = parent_ids + child_ids
    topo_index_tgt = child_ids + parent_ids
    topo_weight = [sum(geo_weight) / len(geo_weight)] * len(topo_index_src)

    edge_index_src = topo_index_src + geo_index_src
    edge_index_tgt = topo_index_tgt + geo_index_tgt

    graph_info = {
        'edge_index': torch.LongTensor([edge_index_src, edge_index_tgt]),
        'edge_weight': torch.FloatTensor(topo_weight + geo_weight),
        'edge_type': torch.LongTensor(np.array([0]*len(topo_weight) + [1]*len(geo_weight))),
        'node_name': all_nodes,
        'node_type': torch.LongTensor(node_levels),
    }
    if dump_flag:
        print(time.asctime(), 'Generate us graph to {}'.format(dump_fp))
        torch.save(graph_info, dump_fp)

    return graph_info


def load_data(data_fp, start_date, min_peak_size, lookback_days, lookahead_days, label='deaths_target', use_mobility=True, logger=print):
    logger('Load Data from ' + data_fp)
    logger('lookback_days={}, lookahead_days={}, '.format(
        lookback_days, lookahead_days))
    data = pd.read_csv(data_fp, parse_dates=[g_date_col])
    # Filter out invalid dates (early pandemic period)
    data = data[data[g_date_col] >= pd.to_datetime(start_date)].reset_index(drop=True)
    # Filter out invalid nodes (with too few confirmed cases)
    node_max_confirm = data.groupby(g_node_col)['confirmed'].max()
    min_peak_size = max(0, min_peak_size)
    valid_nodes = node_max_confirm[node_max_confirm >= np.log1p(min_peak_size)].index.values
    data = data[data[g_node_col].isin(valid_nodes)].reset_index(drop=True)

    used_dates = list(data[g_date_col].unique())
    used_nodes = list(data[g_node_col].unique())
    print('# Dates:', len(used_dates), '# Nodes:', len(used_nodes))

    if use_mobility:
        # note that we should always place 'csse_feas' at last
        # because the model will use the reversed index to locate xxx_rolling for lr module
        used_feas = google_mobility_feas + csse_feas
    else:
        used_feas = csse_feas

    df = pd.DataFrame(index=pd.MultiIndex.from_product([used_nodes, used_dates],
                      names=[g_node_col, g_date_col])).reset_index()
    df = pd.merge(df, data, on=[g_node_col, g_date_col], how='left').fillna(0.0)

    df_gov = df[google_gov_feas].values.reshape(len(used_nodes), len(used_dates), len(google_gov_feas)).copy()
    df = df[used_feas + csse_targets].values.reshape(len(used_nodes), len(used_dates), len(used_feas)+len(csse_targets))
    used_dates = list(map(lambda x: pd.to_datetime(x), used_dates))
    day_inputs = []
    day_gov_inputs = []
    outputs = []
    label_dates = []
    label2idx = {csse_targets[idx]: idx for idx in [-1, -2, -3]}
    label_idx = label2idx.get(label, -2)
    for day_idx in range(lookback_days, len(used_dates)):
        day_input = df[:, day_idx-lookback_days+1:day_idx+1, :-3].copy()
        day_gov_input = df_gov[:, day_idx-lookback_days+1:day_idx+1, :].copy()
        if day_idx + lookahead_days > len(used_dates):
            tmp = df[:, day_idx:day_idx + lookahead_days, label_idx].copy()
            sz = tmp.shape
            tmp_empty = np.zeros((sz[0], lookahead_days - sz[1]))
            output = np.concatenate([tmp, tmp_empty], axis=1)
        else:
            output = df[:, day_idx:day_idx + lookahead_days, label_idx].copy()

        day_inputs.append(day_input)
        day_gov_inputs.append(day_gov_input)
        outputs.append(output)
        label_dates.append(used_dates[day_idx])

    # [num_samples, num_nodes, lookback_days, day_feature_dim]
    day_inputs = np.stack(day_inputs, axis=0)
    # [num_samples, num_nodes, lookback_days, day_gov_feature_dim]
    day_gov_inputs = np.stack(day_gov_inputs, axis=0)
    # [num_samples, num_nodes, lookahead_days]
    outputs = np.stack(outputs, axis=0)

    day_inputs = torch.from_numpy(day_inputs).float()
    day_gov_inputs = torch.from_numpy(day_gov_inputs).float()
    outputs = torch.from_numpy(outputs).float()
    # A = torch.ones(len(used_nodes),len(used_nodes)).to_sparse()
    # edge_index = A._indices().long()

    logger("Input size: {}; {},Output size: {}".format(
        day_inputs.size(), day_gov_inputs.size(), outputs.size()))

    return day_inputs, day_gov_inputs, outputs, label_dates, used_nodes, used_feas


def load_new_data(data_fp, config, logger=print):
    lookback_days = config.lookback_days
    lookahead_days = config.lookahead_days

    logger('='*20 + 'New Data Loading' + '='*20)
    logger('Load ' + data_fp + ', lookback_days={}, lookahead_days={}, '.format(
        lookback_days, lookahead_days))
    df = pd.read_csv(data_fp, parse_dates=[g_date_col])
    # add population info
    cdc_loc = process_cdc_loc()
    df = pd.merge(df, cdc_loc[[g_node_col, 'population']], on=g_node_col, how='left')

    logger('1. Calculate epidemic features')

    epi_feas = []
    epi_labels = []
    for target in ['confirmed', 'deaths']:
        # recover log1p processing, TODO: refactor the whole pipeline
        df[target] = np.expm1(df[target])
        df[f'{target}_target'] = np.expm1(df[f'{target}_target'])
        epi_feas.append(target)
        epi_labels.append(f'{target}_target')

        for f_days in range(7, config.horizon+1, 7):
            fea = f'{target}.rolling({f_days}).sum()'
            df[fea] = df.groupby(g_node_col)[target]\
                .rolling(f_days).sum().reset_index(0, drop=True)
            epi_feas.append(fea)
            fea = f'{target}.rolling({f_days}).sum().shift({f_days})'
            df[fea] = df.groupby(g_node_col)[target]\
                .rolling(f_days).sum().shift(f_days).reset_index(0, drop=True)
            epi_feas.append(fea)

    if config.use_mobility:
        main_feas = google_mobility_feas + epi_feas
    else:
        main_feas = list(epi_feas)
    # always place 'weekday' at last
    main_feas.append('weekday')
    gov_feas = list(google_gov_feas)

    logger('2. Reindex the feature dataframe and normalize')

    df = df[df[g_date_col] >= pd.to_datetime(config.start_date)]
    used_dates = list(df[g_date_col].unique())
    fc_date = pd.to_datetime(config.forecast_date)
    required_fea_date = fc_date + pd.Timedelta(days=-1+config.fea_day_offset)
    if used_dates[-1] >= required_fea_date:
        # we have enough data as features for the forecast date
        # then we supplement enough dates
        while used_dates[-1] < fc_date:
            used_dates.append(pd.to_datetime(used_dates[-1]) + pd.Timedelta(days=1))
    else:
        raise Exception('Do not have enough data (last_date={}, fea_day_offset={} day) to forecast for {}'.format(
            used_dates[-1], config.fea_day_offset, fc_date
        ))
    used_nodes = list(df[g_node_col].unique())
    num_rows = len(used_dates) * len(used_nodes)
    logger(f'# Nodes: {len(used_nodes)}, # Dates: {len(used_dates)}, Max Date: {used_dates[-1]}')
    logger('Raw Feature Statistics:')
    for fea in main_feas:
        logger('\t{:50s} NULL Ratio: {:.3f}, Min: {:.2f}, Median: {:.2f}, Max: {:.2f}'.format(
            fea, df[fea].isnull().sum()/num_rows, df[fea].min(), df[fea].median(), df[fea].max()))
    df_anchor = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [used_nodes, used_dates], names=[g_node_col, g_date_col])
        ).reset_index()
    df = pd.merge(df_anchor, df, on=[g_node_col, g_date_col], how='left').fillna(0.0)

    # normalize features if needed
    for fea in epi_feas + epi_labels:
        if config.use_popu_norm:
            df[fea] = df[fea] * 10**5 / df['population']
        if config.use_logy:
            df[fea] = np.log1p(df[fea])
    if config.use_fea_zscore:
        fea_scaler = StandardScaler()
        df[main_feas[:-1]] = fea_scaler.fit_transform(df[main_feas[:-1]])
    else:
        fea_scaler = None
    logger('Normalized Feature Statistics:')
    for fea in main_feas:
        logger('\t{:50s} NULL Ratio: {:.3f}, Min: {:.2f}, Median: {:.2f}, Max: {:.2f}'.format(
            fea, df[fea].isnull().sum()/num_rows, df[fea].min(), df[fea].median(), df[fea].max()))

    logger('3. Build feature and label tensors')
    node_popu = df[[g_node_col, 'population']].drop_duplicates(subset=[g_node_col], keep='first')['population'].values
    gov_fea_mat = df[gov_feas].values.reshape(len(used_nodes), len(used_dates), len(gov_feas))
    main_fea_mat = df[main_feas].values.reshape(len(used_nodes), len(used_dates), len(main_feas))
    label_mat = df[epi_labels].values.reshape(len(used_nodes), len(used_dates), len(epi_labels))
    used_dates = list(map(lambda x: pd.to_datetime(x), used_dates))
    day_inputs = []
    day_gov_inputs = []
    outputs = []
    label_dates = []
    label2idx = {label: idx for idx, label in enumerate(epi_labels)}
    label_idx = label2idx[config.label]
    for day_idx in range(lookback_days-config.fea_day_offset, len(used_dates)):
        # day_input = main_fea_mat[:, day_idx-lookback_days+1:day_idx+1, :].copy()
        # day_gov_input = gov_fea_mat[:, day_idx-lookback_days+1:day_idx+1, :].copy()
        cur_date_end = day_idx + config.fea_day_offset
        cur_date_start = cur_date_end - lookback_days
        day_input = main_fea_mat[:, cur_date_start:cur_date_end, :].copy()
        day_gov_input = gov_fea_mat[:, cur_date_start:cur_date_end, :].copy()
        if day_idx + lookahead_days > len(used_dates):
            tmp = label_mat[:, day_idx:day_idx + lookahead_days, label_idx].copy()
            sz = tmp.shape
            tmp_empty = np.zeros((sz[0], lookahead_days - sz[1]))
            output = np.concatenate([tmp, tmp_empty], axis=1)
        else:
            output = label_mat[:, day_idx:day_idx + lookahead_days, label_idx].copy()

        day_inputs.append(day_input)
        day_gov_inputs.append(day_gov_input)
        outputs.append(output)
        label_dates.append(used_dates[day_idx])

    # [num_samples, num_nodes, lookback_days, day_feature_dim]
    day_inputs = np.stack(day_inputs, axis=0)
    logger('day inputs shape is ' + str(day_inputs.shape))
    # [num_samples, num_nodes, lookback_days, day_gov_feature_dim]
    day_gov_inputs = np.stack(day_gov_inputs, axis=0)
    # [num_samples, num_nodes, lookahead_days]
    outputs = np.stack(outputs, axis=0)

    node_popu = torch.from_numpy(node_popu).float()
    day_inputs = torch.from_numpy(day_inputs).float()
    day_gov_inputs = torch.from_numpy(day_gov_inputs).float()
    outputs = torch.from_numpy(outputs).float()
    logger("Input size: {}, {}; Output size: {}".format(
        day_inputs.size(), day_gov_inputs.size(), outputs.size()))
    assert node_popu.shape[0] == len(used_nodes)
    assert day_inputs.shape[-1] == len(main_feas)
    assert day_gov_inputs.shape[-1] == len(gov_feas)

    return day_inputs, day_gov_inputs, outputs, label_dates, used_nodes,\
        main_feas, gov_feas, node_popu, fea_scaler


if __name__ == "__main__":
    os.makedirs('../data', exist_ok=True)
    fp_graph = '../data/us_graph.cpt'
    if not os.path.exists(fp_graph):
        generate_us_graph(fp_graph, max_neighbor_num=50)

    for forecast_days in [7, 14, 21, 28]:
        datasets = []
        for level in ['state', 'county']:
            dump_fp = '../data/daily_us_{}_{}.csv'.format(level, forecast_days)
            dataset = generate_dataset_us(dump_fp, forecast_days, level=level, dump_flag=True)
            datasets.append(dataset)
        merge_dataset = pd.concat(datasets, axis=0, ignore_index=True)
        dump_fp = '../data/daily_us_{}.csv'.format(forecast_days)
        print(time.asctime(), f'Dump merged dataset into {dump_fp}')
        merge_dataset.to_csv(dump_fp, index=False)