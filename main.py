from __future__ import division
import os
import os.path
import gdxl
import math
import json
import numpy as np
import pandas as pd
import collections
import bokeh.io as bio
import bokeh.layouts as bl
import bokeh.models as bm
import bokeh.models.widgets as bmw
import bokeh.models.sources as bms
import bokeh.models.tools as bmt
import bokeh.plotting as bp
import datetime
import six.moves.urllib.parse as urlp

PLOT_WIDTH = 300
PLOT_HEIGHT = 300
PLOT_FONT_SIZE = 10
PLOT_AXIS_LABEL_SIZE = 8
PLOT_LABEL_ORIENTATION = 45
OPACITY = 0.8
X_SCALE = 1
Y_SCALE = 1
CIRCLE_SIZE = 9
BAR_WIDTH = 0.5
LINE_WIDTH = 2
COLORS = ['#5e4fa2', '#3288bd', '#66c2a5', '#abdda4', '#e6f598', '#fee08b', '#fdae61', '#f46d43', '#d53e4f', '#9e0142']*1000
C_NORM = "#31AADE"
CHARTTYPES = ['Dot', 'Line', 'Bar', 'Area']
AGGREGATIONS = ['None', 'Sum', 'Ave', 'Weighted Ave']

this_dir_path = os.path.dirname(os.path.realpath(__file__))
inflation_mult = 1.2547221 #2004$ to 2015$ 


#Preprocess functions
def scale_column(df, **kw):
    df[kw['column']] = df[kw['column']] * kw['scale_factor']
    return df

def scale_column_filtered(df, **kw):
    cond = df[kw['by_column']].isin(kw['by_vals'])
    df.loc[cond, kw['change_column']] = df.loc[cond, kw['change_column']] * kw['scale_factor']
    return df

def discount_costs(df, **kw):
    #inner join the cost_cat_type.csv table to get types of costs (Capital, Operation)
    cost_cat_type = pd.read_csv(this_dir_path + '/csv/cost_cat_type.csv')
    df = pd.merge(left=df, right=cost_cat_type, on='cost_cat', sort=False)
    #make new column that is the pv multiplier
    df['pv_mult'] = df.apply(lambda x: get_pv_mult(int(x['year']), x['type']), axis=1)
    df['Discounted Cost (2015$)'] = df['Cost (2015$)'] * df['pv_mult']
    return df

#Return present value multiplier
def get_pv_mult(year, type, dinvest=0.054439024, dsocial=0.03, lifetime=20, refyear=2017, lastyear=2050):
    if type == "Operation":
        pv_mult = 1 / (1 + dsocial)**(year - refyear)
    elif type == "Capital":
        pv_mult = CRF(dinvest, lifetime) / CRF(dinvest, min(lifetime, lastyear + 1 - year)) * 1 / (1 + dsocial)**(year - refyear)
    return pv_mult

#Capital recovery factor
def CRF(i,n):
    return i/(1-(1/(1+i)**n))

def pre_elec_price(df, **kw):
    df = df.pivot_table(index=['n','year'], columns='elem', values='value').reset_index()
    df.drop(['t2','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16'], axis='columns', inplace=True)
    df.columns.name = None
    df['t3'] = df['t3'] * inflation_mult
    df['t17'] = df['t17'] * inflation_mult
    df.rename(columns={'t1': 'load', 't3': 'Reg Price (2015$/MWh)', 't17': 'Comp Price (2015$/MWh)'}, inplace=True)
    return df

#Results metadata
results_meta = collections.OrderedDict((
    ('Capacity (GW)',
        {'file': 'CONVqn.gdx',
        'param': 'CONVqnallyears',
        'columns': ['tech', 'n', 'year', 'Capacity (GW)'],
        'default_xaxis': 'year',
        'default_series': 'tech',
        'default_yaxis': 'Capacity (GW)',
        'default_aggregation': 'sum',
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
            {'func': scale_column_filtered, 'args': {'by_column': 'tech', 'by_vals': ['UPV', 'DUPV', 'distPV'], 'change_column': 'Capacity (GW)', 'scale_factor': 1/1.1}},
        ],
        'unit': 'GW',
        'default_chart_type': 'stacked_area',
        }
    ),
    ('Generation (TWh)',
        {'file': 'CONVqn.gdx',
        'param': 'CONVqmnallyears',
        'columns': ['tech', 'n', 'year', 'Generation (TWh)'],
        'default_xaxis': 'year',
        'default_series': 'tech',
        'default_yaxis': 'Generation (TWh)',
        'default_aggregation': 'sum',
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 0.000001, 'column': 'Generation (TWh)'}},
        ],
        'unit': 'TWh',
        'default_chart_type': 'stacked_area',
        }
    ),
    ('Emissions, Fuel, Prices',
        {'file': 'Reporting.gdx',
        'param': 'AnnualReport',
        'columns': ['n', 'year', 'type', 'value'],
        'default_xaxis': 'year',
        'default_series': 'pollu_type',
        'default_yaxis': 'value',
        'default_aggregation': 'none',
        'unit': 'Million tonnes',
        'default_chart_type': 'line',
        }
    ),
    ('BA-level System Cost',
        {'file': 'systemcost.gdx',
        'param': 'aSystemCost_ba',
        'columns': ['cost_cat', 'n', 'year', 'Cost (2015$)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': inflation_mult, 'column': 'Cost (2015$)'}},
            {'func': discount_costs, 'args': {}},
        ],
        }
    ),
    ('System Cost',
        {'file': 'Reporting.gdx',
        'param': 'aSystemCost',
        'columns': ['cost_cat', 'year', 'Cost (2015$)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': inflation_mult, 'column': 'Cost (2015$)'}},
            {'func': discount_costs, 'args': {}},
        ],
        }
    ),
    ('Gen by m',
        {'file': 'CONVqn.gdx',
        'param': 'CONVqmnallm',
        'columns': ['tech', 'n', 'year', 'm', 'Gen'],
        }
    ),
    ('Electricity Price',
        {'file': 'Reporting.gdx',
        'param': 'ElecPriceOut',
        'columns': ['n', 'year', 'elem', 'value'],
        'preprocess': [
            {'func': pre_elec_price, 'args': {}},
        ],
        }
    ),
    ('cap_wind',
        {'file': 'CONVqn.gdx',
        'param': 'Windiallc',
        'columns': ["windtype", "i", "year", "class", "value"],
        }
    ),
    ('cap_wind_nrr',
        {'file': 'CONVqn.gdx',
        'param': 'WR2GOallyears',
        'columns': ["i",  "class",  "windtype", "bin","year", "value"],
        }
    ),
    ('cap_csp_nrr',
        {'file': 'CONVqn.gdx',
        'param': 'CSP2GOallyears',
        'columns': ["i",  "cspclass", "bin","year", "value"],
        }
    ),
    ('cap_upv_nrr',
        {'file': "CONVqn.gdx",
        'param': 'UPVR2GOallyears',
        'columns': ["n",  "upvclass", "bin","year", "value"],
        }
    ),
    ('cap_dupv_nrr',
        {'file': "CONVqn.gdx",
        'param': 'DUPVR2GOallyears',
        'columns': ["n",  "dupvclass", "bin","year", "value"],
        }
    ),
    ('op_cap',
        {'file': "CONVqn.gdx",
        'param': 'OperCONVqnallyears',
        'columns': ["tech", "n", "year", "value"],
        }
    ),
    ('retire',
        {'file': "CONVqn.gdx",
        'param': 'Retireqnallyears',
        'columns': ["tech", "n", "year", "value"],
        }
    ),
    ('rebuild',
        {'file': "CONVqn.gdx",
        'param': 'Rebuildqnallyears',
        'columns': ["tech", "n", "year", "value"],
        }
    ),
    ('upgrade',
        {'file': "CONVqn.gdx",
        'param': 'Upgradeqnallyears',
        'columns': ["techold", "technew", "n", "year", "value"],
        }
    ),
    ('hydb',
        {'file': "CONVqn.gdx",
        'param': 'HydBin_allyrs',
        'columns': ["cat", "class", "n", "year", "value"],
        }
    ),
    ('pshb',
        {'file': "CONVqn.gdx",
        'param': 'PHSBIN_allyrs',
        'columns': ["class", "n", "year", "value"],
        }
    ),
    ('plan_res',
        {'file': "Reporting.gdx",
        'param': 'PlanRes',
        'columns': ["tech", "n", "year", "m", "value"],
        }
    ),
    ('oper_res',
        {'file': "Reporting.gdx",
        'param': 'OperRes',
        'columns': ["tech", "n", "year", "m", "value"],
        }
    ),
    ('vrre',
        {'file': "Reporting.gdx",
        'param': 'VRREOut',
        'columns': ["n","m","year","tech","type","value"],
        }
    ),
    ('trans',
        {'file': "Reporting.gdx",
        'param': 'Transmission',
        'columns': ["n", "n2", "year", "type", "value"],
        }
    ),
    ('annual_rep',
        {'file': "Reporting.gdx",
        'param': 'AnnualReport',
        'columns': ["n", "year", "type", "value"],
        }
    ),
    ('annual_rep',
        {'file': "Reporting.gdx",
        'param': 'AnnualReport',
        'columns': ["n", "year", "type", "value"],
        }
    ),
    ('fuel_cost',
        {'file': "Reporting.gdx",
        'param': 'fuelcost',
        'columns': ["year", "country", "type", "value"],
        }
    ),
    ('reg_fuel_cost',
        {'file': "Reporting.gdx",
        'param': 'reg_fuelcost',
        'columns': ["nerc", "year", "type", "value"],
        }
    ),
    ('switches',
        {'file': "Reporting.gdx",
        'param': 'ReportSwitches',
        'columns': ["class", "switch", "value", "ignore"],
        }
    ),
    ('st_rps',
        {'file': "StRPSoutputs.gdx",
        'param': 'StRPSoutput',
        'columns': ["tech", "st", "year", "value"],
        }
    ),
    ('st_rps_mar',
        {'file': "StRPSoutputs.gdx",
        'param': 'StRPSmarginalout',
        'columns': ["st", "year", "const", "value"],
        }
    ),
    ('st_rps_rec',
        {'file': "StRPSoutputs.gdx",
        'param': 'RECallyears',
        'columns': ["n", "st2", "tech", "year", "value"],
        }
    ),
    ('ac_flow',
        {'file': "Transmission.gdx",
        'param': 'TransFlowAC',
        'columns': ["n", "n2", "year", "m", "value"],
        }
    ),
    ('dc_flow',
        {'file': "Transmission.gdx",
        'param': 'TransFlowDC',
        'columns': ["n", "n2", "year", "m", "value"],
        }
    ),
    ('cont_flow',
        {'file': "Transmission.gdx",
        'param': 'ContractFlow',
        'columns': ["n", "n2", "year", "m", "value"],
        }
    ),
    ('wat_with',
        {'file': "water_output.gdx",
        'param': 'WaterWqctnallyears',
        'columns': ["tech", "cool_tech", "n", "year", "value"],
        }
    ),
    ('wat_cons',
        {'file': "water_output.gdx",
        'param': 'WaterCqctnallyears',
        'columns': ["tech", "cool_tech", "n", "year", "value"],
        }
    ),
    ('wat_access',
        {'file': "water_output.gdx",
        'param': 'WatAccessallyears',
        'columns': ["n", "class", "year", "value"],
        }
    ),
    ('obj_fnc',
        {'file': "z.gdx",
        'param': 'z_allyrs',
        'columns': ["year", "value"],
        }
    ),
))

#Columns metadata.
columns_meta = {
    'tech':{
        'type': 'string',
        'map': this_dir_path + '/csv/tech_map.csv',
        'style': this_dir_path + '/csv/tech_style.csv',
    },
    'n':{
        'type': 'string',
        'join': this_dir_path + '/csv/hierarchy.csv',
    },
    'year':{
        'type': 'number',
        'filterable': True,
        'seriesable': True,
    },
    'm':{
        'type': 'string',
        'style': this_dir_path + '/csv/m_style.csv',
    },
    'cost_cat':{
        'type': 'string',
        'map': this_dir_path + '/csv/cost_cat_map.csv',
        'style': this_dir_path + '/csv/cost_cat_style.csv',
    },
#     'value':{
#         'type': 'number',
#         'colors': pca_colors,
#         'options': ['yaxis'],
#     },
#     'pollu_type':{
#         'type': 'string',
#         'filter': 'single',
#         'options': ['series'],
#     },
}

def initialize(data_file):
    topwdg['meta'] = bmw.Div(text='Meta', css_classes=['meta-dropdown'])
    for col in columns_meta:
        if 'map' in columns_meta[col]:
            topwdg['meta_map_'+col] = bmw.TextInput(title='"'+col+ '" Map', value=columns_meta[col]['map'], css_classes=['wdgkey-meta_map_'+col, 'meta-drop'])
            if init_load and 'meta_map_'+col in wdg_config: topwdg['meta_map_'+col].value = str(wdg_config['meta_map_'+col])
            topwdg['meta_map_'+col].on_change('value', update_meta)
        if 'join' in columns_meta[col]:
            topwdg['meta_join_'+col] = bmw.TextInput(title='"'+col+ '" Join', value=columns_meta[col]['join'], css_classes=['wdgkey-meta_join_'+col, 'meta-drop'])
            if init_load and 'meta_join_'+col in wdg_config: topwdg['meta_join_'+col].value = str(wdg_config['meta_join_'+col])
            topwdg['meta_join_'+col].on_change('value', update_meta)
        if 'style' in columns_meta[col]:
            topwdg['meta_style_'+col] = bmw.TextInput(title='"'+col+ '" Style', value=columns_meta[col]['style'], css_classes=['wdgkey-meta_style_'+col, 'meta-drop'])
            if init_load and 'meta_style_'+col in wdg_config: topwdg['meta_style_'+col].value = str(wdg_config['meta_style_'+col])
            topwdg['meta_style_'+col].on_change('value', update_meta)

    topwdg['runs'] = bmw.TextInput(title='Run(s)', value=data_file, css_classes=['wdgkey-runs'])
    if init_load and 'runs' in wdg_config: topwdg['runs'].value = str(wdg_config['runs'])
    topwdg['runs'].on_change('value', update_runs)

def get_scenarios():
    if topwdg['runs'].value != '':
        scenarios[:] = []
        runs_paths = topwdg['runs'].value.split('|')
        for runs_path in runs_paths:
            runs_path = runs_path.strip()
            #if the path is pointing to a csv file, gather all scenarios from that file
            if os.path.isfile(runs_path) and runs_path.lower().endswith('.csv'):
                custom_sorts['scenario'] = []
                abs_path = str(os.path.abspath(runs_path))
                df_scen = pd.read_csv(abs_path)
                for i_scen, scen in df_scen.iterrows():
                    if os.path.isdir(scen['path']):
                        abs_path_scen = os.path.abspath(scen['path'])
                        if os.path.isdir(abs_path_scen+'/gdxfiles'):
                            custom_sorts['scenario'].append(scen['name'])
                            scenarios.append({'name': scen['name'], 'path': abs_path_scen})
            #Else if the path is pointing to a directory, check if the directory is a run folder
            #containing gdxfiles/ and use this as the lone scenario. Otherwise, it must contain
            #run folders, so gather all of those scenarios.
            elif os.path.isdir(runs_path):
                abs_path = str(os.path.abspath(runs_path))
                if os.path.isdir(abs_path+'/gdxfiles'):
                    scenarios.append({'name': os.path.basename(abs_path), 'path': abs_path})
                else:
                    subdirs = os.walk(abs_path).next()[1]
                    for subdir in subdirs:
                        if os.path.isdir(abs_path+'/'+subdir+'/gdxfiles'):
                            abs_subdir = str(os.path.abspath(abs_path+'/'+subdir))
                            scenarios.append({'name': subdir, 'path': abs_subdir})
        #If we have scenarios, build widgets for scenario filters and result.

        for key in ["filter_scenarios_dropdown", "filter_scenarios", "result"]:
            topwdg.pop(key, None)

        if scenarios:
            labels = [a['name'] for a in scenarios]
            topwdg['filter_scenarios_dropdown'] = bmw.Div(text='Filter Scenarios', css_classes=['filter-scenarios-dropdown'])
            topwdg['filter_scenarios'] = bmw.CheckboxGroup(labels=labels, active=list(range(len(labels))), css_classes=['wdgkey-filter_scenarios'])
            if init_load and 'filter_scenarios' in wdg_config: topwdg['filter_scenarios'].active = wdg_config['filter_scenarios']
            topwdg['result'] = bmw.Select(title='Result', value='None', options=['None']+list(results_meta.keys()), css_classes=['wdgkey-result'])
            if init_load and 'result' in wdg_config: topwdg['result'].value = str(wdg_config['result'])
            topwdg['result'].on_change('value', update_data)
        controls.children = list(topwdg.values())

def get_data():
    result = topwdg['result'].value
    #A result has been selected, so either we retrieve it from result_dfs,
    #which is a dict with one dataframe for each result, or we make a new key in the result_dfs
    if result not in result_dfs:
            result_dfs[result] = None
            cur_scenarios = []
    else:
        cur_scenarios = result_dfs[result]['scenario'].unique().tolist() #the scenarios that have already been retrieved and stored in result_dfs
    #For each selected scenario, retrieve the data from gdx if we don't already have it,
    #and update result_dfs with the new data.
    result_meta = results_meta[result]
    for i in topwdg['filter_scenarios'].active:
        scenario_name = scenarios[i]['name']
        if scenario_name not in cur_scenarios:
            #get the gdx result and preprocess
            df_scen_result = gdxl.get_df(scenarios[i]['path'] + '\\gdxfiles\\' + result_meta['file'], result_meta['param'])
            df_scen_result.columns = result_meta['columns']
            if 'preprocess' in result_meta:
                for preprocess in result_meta['preprocess']:
                    df_scen_result = preprocess['func'](df_scen_result, **preprocess['args'])
            df_scen_result['scenario'] = scenario_name
            if result_dfs[result] is None:
                result_dfs[result] = df_scen_result
            else:
                result_dfs[result] = pd.concat([result_dfs[result], df_scen_result]).reset_index(drop=True)

def process_data():
    global df, columns, discrete, continuous, filterable, seriesable
    df = result_dfs[topwdg['result'].value].copy()

    #apply joins
    for col in df.columns.values.tolist():
        if 'meta_join_'+col in topwdg and topwdg['meta_join_'+col].value != '':
            df_join = pd.read_csv(topwdg['meta_join_'+col].value)
            #remove columns to left of col in df_join
            for c in df_join.columns.values.tolist():
                if c == col:
                    break
                df_join.drop(c, axis=1, inplace=True)
            #remove duplicate rows
            df_join.drop_duplicates(subset=col, inplace=True)
            #merge df_join into df
            df = pd.merge(left=df, right=df_join, on=col, sort=False)

    #apply mappings
    for col in df.columns.values.tolist():
        if 'meta_map_'+col in topwdg and topwdg['meta_map_'+col].value != '':
            df_map = pd.read_csv(topwdg['meta_map_'+col].value)
            #filter out values that aren't in raw column
            df = df[df[col].isin(df_map['raw'].values.tolist())]
            #now map from raw to display
            map_dict = dict(zip(list(df_map['raw']), list(df_map['display'])))
            df[col] = df[col].map(map_dict)

    #apply custom styling
    for col in df.columns.values.tolist():
        if 'meta_style_'+col in topwdg and topwdg['meta_style_'+col].value != '':
            df_style = pd.read_csv(topwdg['meta_style_'+col].value)
            #filter out values that aren't in order column
            df = df[df[col].isin(df_style['order'].values.tolist())]
            #add to custom_sorts with new order
            custom_sorts[col] = df_style['order'].tolist()

    columns = df.columns.values.tolist()
    for c in columns:
        if c in columns_meta and columns_meta[c]['type'] is 'number':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    discrete = [x for x in columns if df[x].dtype == object]
    continuous = [x for x in columns if x not in discrete]
    filterable = discrete+[x for x in continuous if x in columns_meta and columns_meta[x]['filterable']]
    seriesable = discrete+[x for x in continuous if x in columns_meta and columns_meta[x]['seriesable']]
    df[discrete] = df[discrete].fillna('{BLANK}')
    df[continuous] = df[continuous].fillna(0)

def build_widgets():
    global init_load
    wdg.clear()

    wdg['x_dropdown'] = bmw.Div(text='X-Axis (required)', css_classes=['x-dropdown'])
    wdg['x'] = bmw.Select(title='X-Axis (required)', value='None', options=['None'] + columns, css_classes=['wdgkey-x', 'x-drop'])
    wdg['x_group'] = bmw.Select(title='Group X-Axis By', value='None', options=['None'] + seriesable, css_classes=['wdgkey-x_group', 'x-drop'])
    wdg['y_dropdown'] = bmw.Div(text='Y-Axis (required)', css_classes=['y-dropdown'])
    wdg['y'] = bmw.Select(title='Y-Axis (required)', value='None', options=['None'] + columns, css_classes=['wdgkey-y', 'y-drop'])
    wdg['y_agg'] = bmw.Select(title='Y-Axis Aggregation', value='Sum', options=AGGREGATIONS, css_classes=['wdgkey-y_agg', 'y-drop'])
    wdg['y_weight'] = bmw.Select(title='Weighting Factor', value='None', options=['None'] + columns, css_classes=['wdgkey-y_weight', 'y-drop'])
    wdg['series_dropdown'] = bmw.Div(text='Series', css_classes=['series-dropdown'])
    wdg['series_legend'] = bmw.Div(text='', css_classes=['series-drop'])
    wdg['series'] = bmw.Select(title='Separate Series By', value='None', options=['None'] + seriesable, css_classes=['wdgkey-series', 'series-drop'])
    wdg['series_stack'] = bmw.Select(title='Series Stacking', value='Unstacked', options=['Unstacked', 'Stacked'], css_classes=['wdgkey-series_stack', 'series-drop'])
    wdg['explode_dropdown'] = bmw.Div(text='Explode', css_classes=['explode-dropdown'])
    wdg['explode'] = bmw.Select(title='Explode By', value='None', options=['None'] + seriesable, css_classes=['wdgkey-explode', 'explode-drop'])
    wdg['explode_group'] = bmw.Select(title='Group Exploded Charts By', value='None', options=['None'] + seriesable, css_classes=['wdgkey-explode_group', 'explode-drop'])
    wdg['adv_dropdown'] = bmw.Div(text='Advanced Operations', css_classes=['adv-dropdown'])
    wdg['adv_op'] = bmw.Select(title='Operation', value='None', options=['None', 'Difference'], css_classes=['wdgkey-adv_op', 'adv-drop'])
    wdg['adv_col'] = bmw.Select(title='Column', value='None', options=['None'] + columns, css_classes=['wdgkey-adv_col', 'adv-drop'])
    wdg['adv_col_base'] = bmw.Select(title='Base', value='None', options=['None'], css_classes=['wdgkey-adv_col_base', 'adv-drop'])
    wdg['filters'] = bmw.Div(text='Filters', css_classes=['filters-dropdown'])
    for j, col in enumerate(filterable):
        val_list = [str(i) for i in sorted(df[col].unique().tolist())]
        wdg['heading_filter_'+str(j)] = bmw.Div(text=col, css_classes=['filter-head'])
        wdg['filter_'+str(j)] = bmw.CheckboxGroup(labels=val_list, active=list(range(len(val_list))), css_classes=['wdgkey-filter_'+str(j), 'filter'])
    wdg['update'] = bmw.Button(label='Update Filters', button_type='success', css_classes=['filters-update'])
    wdg['adjustments'] = bmw.Div(text='Plot Adjustments', css_classes=['adjust-dropdown'])
    wdg['chart_type'] = bmw.Select(title='Chart Type', value=CHARTTYPES[0], options=CHARTTYPES, css_classes=['wdgkey-chart_type', 'adjust-drop'])
    wdg['plot_width'] = bmw.TextInput(title='Plot Width (px)', value=str(PLOT_WIDTH), css_classes=['wdgkey-plot_width', 'adjust-drop'])
    wdg['plot_height'] = bmw.TextInput(title='Plot Height (px)', value=str(PLOT_HEIGHT), css_classes=['wdgkey-plot_height', 'adjust-drop'])
    wdg['plot_title'] = bmw.TextInput(title='Plot Title', value='', css_classes=['wdgkey-plot_title', 'adjust-drop'])
    wdg['plot_title_size'] = bmw.TextInput(title='Plot Title Font Size', value=str(PLOT_FONT_SIZE), css_classes=['wdgkey-plot_title_size', 'adjust-drop'])
    wdg['opacity'] = bmw.TextInput(title='Opacity (0-1)', value=str(OPACITY), css_classes=['wdgkey-opacity', 'adjust-drop'])
    wdg['x_scale'] = bmw.TextInput(title='X Scale', value=str(X_SCALE), css_classes=['wdgkey-x_scale', 'adjust-drop'])
    wdg['x_min'] = bmw.TextInput(title='X Min', value='', css_classes=['wdgkey-x_min', 'adjust-drop'])
    wdg['x_max'] = bmw.TextInput(title='X Max', value='', css_classes=['wdgkey-x_max', 'adjust-drop'])
    wdg['x_title'] = bmw.TextInput(title='X Title', value='', css_classes=['wdgkey-x_title', 'adjust-drop'])
    wdg['x_title_size'] = bmw.TextInput(title='X Title Font Size', value=str(PLOT_FONT_SIZE), css_classes=['wdgkey-x_title_size', 'adjust-drop'])
    wdg['x_major_label_size'] = bmw.TextInput(title='X Labels Font Size', value=str(PLOT_AXIS_LABEL_SIZE), css_classes=['wdgkey-x_major_label_size', 'adjust-drop'])
    wdg['x_major_label_orientation'] = bmw.TextInput(title='X Labels Degrees', value=str(PLOT_LABEL_ORIENTATION), css_classes=['wdgkey-x_major_label_orientation', 'adjust-drop'])
    wdg['y_scale'] = bmw.TextInput(title='Y Scale', value=str(Y_SCALE), css_classes=['wdgkey-y_scale', 'adjust-drop'])
    wdg['y_min'] = bmw.TextInput(title='Y  Min', value='', css_classes=['wdgkey-y_min', 'adjust-drop'])
    wdg['y_max'] = bmw.TextInput(title='Y Max', value='', css_classes=['wdgkey-y_max', 'adjust-drop'])
    wdg['y_title'] = bmw.TextInput(title='Y Title', value='', css_classes=['wdgkey-y_title', 'adjust-drop'])
    wdg['y_title_size'] = bmw.TextInput(title='Y Title Font Size', value=str(PLOT_FONT_SIZE), css_classes=['wdgkey-y_title_size', 'adjust-drop'])
    wdg['y_major_label_size'] = bmw.TextInput(title='Y Labels Font Size', value=str(PLOT_AXIS_LABEL_SIZE), css_classes=['wdgkey-y_major_label_size', 'adjust-drop'])
    wdg['circle_size'] = bmw.TextInput(title='Circle Size (Dot Only)', value=str(CIRCLE_SIZE), css_classes=['wdgkey-circle_size', 'adjust-drop'])
    wdg['bar_width'] = bmw.TextInput(title='Bar Width (Bar Only)', value=str(BAR_WIDTH), css_classes=['wdgkey-bar_width', 'adjust-drop'])
    wdg['line_width'] = bmw.TextInput(title='Line Width (Line Only)', value=str(LINE_WIDTH), css_classes=['wdgkey-line_width', 'adjust-drop'])
    wdg['download'] = bmw.Button(label='Download csv', button_type='success')
    wdg['export_config'] = bmw.Div(text='Export Config to URL', css_classes=['export-config'])

    wdg['series_legend'].text = build_series_legend()

    #use wdg_config (from 'widgets' parameter in URL query string) to configure widgets.
    if init_load:
        for key in wdg_config:
            if key in wdg:
                if hasattr(wdg[key], 'value'):
                    wdg[key].value = str(wdg_config[key])
                elif hasattr(wdg[key], 'active'):
                    wdg[key].active = wdg_config[key]
        init_load = False

    wdg['chart_type'].on_change('value', update_sel)
    wdg['x'].on_change('value', update_sel)
    wdg['x_group'].on_change('value', update_sel)
    wdg['y'].on_change('value', update_sel)
    wdg['y_agg'].on_change('value', update_sel)
    wdg['y_weight'].on_change('value', update_sel)
    wdg['series'].on_change('value', update_sel)
    wdg['series_stack'].on_change('value', update_sel)
    wdg['explode'].on_change('value', update_sel)
    wdg['explode_group'].on_change('value', update_sel)
    wdg['adv_op'].on_change('value', update_sel)
    wdg['adv_col'].on_change('value', update_adv_col)
    wdg['adv_col_base'].on_change('value', update_sel)
    wdg['plot_title'].on_change('value', update_sel)
    wdg['plot_title_size'].on_change('value', update_sel)
    wdg['plot_width'].on_change('value', update_sel)
    wdg['plot_height'].on_change('value', update_sel)
    wdg['opacity'].on_change('value', update_sel)
    wdg['x_min'].on_change('value', update_sel)
    wdg['x_max'].on_change('value', update_sel)
    wdg['x_scale'].on_change('value', update_sel)
    wdg['x_title'].on_change('value', update_sel)
    wdg['x_title_size'].on_change('value', update_sel)
    wdg['x_major_label_size'].on_change('value', update_sel)
    wdg['x_major_label_orientation'].on_change('value', update_sel)
    wdg['y_min'].on_change('value', update_sel)
    wdg['y_max'].on_change('value', update_sel)
    wdg['y_scale'].on_change('value', update_sel)
    wdg['y_title'].on_change('value', update_sel)
    wdg['y_title_size'].on_change('value', update_sel)
    wdg['y_major_label_size'].on_change('value', update_sel)
    wdg['circle_size'].on_change('value', update_sel)
    wdg['bar_width'].on_change('value', update_sel)
    wdg['line_width'].on_change('value', update_sel)
    wdg['update'].on_click(update_plots)
    wdg['download'].on_click(download)

    controls.children = list(topwdg.values()) + list(wdg.values())

def set_df_plots():
    global df_plots
    df_plots = df.copy()

    #Apply filters
    for j, col in enumerate(filterable):
        active = [wdg['filter_'+str(j)].labels[i] for i in wdg['filter_'+str(j)].active]
        if col in continuous:
            active = [float(i) for i in active]
        df_plots = df_plots[df_plots[col].isin(active)]

    #Scale Axes
    if wdg['x_scale'].value != '' and wdg['x'].value in continuous:
        df_plots[wdg['x'].value] = df_plots[wdg['x'].value] * float(wdg['x_scale'].value)
    if wdg['y_scale'].value != '' and wdg['y'].value in continuous:
        df_plots[wdg['y'].value] = df_plots[wdg['y'].value] * float(wdg['y_scale'].value)

    #Apply Aggregation
    if wdg['y'].value in continuous and wdg['y_agg'].value != 'None':
        groupby_cols = [wdg['x'].value]
        if wdg['x_group'].value != 'None': groupby_cols = [wdg['x_group'].value] + groupby_cols
        if wdg['series'].value != 'None': groupby_cols = [wdg['series'].value] + groupby_cols
        if wdg['explode'].value != 'None': groupby_cols = [wdg['explode'].value] + groupby_cols
        if wdg['explode_group'].value != 'None': groupby_cols = [wdg['explode_group'].value] + groupby_cols
        df_grouped = df_plots.groupby(groupby_cols, sort=False)
        if wdg['y_agg'].value == 'Sum':
            df_plots = df_grouped[wdg['y'].value].sum().reset_index()
        elif wdg['y_agg'].value == 'Ave':
            df_plots = df_grouped[wdg['y'].value].mean().reset_index()
        elif wdg['y_agg'].value == 'Weighted Ave' and wdg['y_weight'].value in continuous:
            df_plots = df_grouped.apply(wavg, wdg['y'].value, wdg['y_weight'].value).reset_index()
            df_plots.rename(columns={0: wdg['y'].value}, inplace=True)

    #Do Advanced Operations
    op = wdg['adv_op'].value
    col = wdg['adv_col'].value
    col_base = wdg['adv_col_base'].value
    y_val = wdg['y'].value
    y_agg = wdg['y_agg'].value
    if op != 'None' and col != 'None' and col in df_plots and col_base != 'None' and y_agg != 'None' and y_val in continuous:
        if col in continuous and col_base != 'Consecutive':
            col_base = float(col_base)
        col_list = df_plots[col].unique().tolist()
        if col_base != 'Consecutive':
            col_list.remove(col_base)
            col_list = [col_base] + col_list
        df_plots['tempsort'] = df_plots[col].map(lambda x: col_list.index(x))
        df_plots = df_plots.sort_values('tempsort').reset_index(drop=True)
        df_plots.drop(['tempsort'], axis='columns', inplace=True)
        groupcols = [i for i in df_plots.columns.values.tolist() if i not in [col, y_val]]
        if op == 'Difference':
            if col_base == 'Consecutive':
                df_plots[y_val] = df_plots.groupby(groupcols, sort=False)[y_val].diff()
            else:
                df_plots[y_val] = df_plots[y_val] - df_plots.groupby(groupcols, sort=False)[y_val].transform('first')
        df_plots = df_plots[~df_plots[col].isin([col_base])]
        df_plots = df_plots[pd.notnull(df_plots[y_val])]

    #Sort Dataframe
    sortby_cols = [wdg['x'].value]
    if wdg['x_group'].value != 'None': sortby_cols = [wdg['x_group'].value] + sortby_cols
    if wdg['series'].value != 'None': sortby_cols = [wdg['series'].value] + sortby_cols
    if wdg['explode'].value != 'None': sortby_cols = [wdg['explode'].value] + sortby_cols
    if wdg['explode_group'].value != 'None': sortby_cols = [wdg['explode_group'].value] + sortby_cols
    #change sortby_cols if there is custom sorting, and add custom sorting columns
    temp_sort_cols = sortby_cols[:]
    for col in custom_sorts:
        if col in sortby_cols:
            df_plots[col + '__sort_col'] = df_plots[col].map(lambda x: custom_sorts[col].index(x))
            temp_sort_cols[sortby_cols.index(col)] = col + '__sort_col'
    #Do sorting
    df_plots = df_plots.sort_values(temp_sort_cols).reset_index(drop=True)
    #Remove custom sort columns
    for col in custom_sorts:
        if col in sortby_cols:
            df_plots = df_plots.drop(col + '__sort_col', 1)

    #Rearrange column order for csv download
    unsorted_columns = [col for col in df_plots.columns if col not in sortby_cols + [wdg['y'].value]]
    df_plots = df_plots[sortby_cols + unsorted_columns + [wdg['y'].value]]

def create_figures():
    plot_list = []
    df_plots_cp = df_plots.copy()
    if wdg['explode'].value == 'None':
        plot_list.append(create_figure(df_plots_cp))
    else:
        if wdg['explode_group'].value == 'None':
            for explode_val in df_plots_cp[wdg['explode'].value].unique().tolist():
                df_exploded = df_plots_cp[df_plots_cp[wdg['explode'].value].isin([explode_val])]
                plot_list.append(create_figure(df_exploded, explode_val))
        else:
            for explode_group in df_plots_cp[wdg['explode_group'].value].unique().tolist():
                df_exploded_group = df_plots_cp[df_plots_cp[wdg['explode_group'].value].isin([explode_group])]
                for explode_val in df_exploded_group[wdg['explode'].value].unique().tolist():
                    df_exploded = df_exploded_group[df_exploded_group[wdg['explode'].value].isin([explode_val])]
                    plot_list.append(create_figure(df_exploded, explode_val, explode_group))
    plots.children = plot_list

def create_figure(df_exploded, explode_val=None, explode_group=None):
    # If x_group has a value, create a combined column in the dataframe for x and x_group
    x_col = wdg['x'].value
    if wdg['x_group'].value != 'None':
        x_col = str(wdg['x_group'].value) + '_' + str(wdg['x'].value)
        df_exploded[x_col] = df_exploded[wdg['x_group'].value].map(str) + ' ' + df_exploded[wdg['x'].value].map(str)

    #Build x and y ranges and figure title
    kw = dict()

    #Set x and y ranges. When x is grouped, there is added complication of separating the groups
    xs = df_exploded[x_col].values.tolist()
    ys = df_exploded[wdg['y'].value].values.tolist()
    if wdg['x_group'].value != 'None':
        kw['x_range'] = []
        unique_groups = df_exploded[wdg['x_group'].value].unique().tolist()
        unique_xs = df_exploded[wdg['x'].value].unique().tolist()
        for i, ugr in enumerate(unique_groups):
            for uxs in unique_xs:
                kw['x_range'].append(str(ugr) + ' ' + str(uxs))
            #Between groups, add entries that consist of spaces. Increase number of spaces from
            #one break to the next so that each entry is unique
            kw['x_range'].append(' ' * (i + 1))
    elif wdg['x'].value in discrete:
        kw['x_range'] = df_exploded[x_col].unique().tolist()
    if wdg['y'].value in discrete:
        kw['y_range'] = df_exploded[wdg['y'].value].unique().tolist()

    #Set figure title
    kw['title'] = wdg['plot_title'].value
    seperator = '' if kw['title'] == '' else ', '
    if explode_val is not None:
        if explode_group is not None:
            kw['title'] = kw['title'] + seperator + "%s = %s" % (wdg['explode_group'].value, str(explode_group))
        seperator = '' if kw['title'] == '' else ', '
        kw['title'] = kw['title'] + seperator + "%s = %s" % (wdg['explode'].value, str(explode_val))

    #Add figure tools
    hover = bmt.HoverTool(
            tooltips=[
                ("ser", "@ser_legend"),
                ("x", "@x_legend"),
                ("y", "@y_legend"),
            ]
    )
    TOOLS = [bmt.BoxZoomTool(), bmt.PanTool(), hover, bmt.ResetTool(), bmt.SaveTool()]

    #Create figure with the ranges, titles, and tools, and adjust formatting and labels
    p = bp.figure(plot_height=int(wdg['plot_height'].value), plot_width=int(wdg['plot_width'].value), tools=TOOLS, **kw)
    p.toolbar.active_drag = TOOLS[0]
    p.title.text_font_size = wdg['plot_title_size'].value + 'pt'
    p.xaxis.axis_label = wdg['x_title'].value
    p.yaxis.axis_label = wdg['y_title'].value
    p.xaxis.axis_label_text_font_size = wdg['x_title_size'].value + 'pt'
    p.yaxis.axis_label_text_font_size = wdg['y_title_size'].value + 'pt'
    p.xaxis.major_label_text_font_size = wdg['x_major_label_size'].value + 'pt'
    p.yaxis.major_label_text_font_size = wdg['y_major_label_size'].value + 'pt'
    p.xaxis.major_label_orientation = 'horizontal' if wdg['x_major_label_orientation'].value == '0' else math.radians(float(wdg['x_major_label_orientation'].value))
    if wdg['x'].value in continuous:
        if wdg['x_min'].value != '': p.x_range.start = float(wdg['x_min'].value)
        if wdg['x_max'].value != '': p.x_range.end = float(wdg['x_max'].value)
    if wdg['y'].value in continuous:
        if wdg['y_min'].value != '': p.y_range.start = float(wdg['y_min'].value)
        if wdg['y_max'].value != '': p.y_range.end = float(wdg['y_max'].value)

    #Add glyphs to figure
    c = C_NORM
    if wdg['series'].value == 'None':
        if wdg['y_agg'].value != 'None' and wdg['y'].value in continuous:
            xs = df_exploded[x_col].values.tolist()
            ys = df_exploded[wdg['y'].value].values.tolist()
        add_glyph(p, xs, ys, c)
    else:
        full_series = df_plots[wdg['series'].value].unique().tolist() #for colors only
        if wdg['series_stack'].value == 'Stacked':
            xs_full = df_exploded[x_col].unique().tolist()
            y_bases_pos = [0]*len(xs_full)
            y_bases_neg = [0]*len(xs_full)
        for i, ser in enumerate(df_exploded[wdg['series'].value].unique().tolist()):
            c = COLORS[full_series.index(ser)]
            df_series = df_exploded[df_exploded[wdg['series'].value].isin([ser])]
            xs_ser = df_series[x_col].values.tolist()
            ys_ser = df_series[wdg['y'].value].values.tolist()
            if wdg['series_stack'].value == 'Unstacked':
                add_glyph(p, xs_ser, ys_ser, c, series=ser)
            else:
                ys_pos = [ys_ser[xs_ser.index(x)] if x in xs_ser and ys_ser[xs_ser.index(x)] > 0 else 0 for i, x in enumerate(xs_full)]
                ys_neg = [ys_ser[xs_ser.index(x)] if x in xs_ser and ys_ser[xs_ser.index(x)] < 0 else 0 for i, x in enumerate(xs_full)]
                ys_stacked_pos = [ys_pos[i] + y_bases_pos[i] for i in range(len(xs_full))]
                ys_stacked_neg = [ys_neg[i] + y_bases_neg[i] for i in range(len(xs_full))]
                add_glyph(p, xs_full, ys_stacked_pos, c, y_bases=y_bases_pos, series=ser)
                add_glyph(p, xs_full, ys_stacked_neg, c, y_bases=y_bases_neg, series=ser)
                y_bases_pos = ys_stacked_pos
                y_bases_neg = ys_stacked_neg
    return p

def add_glyph(p, xs, ys, c, y_bases=None, series=None):
    alpha = float(wdg['opacity'].value)
    y_unstacked = list(ys) if y_bases is None else [ys[i] - y_bases[i] for i in range(len(ys))]
    ser = ['None']*len(xs) if series is None else [series]*len(xs)
    if wdg['chart_type'].value == 'Dot':
        source = bms.ColumnDataSource({'x': xs, 'y': ys, 'x_legend': xs, 'y_legend': y_unstacked, 'ser_legend': ser})
        p.circle('x', 'y', source=source, color=c, size=int(wdg['circle_size'].value), fill_alpha=alpha, line_color=None, line_width=None)
    elif wdg['chart_type'].value == 'Line':
        source = bms.ColumnDataSource({'x': xs, 'y': ys, 'x_legend': xs, 'y_legend': y_unstacked, 'ser_legend': ser})
        p.line('x', 'y', source=source, color=c, alpha=alpha, line_width=float(wdg['line_width'].value))
    elif wdg['chart_type'].value == 'Bar':
        if y_bases is None: y_bases = [0]*len(ys)
        centers = [(ys[i] + y_bases[i])/2 for i in range(len(ys))]
        heights = [abs(ys[i] - y_bases[i]) for i in range(len(ys))]
        source = bms.ColumnDataSource({'x': xs, 'y': centers, 'x_legend': xs, 'y_legend': y_unstacked, 'h': heights, 'ser_legend': ser})
        p.rect('x', 'y', source=source, height='h', color=c, fill_alpha=alpha, width=float(wdg['bar_width'].value), line_color=None, line_width=None)
    elif wdg['chart_type'].value == 'Area':
        if y_bases is None: y_bases = [0]*len(ys)
        xs_around = xs + xs[::-1]
        ys_around = y_bases + ys[::-1]
        source = bms.ColumnDataSource({'x': xs_around, 'y': ys_around})
        p.patch('x', 'y', source=source, alpha=alpha, fill_color=c, line_color=None, line_width=None)


def build_series_legend():
    series_legend_string = '<div class="legend-header">Series Legend</div><div class="legend-body">'
    if wdg['series'].value != 'None':
        active_list = df_plots[wdg['series'].value].unique().tolist()
        series_iter = list(enumerate(active_list))
        if wdg['series_stack'].value == 'Stacked': series_iter = reversed(series_iter)
        for i, txt in series_iter:
            series_legend_string += '<div class="legend-entry"><span class="legend-color" style="background-color:' + str(COLORS[i]) + ';"></span>'
            series_legend_string += '<span class="legend-text">' + str(txt) +'</span></div>'
    series_legend_string += '</div>'
    wdg['series_legend'].text =  series_legend_string

def wavg(group, avg_name, weight_name):
    """ http://pbpython.com/weighted-average.html
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return 0

def update_runs(attr, old, new):
    get_scenarios()

def update_data(attr, old, new):
    get_data_and_build()

def get_data_and_build():
    if 'result' in topwdg and topwdg['result'].value is not 'None':
        get_data()
        process_data()
        build_widgets()
        update_plots()

def update_adv_col(attr, old, new):
    if wdg['adv_col'].value != 'None':
        wdg['adv_col_base'].options = ['None', 'Consecutive'] + [str(i) for i in sorted(df[wdg['adv_col'].value].unique().tolist())]
        update_plots()

def update_sel(attr, old, new):
    update_plots()

def update_meta(attr, old, new):
    if 'result' in topwdg and topwdg['result'].value is not 'None':
        process_data()
        build_widgets()
        update_plots()

def update_plots():
    if 'x' not in wdg or wdg['x'].value == 'None' or wdg['y'].value == 'None':
        plots.children = []
        return
    set_df_plots()
    build_series_legend()
    create_figures()

def download():
    df_plots.to_csv(this_dir_path + '/downloads/out '+datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")+'.csv', index=False)

#read 'widgets' parameter from URL query string and use to set data source (data_file)
#and widget configuration object (wdg_config)
init_load = True
wdg_config = {}
custom_sorts = {}
scenarios = []
result_dfs = {}
data_file = ''
args = bio.curdoc().session_context.request.arguments
wdg_arr = args.get('widgets')
if wdg_arr is not None:
    wdg_config = json.loads(urlp.unquote(wdg_arr[0].decode('utf-8')))
    if 'data' in wdg_config:
        data_file = str(wdg_config['data'])

#build widgets and plots
wdg = collections.OrderedDict()
topwdg = collections.OrderedDict()
initialize(data_file)
controls = bl.widgetbox(list(topwdg.values()), id='widgets_section')
plots = bl.column([], id='plots_section')
get_scenarios()
get_data_and_build()
layout = bl.row(controls, plots, id='layout')

bio.curdoc().add_root(layout)
bio.curdoc().title = "Exploding Pivot Chart Maker"