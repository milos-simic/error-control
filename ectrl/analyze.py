import os 

import numpy as np
import pandas as pd
from plotnine import *
import scipy.integrate

def style(legend_position=None, legend_key_height=10,
    legend_title=True,
    legend_key_width=40, legend_text_size=15):
    if legend_title:
        legend_title = element_text(size=15)
    else:
        legend_title = element_blank()
    my_style = theme(text=element_text(size=15),
                     legend_text=element_text(size=legend_text_size),
                     legend_title=legend_title,
                     legend_key_width=legend_key_width,
                     legend_key_height=legend_key_height,
                     axis_text_x=element_text(size=12),
                     axis_text_y=element_text(size=12))

    if legend_position is not None:
        my_style += theme(legend_position=legend_position,
            legend_box_margin=0)

    return my_style

linetype_scale = scale_linetype_manual(
        values = ['solid', 'dashed', 'dotted', 'dashdot',
                  (0, (2, 1, 2, 2)), (0, (3, 1, 3, 2)),
                  (0, (1, 1, 1, 2))
        ]
)

def factorize(name):
    return f'factor({name})'

def plot_3(df, x, y,
           lcf_parameter,
           facet_parameter=None,
           ab=None,
           ribbon=None,
           x_name='', y_name='', legend_name=None,
           display_plot=True, save_plot=True, prefix='', location='',
           width=5, height=4,
           x_lim=(0, 1),
           y_lim=(0, 1),
           line_size=1.5, legend_title=True,
           legend_position=None, legend_key_width=40, legend_ncol=None,
           legend_text_size=15, legend_key_height=10,
          brewer=True, palette_type='qual', palette_number=2):
    #linetype_parameter = factorize(linetype_parameter)
    #color_parameter = factorize(color_parameter)
    #fill_parameter = factorize(fill_parameter)
    if legend_name is None:
        legend_name = lcf_parameter

    lcf_parameter = factorize(lcf_parameter)
    g = ggplot(df, aes(x=x, y=y, color=lcf_parameter, linetype=lcf_parameter)) +\
        geom_line(size=line_size) +\
        theme_classic() + linetype_scale +\
        style(legend_position=legend_position,
            legend_text_size=legend_text_size,
            legend_key_width=legend_key_width,
            legend_title=legend_title,
            legend_key_height=legend_key_height) +\
        coord_cartesian(ylim=y_lim, xlim=x_lim) +\
        labs(x=x_name, y=y_name, color=legend_name, linetype=legend_name, fill=legend_name)
    if legend_ncol is not None:
        legend_spec = guide_legend(ncol=legend_ncol)
        g = g + guides(fill=legend_spec, color=legend_spec, linetype=legend_spec)
    if brewer:
        g = g + scale_color_brewer(type=palette_type, palette=palette_number)
    if ab is not None:
        slope, intercept = ab
        g = g + geom_abline(aes(slope=slope, intercept=intercept), linetype='dashed')
    if facet_parameter:
        #facet_parameter = '+'.join([factorize(x) for x in facet_parameter.split('+')])
        #print(facet_parameter)
        if '+' in facet_parameter:
            params = facet_parameter.split('+')
            nrow = len(df[params[0]].unique())
            ncol = len(df[params[1]].unique())
        else:
            ncol=None
            nrow=None
        g = g + facet_wrap(f'~{facet_parameter}', ncol=ncol, nrow=nrow)
    if ribbon is not None:
        ymin, ymax = ribbon
        g = g + geom_ribbon(aes(ymin=ymin, ymax=ymax,
                               color=lcf_parameter,
                               linetype=lcf_parameter,
                               fill=lcf_parameter), alpha=0.1)
        if brewer:
            g = g + scale_fill_brewer(type=palette_type, palette=palette_number)
    if save_plot:
        filename = f'{prefix}_{x}__vs__{y}__{lcf_parameter}_{facet_parameter}.jpg'
        filename = os.path.join(location, filename)
        g.save(filename, dpi=500, width=width, height=height)
    if display_plot:
        display(g)
    return g

def partition(mask):
    i = 0
    start = None
    end = None
    endpoints = []
    while i < len(mask):
        while i < len(mask) and mask[i]:
            if start is None:
                start = i
                end = i
            else:
                end = i
            i = i + 1
        if start is not None and end is not None:
            endpoints.append((start, end))
        start = None
        i = i + 1
    return endpoints

def cube_analysis(nominal, empirical, other, nan_policy='ignore'):
    if nan_policy == 'ignore':
        nan_mask = np.isnan(empirical)
        empirical = empirical[~nan_mask]
        other = other[~nan_mask]
        nominal = nominal[~nan_mask]
    
    mask = empirical < nominal
    below = 0
    for (start, end) in partition(mask):
        x = nominal[start : (end + 1)]
        z = empirical[start : (end + 1)]
        below += scipy.integrate.trapezoid(x, x) - scipy.integrate.trapezoid(z, x)
    below = 2 * below
    
    
    mask = empirical > nominal
    above = 0
    for (start, end) in partition(mask):
        x = nominal[start : (end + 1)]
        z = empirical[start : (end + 1)]
        above += scipy.integrate.trapezoid(z, x) - scipy.integrate.trapezoid(x, x)
    above = 2 * above
    
    other_area = scipy.integrate.trapezoid(other, nominal)

    return below, above, other_area

def analyze_numerically(df, params, method=None):
    if method is not None:
        mask = df['method'] == method
    else:
        mask = np.repeat(True, len(df))
    rows = []

    configurations = df[mask][params].drop_duplicates()
    for i, config in configurations.iterrows():
        select = np.copy(mask)
        for param in params:
            select = select & (df[param] == config[param])
        
        nominal = df[select]['nominal'].values
        empirical = df[select]['target_estimate'].values
        other = df[select]['nontarget_estimate'].values
        
        # below, above, other rate's area
        b, a, oa = cube_analysis(nominal, empirical, other)
        
        row = {param : config[param] for param in params}
        row['D(A, OA)'] =  np.linalg.norm([a, oa])
        row['D(A, B)'] = np.linalg.norm([a, b])
        row['D(A, B, OA)'] = np.linalg.norm([a, b, oa])
        row['A'] = a
        row['B'] = b
        row['OA'] = oa
        rows.append(row)
    return pd.DataFrame(rows)

def select(df, choices):
    selection = np.repeat(False, len(df))
    for method in choices:
        mask = df['method'] == method
        for param in choices[method]:
            value = choices[method][param]
            mask = mask & (df[param] == value)
        selection = selection | mask
    return df[selection]

def plot_time(df, 
    x, y,
    cf,
    palette='qual', palette_number=2,
    x_name='', y_name='', cf_name='',
    save_plot=True, location='', name='', width=5, height=4,
    display_plot=True,
    ):
    g = ggplot(df, aes(x=x, y=y, color=cf, fill=cf)) +\
    geom_bar(stat='identity') +\
    theme_classic() +\
    style() +\
    scale_fill_brewer(type=palette, palette=palette_number) +\
    scale_color_brewer(type=palette, palette=palette_number) +\
    labs(x=x_name, y=y_name, fill=cf_name, color=cf_name)

    if save_plot:
        filename = os.path.join(location, name)
        g.save(filename, dpi=500, width=width, height=height)

    if display_plot:
        display(g)