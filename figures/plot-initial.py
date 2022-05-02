#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from constants import *
import pickle
from conversions import *
import scipy as sp
from scipy.interpolate import interp2d
import numpy as np
from plotTools import imagesc, turn_off_ticks_xaxis, turn_ticklabels_off_xaxis
from constant_labels import create_label
from matplotlib.dates import DateFormatter
from baseflowsep import *
from durationprocessing import *
plt.rcParams.update({'font.size': FONT_SIZ})


def load_pickle_lookat_precip():
    """
    Exploratory file to see which gauges are useful over the period
    :return:
    """
    table = pd.read_pickle(PICKLE_RAW_PRECIPFILE)
    for key, value in table.items():
        print(key)
        try:
            df = table[key]
            precip = df['precipRate']
            plt.figure()
            plt.title(key)
            precip.plot()
            plt.show(block=False)
            plt.close()
        except TypeError:
            print('BAD: ' + key)
            plt.close()
    # DONE


def remove_pickle_bad_lookat():
    badfiles = [
        "IALBERTA68",
        "ICALGA153",
        "ICALGA167",
        "ICALGA182",
        "ICALGA197",
        "ICALGA227",
        "ICALGA230",
        "ICALGA231",
        "ICALGA232",
        "ICALGA232",
        "ICALGA237",
        "ICALGA241",
        "ICALGA242",
        "ICALGA248",
        "ICALGA249",
        "ICALGA25",
        "ICALGA250",
        "ICALGA60",
        "ICALGARY83",
        "ICALGA258",
        "ICALGA264",
        "ICALGA265",
        "ICALGA267",
        "ICALGA271",
        "ICALGA272",
        "ICALGA273",
        "ICALGA274",
        "ICALGA30",
        "ICALGA33",
        "ICALGA39",
        "ICALGA48",
        "ICALGA87",
        "ICALGA163",
        "ICALGA177",
        "ICALGARY101",
        "ICALGARY123",
        "ICALGARY135",
        "ICALGARY23",
        "ICALGARY42",
        "ICOCHR11",
        "ICOCHR18",
        "ICOCHR21",
        "ICALGA204",
        "ICALGA258",
        "ICOCHR23",
        "ICOCHR24",
        "ICOCHR27",
        "IROCKY17",
        "IROCKY22",
        "IROCKY26",
        "IROCKY53",
        "IROCKY55",
        "IROCKY60",
        "IROCKY66",
        "IROCKY80",
        "IROCKYVI30",
        "IROCKYVI17"
    ]
    table = pd.read_pickle(PICKLE_RAW_PRECIPFILE)
    final_dict = {key: table[key] for key in table if key not in badfiles}
    for key, value in final_dict.items():
        print(key)
        try:
            df = table[key]
            precip = df['precipRate']
            plt.figure()
            plt.title(key)
            precip.plot()
            plt.show(block=False)
            plt.close()
        except TypeError:
            print('BAD: ' + key)
            plt.close()
    print('writing to file')
    with open(PICKLE_PRUNED_PRECIPFILE, 'wb') as handle:
        pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_num_rainfall_gauges():
    table = pd.read_pickle(PICKLE_PRUNED_PRECIPFILE)
    print('number of rainfall gauges:')
    print(len(table))


def export_csv_gauge_lat_lng():
    print('gauge locations file')
    precip = pd.read_pickle(PICKLE_PRUNED_PRECIPFILE)
    name = []
    latitude = []
    longitude = []
    for key, df in precip.items():
        lat = df[['lat']].values[0][0]
        lon = df[['lon']].values[0][0]
        name.append(key)
        latitude.append(lat)
        longitude.append(lon)
    n = len(name)
    with open(PRECIP_GAUGE_LOCATIONS_FILE, 'w') as f:
        for k in range(n):
            f.write("{},{},{}\n".format(name[k], latitude[k], longitude[k]))
    print('done')
# DONE


def downsample_precip(start_date, end_date, plot=False):
    p = pd.read_pickle(PICKLE_PRUNED_PRECIPFILE)
    pout = {}
    for key, val in p.items():
        df = p[key]
        precip = df[['precipRate']]
        precip_cut = precip.sort_index()[start_date:end_date]
        precip_resample = precip_cut.resample(rule='H').mean().fillna(0)
        precip_mm = convert_inches_to_mm(precip_resample)
        lat = df[['lat']].values[0][0]
        lon = df[['lon']].values[0][0]
        if lat == 0 or lon == 0:
            continue
        dout = {
            'precip_mm': precip_mm,
            'lat': lat,
            'lon': lon
        }
        pout[key] = dout
        if plot:
            precip_mm.plot()
            plt.title(key)
            block = True
            plt.show(block=block)
            plt.close()
    # DONE loop
    print('writing to file ')
    with open(PICKLE_DS_PRECIPFILE, 'wb') as handle:
        pickle.dump(pout, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done writing to file')


def spatial_average_precip(fn, calc_thiessen=True):
    thiessen_dict = {}
    if calc_thiessen:
        df_area = pd.read_csv(AREA_POLYGON_FILE)
        name = list(df_area['field_1'])
        area = list(df_area['area'])
        thiessen_dict = {name[i]: area[i] for i in range(len(name))}
    p = pd.read_pickle(PICKLE_DS_PRECIPFILE)
    first_key = list(p.keys())[0]
    ts = list(p[first_key]['precip_mm'].index)
    n = len(ts)
    zav = []
    # cycle over all timestamps
    for k in range(n):
        z = []
        for key, val in p.items():  # cycle over all gauges
            gauge_precip_table = list(p[key]['precip_mm']['precipRate'])
            if k < len(gauge_precip_table):
                pt = gauge_precip_table[k]  # precipitation value
                if not calc_thiessen:
                    z.append(pt)
                else:  # thiessen polygons
                    a = thiessen_dict[key]
                    z.append(pt*a)
        # average for each timestep
        z = np.asarray(z)
        if not calc_thiessen:
            zav.append(np.average(z))
        else:  # thiessen polygons
            zav.append(np.sum(z)/TOTAL_AREA_BASIN)
    # DONE for loop
    dictout = {
        'time': ts,
        'precip_mm': zav
    }
    dfout = pd.DataFrame(dictout)
    dfout = dfout.set_index('time')

    dfout.plot()
    plt.show(block=False)

    print('saving')
    with open(fn, 'wb') as handle:
        pickle.dump(dfout, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done')
    # DONE


def set_ticks_plot(n):
    """
    Function to set the number of ticks on the x axis
    :param n:   as the number of ticks
    :return:
    """
    ax = plt.gca()
    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::n])
    ax.xaxis.set_ticklabels(ticklabels[::n])


def plot_precip_inputs(fn_output, fn_in, precip_range, tl=True, block=False):
    df = pd.read_pickle(fn_in)
    fig = df.plot(kind='bar', legend=False, lw=LINE_WIDTH, width=PRECIP_WIDTH).get_figure()
    plt.ylim(precip_range)
    plt.ylabel('Precipitation (mm)')
    plt.xlabel('')
    if tl:
        turn_ticklabels_off_xaxis()
    x_labels = df.index.strftime('%H:00')
    ax = plt.gca()
    ax.set_xticklabels(x_labels)
    set_ticks_plot(5)
    plt.xlabel('Day Hour')
    plt.tight_layout()
    plt.show(block=block)
    fig.savefig(OUT_DIR + fn_output)


def load_save_plot_discharge_data(fn, start_q, end_q):
    """
    This function loads the discharge data, cuts the data, and saves the data
    along with precipitation data in the same dataframe
    :param fn:          as the filename to export
    :param start_q:      as the start date for the discharge
    :param end_q:        as the end date for the discharge
    :return:
    """
    df = pd.read_csv(DISCHARGE_DATA_FILE)
    df_precip = pd.read_pickle(SPATIAL_AV_PRECIP_FN)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    q = df[['Value']]
    q.index += pd.DateOffset(hours=1)  # need to correct for Daylight Savings Time
    qcut = q[start_q:end_q]
    qcut_mean = qcut.rolling(3).mean()
    precip_mm = df_precip.loc[:, 'precip_mm'].tolist()
    index_precip = df_precip.index

    # merge qcut_mean with df_precip tables as a dict
    dout = {
        'precip_mm': precip_mm,
        'index_precip': index_precip,
        'q': qcut_mean.loc[:, 'Value'].tolist(),
        'index_q': qcut_mean.index
    }

    # plot the graph
    fig = qcut.plot(figsize=FIG_SIZE_HYDROGRAPH, lw=LINE_WIDTH, color='green', marker='o',
                    linestyle='None', label='Measured')
    ax1 = plt.gca()
    ax1.plot(qcut_mean, color='darkorange', lw=LINE_WIDTH, label='Moving Average Discharge')
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    plt.xlabel('Timestamp')
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(date_form)
    L = plt.legend(loc='upper right')
    L.get_texts()[0].set_text('Measured Discharge')

    ax2 = ax1.twinx()
    ax2.bar(index_precip, precip_mm, width=0.01, label='Precipitation')
    ax2.invert_yaxis()
    ax2.set_ylabel('Precipitation (mm)')
    plt.tight_layout()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.set_frame_on(False)
    plt.legend(loc='upper left')

    with open(DATA_TABLE_ALL, 'wb') as handle:
        pickle.dump(dout, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.savefig(OUT_DIR + fn)
    plt.show(block=False)
    plt.close()

    fig = qcut_mean.plot(lw=LINE_WIDTH, color='darkorange',
                    linestyle='None', label='Measured', legend=None)
    ax1 = plt.gca()
    ax1.plot(qcut_mean, color='darkorange', lw=LINE_WIDTH, label='Moving Average Discharge')
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    plt.xlabel('Timestamp')
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(date_form)
    plt.tight_layout()
    plt.savefig(OUT_DIR + SAMPLE_DISCHARGE_PLOT)
    plt.show(block=False)
# DONE


def plot_baseflow_separation():
    """
    Plot the baseflow separation example
    :return:
    """
    d = pd.read_pickle(DATA_TABLE_ALL)
    precip_mm = d['precip_mm']
    index_precip = d['index_precip']
    q = d['q']
    index_q = d['index_q']
    alpha = 0.98

    q = q[2:]  # remove nan at the beginning due to the smoothing
    index_q = index_q[2:]
    quick = baseflow_sep_run(q, alpha) # run the algorithm
    base = q - quick
    index_precip = index_precip[2:]  # remove discontinuities
    precip_mm = precip_mm[2:]

    dout = {
        'precip_mm':  precip_mm,
        'index_precip': index_precip,
        'index_q': index_q,
        'quick': quick,
        'base': base
    }
    with open(DATA_TABLE_QUICK, 'wb') as handle:
        pickle.dump(dout, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.figure(figsize=FIG_SIZE_HYDROGRAPH)
    plt.plot(index_q, q, color='darkorange', linewidth=LINE_WIDTH, label='Original Hydrograph')     # original
    plt.plot(index_q, base, color='purple', linewidth=LINE_WIDTH, label='Baseflow Hydrograph')      # baseflow
    plt.legend()
    ax1 = plt.gca()
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(date_form)
    turn_ticklabels_off_xaxis()

    plt.tight_layout()
    plt.savefig(OUT_DIR + BASEFLOW_SEP_PLOT1)
    plt.show(block=False)

    plt.figure(figsize=FIG_SIZE_HYDROGRAPH)
    plt.plot(index_q, quick, color='blue', linewidth=LINE_WIDTH, label='Quickflow Hydrograph')     # quickflow
    plt.legend()
    ax1 = plt.gca()
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(date_form)

    ax2 = ax1.twinx()
    ax2.bar(index_precip, precip_mm, width=0.01, label='Precipitation')
    plt.legend(loc='upper left')
    ax2.invert_yaxis()
    ax2.set_ylabel('Precipitation (mm)')
    plt.tight_layout()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.set_frame_on(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR + BASEFLOW_SEP_PLOT2)
    plt.show(block=False)


def plot_summation_under_curve():
    d = pd.read_pickle(DATA_TABLE_QUICK)
    precip_mm = d['precip_mm']
    index_precip = d['index_precip']
    index_q = d['index_q']
    quick = d['quick']
    base = d['base']
    volume = np.round(np.trapz(quick, dx=60*60))  # numerical integration to get area under curve (3600 seconds in hr)
    print('volume = ', volume)
    excess_precip = volume / TOTAL_AREA_BASIN
    excess_precip_mm = excess_precip / 1.0e-3
    print('Excess precip in mm = ', excess_precip_mm)
    Cf = 1.0 / excess_precip_mm
    print('Cf = ', Cf)
    quick_scaled = Cf * quick  # this is the unit hydrograph

    plt.figure(figsize=FIG_SIZE_HYDROGRAPH)
    plt.plot(index_q, quick, linewidth=LINE_WIDTH, color='blue', label='Quickflow Hydrograph')     # quickflow
    ax1 = plt.gca()
    ax1.fill_between(index_q, quick, 0, color='lightblue')
    plt.legend()
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(date_form)

    plt.tight_layout()
    plt.savefig(OUT_DIR + SUM_UNDER_CURVE_PLOT)
    plt.show(block=False)

    plt.figure()
    plt.plot(index_q, quick, linewidth=LINE_WIDTH, color='blue', label='Quickflow Hydrograph')
    plt.ylim([0, 25])
    ax1 = plt.gca()
    plt.legend()
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(date_form)
    ax1.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(OUT_DIR + SCALE1_PLOT)
    plt.show(block=False)

    plt.figure()
    plt.plot(index_q, quick_scaled, linewidth=LINE_WIDTH, color='blue', label='Unit Hydrograph')
    plt.ylim([0, 25])
    ax1 = plt.gca()
    plt.legend()
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(date_form)
    ax1.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(OUT_DIR + SCALE2_PLOT)
    plt.show(block=False)

    # plot precipitation inputs that are to be truncated
    df = pd.read_pickle(SPATIAL_AV_PRECIP_FN)
    fig = df.plot(kind='bar', legend=False, lw=LINE_WIDTH, width=PRECIP_WIDTH).get_figure()
    plt.ylim(PRECIP_RANGE)
    plt.ylabel('Precipitation (mm)')
    plt.xlabel('')
    plt.axhline(y=excess_precip_mm, color='black', linestyle='--')
    x_labels = df.index.strftime('%H:00')
    ax = plt.gca()
    ax.set_xticklabels(x_labels)
    set_ticks_plot(5)
    plt.xlabel('Day Hour')
    plt.tight_layout()
    plt.savefig(OUT_DIR + EXCESS_PRECIP_PLOT)
    plt.show(block=False)

    # truncate precip and plot
    df_truncated_precip = df
    df_truncated_precip[:'2021-06-05 10:00'] = np.NAN
    fig = df_truncated_precip.plot(kind='bar', legend=False, lw=LINE_WIDTH, width=PRECIP_WIDTH).get_figure()
    plt.ylim(PRECIP_RANGE)
    plt.ylabel('Precipitation (mm)')
    plt.xlabel('')
    x_labels = df.index.strftime('%H:00')
    ax = plt.gca()
    ax.set_xticklabels(x_labels)
    set_ticks_plot(5)
    plt.xlabel('Day Hour')
    plt.tight_layout()
    plt.savefig(OUT_DIR + EXCESS_PRECIP_PLOT1)
    plt.show(block=False)

    # truncate the curves
    nbeginning_q = 130
    index_q_trunc = index_q[nbeginning_q:]
    quick_scaled_trunc = quick_scaled[nbeginning_q:]
    nend_q = 60
    index_q_trunc = index_q_trunc[:nend_q]
    quick_scaled_trunc = quick_scaled_trunc[:nend_q]
    nprecip0 = 9
    index_precip_trunc = index_precip[nprecip0:]
    precip_mm_trunc = precip_mm[nprecip0:]

    # plot the output hydrograph and the precipitation input
    plt.figure(figsize=FIG_SIZE_HYDROGRAPH)
    plt.plot(index_q_trunc, quick_scaled_trunc, color='blue', linewidth=LINE_WIDTH, label='Unit Hydrograph')
    plt.ylim([0, 20])
    plt.legend(loc='upper right')
    ax1 = plt.gca()
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    ax1.xaxis.set_major_formatter(date_form)
    ax2 = ax1.twinx()
    ax2.bar(index_precip_trunc, precip_mm_trunc, width=0.01, label='Precipitation')
    plt.legend(loc='upper left')
    ax2.invert_yaxis()
    ax2.set_ylabel('Precipitation (mm)')
    plt.tight_layout()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.set_frame_on(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR + UNIT_PLOT)
    plt.show(block=False)

    # plot the example input hydrograph
    plt.figure()
    plt.plot(index_q_trunc, quick_scaled_trunc, color='blue', linewidth=LINE_WIDTH, label='Unit Hydrograph')
    plt.xticks(rotation=90)
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    plt.xlabel('Timestamp')
    plt.ylim([0, 20])
    plt.tight_layout()
    plt.savefig(OUT_DIR + EXAMPLE_UNIT_PLOT)
    plt.show(block=False)

    # save out the curves
    # index_q_trunc, quick_scaled_trunc
    # index_precip_trunc, precip_mm_trunc
    dout = {
        'q': quick_scaled_trunc,
        'q_index': index_q_trunc,
        'precip_mm': precip_mm_trunc,
        'precip_mm_index': index_precip_trunc
    }
    with open(FINAL_DATA_OUTPUT_UNIT_HYDROGRAPH, 'wb') as handle:
        pickle.dump(dout, handle, protocol=pickle.HIGHEST_PROTOCOL)
# DONE


def plot_shorter_duration():
    d = pd.read_pickle(FINAL_DATA_OUTPUT_UNIT_HYDROGRAPH)
    q = d['q']
    q_index = d['q_index']
    showplot = True
    Tsample = 1
    n = len(q)
    nup = 60 * 60   # 60 hrs * 60 min = 3600 min
    x = np.linspace(0, n, nup)
    xp = np.linspace(0, n, n)
    qy = np.interp(x, xp, q)
    Tknown = 60     # min
    Treq = 30       # min
    null_after = 725

    fname = OUT_DIR + 'shorter'
    y = shorter_duration(qy, Tknown, Treq, showplot, fname)
    y[null_after:] = 0
    plt.figure()
    plt.plot(y, label='Calculated Hydrograph', linewidth=LINE_WIDTH, color='blue')
    plt.xlabel('Time (minutes)')
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname + '-final')
    plt.show(block=False)
# DONE


def plot_longer_duration():
    d = pd.read_pickle(FINAL_DATA_OUTPUT_UNIT_HYDROGRAPH)
    q = d['q']
    q_index = d['q_index']
    showplot = True
    Tknown = 1
    Treq_n = 2

    fname = OUT_DIR + 'longer-duration'
    y, s = longer_duration(q, Tknown, Treq_n, fname)
    plt.figure()
    plt.plot(s, label='Summed Curve', linewidth=LINE_WIDTH)
    plt.xlabel('Time (hours)')
    plt.ylim([0, 40])
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname + '-summed')
    plt.show(block=False)

    plt.figure()
    plt.plot(y, label='Calculated Hydrograph', linewidth=LINE_WIDTH, color='blue')
    plt.xlabel('Time (hours)')
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    plt.ylim([0, 40])
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.show(block=False)


#######################################################################

def main():
    load_pickle_lookat_precip()
    remove_pickle_bad_lookat()
    get_num_rainfall_gauges()
    export_csv_gauge_lat_lng()

    start_precip = '2021-06-05 00:00'
    end_precip = '2021-06-05 13:00'
    downsample_precip(start_precip, end_precip, plot=False)

    spatial_average_precip(AV_PRECIP_FN, calc_thiessen=False)
    spatial_average_precip(SPATIAL_AV_PRECIP_FN, calc_thiessen=True)

    plot_precip_inputs('averaged-rain.png', AV_PRECIP_FN, PRECIP_RANGE, tl=False, block=False)
    plot_precip_inputs('thiessen-rain.png', SPATIAL_AV_PRECIP_FN, PRECIP_RANGE, tl=False, block=False)

    start_q = '2021-06-05 6:00'
    end_q = '2021-06-06 3:00'
    load_save_plot_discharge_data('main-overview.png', start_q, end_q)

    plot_baseflow_separation()
    plot_summation_under_curve()
    plot_shorter_duration()
    plot_longer_duration()
# DONE


if __name__ == '__main__':
    main()

    """
    Index(['tempHigh', 'tempLow', 'tempAvg', 'windspeedHigh', 'windspeedLow',
           'windspeedAvg', 'windgustHigh', 'windgustLow', 'windgustAvg',
           'dewptHigh', 'dewptLow', 'dewptAvg', 'windchillHigh', 'windchillLow',
           'windchillAvg', 'heatindexHigh', 'heatindexLow', 'heatindexAvg',
           'pressureMax', 'pressureMin', 'pressureTrend', 'precipRate',
           'precipTotal', 'tz', 'id', 'lat', 'lon', 'epoch'],
          dtype='object')
    """

