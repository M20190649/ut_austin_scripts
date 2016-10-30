from datetime import datetime, timedelta, time
import json
import urllib

from dateutil import parser as date_parser
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt

from pynwm import nwm


def _get_usgs_flow(gage_id, start_date):
    uri = ('http://waterservices.usgs.gov/nwis/iv/?format=json&sites={0}'
           '&startDT={1}&parameterCd=00060').format(
               gage_id, start_date.isoformat())
    handle = urllib.urlopen(uri)
    response = json.load(handle)['value']['timeSeries']
    if not len(response):
        return None, [], None
    gage = response[0]['sourceInfo']['siteName']
    series = response[0]['values'][0]['value']
    tmp_vals = [s['value'] for s in series]
    tmp_dates = [date_parser.parse(s['dateTime']) for s in series]

    vals = []
    dates = []
    for i, v in enumerate(tmp_vals):
        if float(v) >= 0:
            vals.append(float(v))
            dates.append(tmp_dates[i])
        
    return gage, vals, dates


def _get_limits(gage_vals, gage_dates, s_series, m_series, l_series, clip_to_gage_dates):
    max_val = 0
    max_date = date_parser.parse('1/1/2001 00:00-06')
    min_date = date_parser.parse('1/1/2091 00:00-06')

    if gage_dates:
        if max(gage_dates) > max_date:
            max_date = max(gage_dates)
        if min(gage_dates) < min_date:
            min_date = min(gage_dates)
    if not clip_to_gage_dates:
        if s_series:
            for s in s_series:
                if max(s['dates']) > max_date:
                    max_date = max(s['dates'])
                if min(s['dates']) < min_date:
                    min_date = min(s['dates'])
        if m_series:
            for s in m_series:
                if max(s['dates']) > max_date:
                    max_date = max(s['dates'])
                if min(s['dates']) < min_date:
                    min_date = min(s['dates'])
        if l_series:
            for s in l_series:
                if max(s['dates']) > max_date:
                    max_date = max(s['dates'])
                if min(s['dates']) < min_date:
                    min_date = min(s['dates'])
    
    if gage_vals:
        for i, date in enumerate(gage_dates):
            if date >= min_date and date <= max_date:
                if gage_vals[i] > max_val:
                    max_val = gage_vals[i]
    if s_series:
        for s in s_series:
            for i, date in enumerate(s['dates']):
                if date >= min_date and date <= max_date:
                    if s['values'][i] > max_val:
                        max_val = s['values'][i]
    if m_series:
        for s in m_series:
            for i, date in enumerate(s['dates']):
                if date >= min_date and date <= max_date:
                    if s['values'][i] > max_val:
                        max_val = s['values'][i]
    if l_series:
        for s in l_series:
            for i, date in enumerate(s['dates']):
                if date >= min_date and date <= max_date:
                    if s['values'][i] > max_val:
                        max_val = s['values'][i]
    return max_val, min_date, max_date


def make_graph(products, comid, gage, start_date, clip_to_gage_dates, width):
    print 'COMID:', comid
    fig_size = [width, width / 1.61803398875]  # golden ratio
    gage_vals = None
    gage_dates = None
    s_series = None
    m_series = None
    l_series = None

    # Download data and figure out data ranges
    print '    Getting USGS data'
    gage_start = start_date - timedelta(hours=6)
    gage_name, gage_vals, gage_dates = _get_usgs_flow(gage, gage_start)
    if 'short' in products:
        print '    Getting short range forecast from HydroShare'
        s_series = nwm.get_streamflow('short_range', comid, start_date, 'America/Chicago')
    if 'medium' in products:
        medium_start = datetime.combine(start_date.date(), time(6))
        print '    Getting medium range forecast from HydroShare'
        m_series = nwm.get_streamflow('medium_range', comid, medium_start, 'America/Chicago')
    if 'long' in products:
        print '    Getting long range forecast from HydroShare'
        l_series = nwm.get_streamflow('long_range', comid, start_date, 'America/Chicago')
    max_val, min_date, max_date = _get_limits(gage_vals, gage_dates, s_series, m_series, l_series, clip_to_gage_dates)

    if max_val > 0 and min_date < max_date:
        try:
            # Make the plot
            fig, ax = plt.subplots(figsize=fig_size)
            plt.title(gage_name, fontsize=12)
            if gage_vals:
                ax.plot(gage_dates, gage_vals, label='Gage', color='green', linewidth='4')

            # Short
            if s_series:
                for s in s_series:
                    nwm_dates = s['dates']
                    nwm_vals = s['values']
                    ax.plot(nwm_dates, nwm_vals, label='Short Range', color='blue', linewidth='7')

            # Medium
            if m_series:
                for s in m_series:
                    nwm_dates = s['dates']
                    nwm_vals = s['values']
                    ax.plot(nwm_dates, nwm_vals, label='Medium Range', color='purple', linewidth='2')

            # Long
            if l_series:
                for i, s in enumerate(l_series):
                    nwm_dates = s['dates']
                    nwm_vals = s['values']
                    if i == 0:
                        ax.plot(nwm_dates, nwm_vals, label='Ensemble', color='brown', linestyle=':')
                    else:
                        ax.plot(nwm_dates, nwm_vals, color='brown', linestyle=':')

            # Format plot layout
            ax.legend(loc='best', fontsize=10)
            plt.ylabel('Flow (cfs)', fontsize=12)
            ax.tick_params(labelsize=10)
            ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
            ax.set_ylim(0.0, max_val * 1.05)
            ax.set_xlim(min_date, max_date)

            ax.margins(0, 0.1)
            fig.autofmt_xdate()  # Show date labels at an angle
            fig.tight_layout()  # Removes some of the margin around the graph

            # Save the result
            png_file = '{0}-{1}-{2}.png'.format(gage_name, gage, comid)
            plt.savefig(png_file, facecolor=fig.get_facecolor())
            print '    Saved', png_file
        except:
            raise
        finally:
            plt.close()  # free memory after each plot
    else:
        print '  Max value is zero or no date intersection found. Skipping plot.'

    
if __name__ == '__main__':
    start_date = date_parser.parse('8/17/2016 06:00-05')
    width = 10  # plot width in hectopixels
    products = ['short', 'medium', 'long']  # May have to run a few times if including 'long'
    clip_to_gage_dates = True  # Determines if x-axis will be trimmed to date range of gage data
    
    comid = '1630223'  # 1630223 is Blanco River at Wimberley
    gage = '08171000'  # 08171000 is Blanco River at Wimberley

    make_graph(products, comid, gage, start_date, clip_to_gage_dates, width)
