"""
BSD 3-Clause License
Copyright (c) 2016, Met Office.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of nc-time-axis nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Support for cftime axis in matplotlib.

This code originally taken from the nc-time-axis library.
https://github.com/aulemahal/nc-time-axis
"""

from collections import namedtuple

import cftime
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
import numpy as np

# Lower and upper are in number of days.
FormatOption = namedtuple('FormatOption', ['lower', 'upper', 'format_string'])


class CalendarDateTime(object):
    """
    Container for :class:`cftime.datetime` object and calendar.
    """

    def __init__(self, datetime, calendar):
        self.datetime = datetime
        self.calendar = calendar

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.datetime == other.datetime
            and self.calendar == other.calendar
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        msg = '<{}: datetime={}, calendar={}>'
        return msg.format(type(self).__name__, self.datetime, self.calendar)


class NetCDFTimeDateFormatter(mticker.Formatter):
    """
    Formatter for cftime.datetime data.
    """

    # Some magic numbers. These seem to work pretty well.
    format_options = [
        FormatOption(0.0, 0.2, '%H:%M:%S'),
        FormatOption(0.2, 0.8, '%H:%M'),
        FormatOption(0.8, 15, '%Y-%m-%d %H:%M'),
        FormatOption(15, 90, '%Y-%m-%d'),
        FormatOption(90, 900, '%Y-%m'),
        FormatOption(900, 6000000, '%Y'),
    ]

    def __init__(self, locator, calendar, time_units):
        #: The locator associated with this formatter. This is used to get hold
        #: of the scaling information.
        self.locator = locator
        self.calendar = calendar
        self.time_units = time_units

    def pick_format(self, ndays):
        """
        Returns a format string for an interval of the given number of days.
        """
        for option in self.format_options:
            if option.lower < ndays <= option.upper:
                return option.format_string
        else:
            msg = 'No formatter found for an interval of {} days.'
            raise ValueError(msg.format(ndays))

    def __call__(self, x, pos=0):
        format_string = self.pick_format(ndays=self.locator.ndays)
        dt = cftime.num2date(x, self.time_units, self.calendar)
        return dt.strftime(format_string)


class NetCDFTimeDateLocator(mticker.Locator):
    """
    Determines tick locations when plotting cftime.datetime data.
    """

    def __init__(self, max_n_ticks, calendar, date_unit, min_n_ticks=3):
        # The date unit must be in the form of days since ...

        self.max_n_ticks = max_n_ticks
        self.min_n_ticks = min_n_ticks
        self._max_n_locator = mticker.MaxNLocator(max_n_ticks, integer=True)
        self._max_n_locator_days = mticker.MaxNLocator(
            max_n_ticks, integer=True, steps=[1, 2, 4, 7, 10]
        )
        self.calendar = calendar
        self.date_unit = date_unit
        if not self.date_unit.lower().startswith('days since'):
            msg = 'The date unit must be days since for a NetCDF time locator.'
            raise ValueError(msg)

        self._cached_resolution = {}

    def compute_resolution(self, num1, num2, date1, date2):
        """
        Returns the resolution of the dates (hourly, minutely, yearly), and
        an **approximate** number of those units.
        """
        num_days = float(np.abs(num1 - num2))
        resolution = 'SECONDLY'
        n = mdates.SEC_PER_DAY
        if num_days * mdates.MINUTES_PER_DAY > self.max_n_ticks:
            resolution = 'MINUTELY'
            n = int(num_days / mdates.MINUTES_PER_DAY)
        if num_days * mdates.HOURS_PER_DAY > self.max_n_ticks:
            resolution = 'HOURLY'
            n = int(num_days / mdates.HOURS_PER_DAY)
        if num_days > self.max_n_ticks:
            resolution = 'DAILY'
            n = int(num_days)
        if num_days > 30 * self.max_n_ticks:
            resolution = 'MONTHLY'
            n = num_days // 30
        if num_days > 365 * self.max_n_ticks:
            resolution = 'YEARLY'
            n = abs(date1.year - date2.year)

        return resolution, n

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=1e-7, tiny=1e-13)

        self.ndays = float(abs(vmax - vmin))

        lower = cftime.num2date(vmin, self.date_unit, self.calendar)
        upper = cftime.num2date(vmax, self.date_unit, self.calendar)

        resolution, n = self.compute_resolution(vmin, vmax, lower, upper)

        if resolution == 'YEARLY':
            # TODO START AT THE BEGINNING OF A DECADE/CENTURY/MILLENIUM as
            # appropriate.
            years = self._max_n_locator.tick_values(lower.year, upper.year)
            ticks = [cftime.datetime(int(year), 1, 1, calendar=self.calendar) for year in years]
        elif resolution == 'MONTHLY':
            # TODO START AT THE BEGINNING OF A DECADE/CENTURY/MILLENIUM as
            # appropriate.
            months_offset = self._max_n_locator.tick_values(0, n)
            ticks = []
            for offset in months_offset:
                year = lower.year + np.floor((lower.month + offset) / 12)
                month = ((lower.month + offset) % 12) + 1
                ticks.append(cftime.datetime(int(year), int(month), 1, calendar=self.calendar))
        elif resolution == 'DAILY':
            # TODO: It would be great if this favoured multiples of 7.
            days = self._max_n_locator_days.tick_values(vmin, vmax)
            ticks = [cftime.num2date(dt, self.date_unit, self.calendar) for dt in days]
        elif resolution == 'HOURLY':
            hour_unit = 'hours since 2000-01-01'
            in_hours = cftime.date2num([lower, upper], hour_unit, self.calendar)
            hours = self._max_n_locator.tick_values(in_hours[0], in_hours[1])
            ticks = [cftime.num2date(dt, hour_unit, self.calendar) for dt in hours]
        elif resolution == 'MINUTELY':
            minute_unit = 'minutes since 2000-01-01'
            in_minutes = cftime.date2num([lower, upper], minute_unit, self.calendar)
            minutes = self._max_n_locator.tick_values(in_minutes[0], in_minutes[1])
            ticks = [cftime.num2date(dt, minute_unit, self.calendar) for dt in minutes]
        elif resolution == 'SECONDLY':
            second_unit = 'seconds since 2000-01-01'
            in_seconds = cftime.date2num([lower, upper], second_unit, self.calendar)
            seconds = self._max_n_locator.tick_values(in_seconds[0], in_seconds[1])
            ticks = [cftime.num2date(dt, second_unit, self.calendar) for dt in seconds]
        else:
            msg = 'Resolution {} not implemented yet.'.format(resolution)
            raise ValueError(msg)
        # Some calenders do not allow a year 0.
        # Remove ticks to avoid raising an error.
        if self.calendar in [
            'proleptic_gregorian',
            'gregorian',
            'julian',
            'standard',
        ]:
            ticks = [t for t in ticks if t.year != 0]
        return cftime.date2num(ticks, self.date_unit, self.calendar)


class NetCDFTimeConverter(mdates.DateConverter):
    """
    Converter for cftime.datetime data.
    """

    standard_unit = 'days since 2000-01-01'

    @staticmethod
    def axisinfo(unit, axis):
        """
        Returns the :class:`~matplotlib.units.AxisInfo` for *unit*.
        *unit* is a tzinfo instance or None.
        The *axis* argument is required but not used.
        """
        calendar, date_unit, date_type = unit

        majloc = NetCDFTimeDateLocator(4, calendar=calendar, date_unit=date_unit)
        majfmt = NetCDFTimeDateFormatter(majloc, calendar=calendar, time_units=date_unit)
        if date_type is CalendarDateTime:
            datemin = CalendarDateTime(cftime.datetime(2000, 1, 1), calendar=calendar)
            datemax = CalendarDateTime(cftime.datetime(2010, 1, 1), calendar=calendar)
        else:
            datemin = date_type(2000, 1, 1)
            datemax = date_type(2010, 1, 1)
        return munits.AxisInfo(
            majloc=majloc, majfmt=majfmt, label='', default_limits=(datemin, datemax)
        )

    @classmethod
    def default_units(cls, sample_point, axis):
        """
        Computes some units for the given data point.
        """
        if hasattr(sample_point, '__iter__'):
            # Deal with nD `sample_point` arrays.
            if isinstance(sample_point, np.ndarray):
                sample_point = sample_point.reshape(-1)
            calendars = np.array([point.calendar for point in sample_point])
            if np.all(calendars == calendars[0]):
                calendar = calendars[0]
            else:
                raise ValueError('Calendar units are not all equal.')
            date_type = type(sample_point[0])
        else:
            # Deal with a single `sample_point` value.
            if not hasattr(sample_point, 'calendar'):
                msg = 'Expecting cftimes with an extra ' '"calendar" attribute.'
                raise ValueError(msg)
            else:
                calendar = sample_point.calendar
            date_type = type(sample_point)
        return calendar, cls.standard_unit, date_type

    @classmethod
    def convert(cls, value, unit, axis):
        """
        Converts value, if it is not already a number or sequence of numbers,
        with :func:`cftime.date2num`.
        """
        shape = None
        if isinstance(value, np.ndarray):
            # Don't do anything with numeric types.
            if value.dtype != object:
                return value
            shape = value.shape
            value = value.reshape(-1)
            first_value = value[0]
        else:
            # Don't do anything with numeric types.
            if munits.ConversionInterface.is_numlike(value):
                return value
            first_value = value

        if not isinstance(first_value, (CalendarDateTime, cftime.datetime)):
            raise ValueError(
                'The values must be numbers or instances of '
                '"nc_time_axis.CalendarDateTime" or '
                '"cftime.datetime".'
            )

        if isinstance(first_value, CalendarDateTime):
            if not isinstance(first_value.datetime, cftime.datetime):
                raise ValueError(
                    'The datetime attribute of the '
                    'CalendarDateTime object must be of type '
                    '`cftime.datetime`.'
                )

        if isinstance(value, (CalendarDateTime, cftime.datetime)):
            value = [value]

        if isinstance(first_value, CalendarDateTime):
            result = cftime.date2num(
                [v.datetime for v in value], cls.standard_unit, first_value.calendar
            )
        else:
            result = cftime.date2num(value, cls.standard_unit, first_value.calendar)

        if shape is not None:
            result = result.reshape(shape)

        return result


# Automatically register NetCDFTimeConverter with matplotlib.unit's converter
# dictionary.
if CalendarDateTime not in munits.registry:
    munits.registry[CalendarDateTime] = NetCDFTimeConverter()

CFTIME_TYPES = [
    cftime.DatetimeNoLeap,
    cftime.DatetimeAllLeap,
    cftime.DatetimeProlepticGregorian,
    cftime.DatetimeGregorian,
    cftime.Datetime360Day,
    cftime.DatetimeJulian,
]
for date_type in CFTIME_TYPES:
    if date_type not in munits.registry:
        munits.registry[date_type] = NetCDFTimeConverter()