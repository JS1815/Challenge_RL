import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import holidays
from datetime import date, datetime, timedelta

class InterArrivalTimes:
    def __init__(self):
        self.__starting_time = datetime(2018, 1, 1)
        self.__holidays = holidays.Germany(state='HE')

        self.__init__EM_hourly_arrival()

    def sample(self, hours, case_type):
        if case_type == 'EM':
            return self.EM_arrival(hours)
        else:
            return self.A_B_arrival(hours)

    def EM_arrival(self, hours : float) -> float:
        dt = self.get_datetime_from_hours(hours)
        d = self.get_date_from_hours(hours)
        # get hourly basis
        hourly_factor = self.get_EM_hourly_arrival(dt)
        # get weekend and holiday factor
        holiday_factor = 1
        if self.is_weekend(d) or self.is_holiday(d):
            holiday_factor = 2
        # get seasonal factor
        seasonal_factor = self.get_EM_seasonal_factor(d)

        base_arrival = 1
        sample_arrival = random.expovariate(base_arrival * hourly_factor * holiday_factor * seasonal_factor)
        return sample_arrival


    def A_B_arrival(self, hours : float) -> float:
        next_working_time_offset = self.get_next_working_time_offset(hours)

        d = self.get_date_from_hours(hours + next_working_time_offset)
        seasonal_factor = 1 / self.get_EM_seasonal_factor(d)

        base_arrival = 1
        sample_arrival =  next_working_time_offset + random.expovariate(base_arrival * seasonal_factor)

        # arrival time can be sampled out of working time! so adjust to next working day earliest!
        new_sample_arrival = sample_arrival + self.get_next_working_time_offset(hours + sample_arrival)

        # might have some rounding erros, i.e. time is 8.9999 instead of 9, therefore round
        new_sample_arrival = float(int(new_sample_arrival + 0.5))
        return new_sample_arrival


    def get_next_working_time_offset(self, hours : float) -> float:
        starting_dt = self.get_datetime_from_hours(hours)
        dt = self.get_datetime_from_hours(hours)
        hid = self.get_hours_in_day(dt)

        # if out of working time adjust to next day
        if hid > 17:
            dt += timedelta(hours = 9 + (24 - hid))
        elif hid < 9:
            dt += timedelta(hours = 9 - hid)

        # if no workign day adjust to next working day
        while self.is_weekend(dt.date()) or self.is_holiday(dt.date()):
            dt += timedelta(days=1)

        return (dt - starting_dt).total_seconds() / 3600
        
    def get_EM_seasonal_factor(self, d : date) -> float:
        x = d.day
        A = 0.5  # Amplitude
        P = 364  # Period
        C = 1.0  # Vertical offset
        phi = -np.pi / 2  # Phase shift
        return A * np.sin((2 * np.pi / P) * x + phi) + C

    def get_EM_hourly_arrival(self, time : datetime) -> float:
        hour_in_day = self.get_hours_in_day(time)
        x = self.EM_model_encoder.transform([[hour_in_day]])
        return self.EM_model_hourly.predict(x)[0]

    def __init__EM_hourly_arrival(self):
        X = np.array([0, 3, 6, 12, 15, 18]).reshape(-1, 1)  # Features
        y = np.array([0.8, 0.1, 0.7, 1, 0.5, 0.7])     # Target

        # Transform features into polynomial features
        degree = 4
        self.EM_model_encoder = PolynomialFeatures(degree=degree)
        X_poly = self.EM_model_encoder.fit_transform(X)

        self.EM_model_hourly = LinearRegression()
        self.EM_model_hourly.fit(X_poly, y)

    def get_datetime_from_hours(self, hours : float) -> datetime:
        return self.__starting_time + timedelta(hours=hours)
    
    def get_date_from_hours(self, hours : float) -> date:
        return self.__starting_time.date() + timedelta(hours=hours)
    
    def get_hours_in_day(self, time : datetime) -> float:
        return (time - datetime(time.year, time.month, time.day)).total_seconds() / 3600

    def is_holiday(self, time : date) -> bool:
        return time in self.__holidays
    
    def is_weekend(self, time : date) -> bool:
        return time.weekday() > 4