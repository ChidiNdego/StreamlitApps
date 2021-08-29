######################
# Import libraries
######################

import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import matplotlib.pyplot as plt
import re
import itertools as it

import os
import glob
import os
import glob
import streamlit as st
import altair as alt
from PIL import Image

image = Image.open("EmPowerYu\Images\empoweryu_logo.jpg")
st.image(image, width=200)

class BehavioralDataCleaner:

  '''
  WRITE DOC
  '''
  def __init__(self, person_id, name, batch = 'omdena_extract-2021-08-18'):
    self.id = person_id
    self.name = name
    self.batch = batch
    self.path = self.get_file_path()

  def get_file_path(self):
    '''
    Get file path of a certain file in a certain batch (omdena_extract).
    'omdena_extract-2021-08-18' batch currently has all the behavioral data.

    Parameters:
    -----------

    name: Name of file. Possible values:
          'person' for person.txt
          'status' for home_status_event.txt
          'meal' for meal_period.txt

          To be updated to include 'device_event' and 'device'

    batch: Name of batch (omdena_extract).
          Currently "omdena_extract-2021-08-18" batch contains all prod. data.
    '''

    # Dictionary linking names to file names
    file_dict = {'person': 'person.txt', 'status': 'home_status_event.txt', 'meal': 'meal_period.txt'}

    # This function assumes all batch dirs are in "epyu_data" dir
    batch_path = 'EmPowerYu/Dataset/epyu_behavioral_data/' + self.batch

    file_paths = []

    # List of file paths in this batch dir
    file_paths = glob.glob(batch_path + '/*.txt')

    if self.name not in file_dict.keys():
      print('Name parameter not understood.\nEnter one of the following: {}'.format(list(file_dict.keys())))
      return None
    

    # Getting file path from list of file paths
    file_path = [file for file in file_paths if file.endswith(file_dict[self.name])][0]

    return file_path


  def clean(self):
        """
        Read dataframe, then:
        1. Define column names
        2. Extract the required id
        3. Drop instances with duplicate times and keep the last one
           (Others will have 0 duration)
           Ex:
            0 2020-08-15 23:15 sleep
            1 2020-08-15 02:50 active
            2 2020-08-15 02:50 sleep
            In this case index (1) will be drop, the duration of active status
            in this case is 2:50 --> 2:50 = 0 min

        4. Reset index

        To be updated to clean device_event datasets
        """
        
        df = pd.read_csv(self.path, sep = '\t', header = None, parse_dates = [1])
        
        # Dict. linking file type to column headers
        cols_dict = {
            'person': ['id', 'date', 'loc', 'factor', 'gender'],
            'meal': ['id', 'date', 'duration', 'score'],
            'status': ['id', 'date', 'status', 'loc']
        }
        # Assigning column names
        df.columns = cols_dict[self.name]
        
        # Extracting id
        cond = df['id'] == self.id
        
        # Check if person_id is available
        # Replace with raise error
        if df[cond].shape[0] == 0:
          print('Person {} not found in behavioral data, available ids: {}'.format(self.id, df['id'].unique()))
          return None
        
        df = df[cond]
        
        # Dropping duplicated date values to avoid problems with asfreq()
        # These will be kitchen activity scores or home statuses
        # with 0 durations.
        df = df.drop_duplicates(keep = 'last', subset = ['date'])
        
        # Resetting index
        df = df.reset_index(drop = True)
        
        self.df = df

        return df

  def get_status_duration(self, freq = None):
    """
    NOTE: This method is only applicable to home_status_event datasets.
    1. Change the time frequency to the specified freq value (if a freq value
       is defined).
    2. Create columns for each status containing the duration of that status
       in seconds.
       The status duration is the time difference between every status record
       and the following status record.
       Ex:
       0 17:15 active
       1 18:15 sleep

       At 18:15, the system changed the home status from active to sleep,
       therefore active status duration is 17:15 --> 18:15 = 1h.

    Parameters:
    -----------

    freq: Frequency in minutes to resample the dataframe according to.
          If None is passed the frequency is not changed.
    """

    if self.name != 'status':
      print('This method is only applicable to home_status_event datasets.')
      return None

    if not hasattr(self, 'df'):
      self.df = self.clean()

    df = self.df.copy()

    # Changing frequency to the defined freq value
    if freq is not None:
      df = df.set_index('date').asfreq(DateOffset(minutes = freq), method = 'ffill').reset_index()

    # Creating status duration column
    df['duration'] = pd.Timedelta(days = 1) - df['date'].diff(periods = -1) - pd.Timedelta(days = 1)

    # Changing duration dtype
    # NaN value will be at the end of the dataframe
    # where the following status is not available.
    df['duration'] = df['duration'].astype('timedelta64[s]').fillna(0.0)

    # Creating a column for every home status
    categories = pd.get_dummies(df['status'])

    for col in categories.columns:
      df.loc[:, col] = pd.Series(np.where(categories[col] == 1, df['duration'], 0))

    df = df.drop(columns = ['status', 'duration'])

    return df

  def get_cum_mean(self, hours, cols = None, suffix = '_mean'):

    """
    Compute the mean score/duration of a certain period in a dataframe.
    For every data point, the function computes the mean score/duration
    of n hours preceding the data point.

    Parameters:
    -----------

    hours: The number of hours to compute the mean of.

    cols: List of the column names to compute to apply function on.
          For status datasets this could be one or more of the following:
          ['active', 'still', 'sleep', 'away']

          For meal datasets this is ['score'] as default.

          Function will use all possible columns if None is passed.


    suffix: Suffix of the new column containing means.
            (Added to the column original name)
    """
    
    if self.name == 'meal':

      # Checking if data has been cleaned
      if not hasattr(self, 'df'):
        self.df = self.clean()

      df = self.df.copy()
      cols = ['score']

      # Resampling to 0.5 h intervals
      # In general, meal dataset has a 0.5h time step, however there are some
      # larger time steps, especially during sleep time. These are assumed
      # to have zero score.
      df = df.set_index('date').asfreq(DateOffset(minutes = 30)).reset_index()
      df['score'] = df['score'].fillna(0.0)
      df['id'] = df['id'].fillna(self.id).astype('int')
      df = df.drop(columns = ['duration'])

    if self.name == 'status':
      df = self.get_status_duration(freq = 10)
      df = df.drop(columns = ['loc'])

      if cols is None:
        cols = ['active', 'still', 'sleep', 'away']

      # Making sure all columns are available
      # In some cases a status may not be available
      # Ex: Person 21 in year 2020 has only active and sleep statuses
      cols = df.columns[df.columns.isin(cols)]

      if len(cols)<1:
        cols = ['active', 'still', 'sleep', 'away']
        cols = df.columns[df.columns.isin(cols)]
        print('Columns not found, possible columns: ', cols)


    # Date frequency in minutes
    freq = (df.loc[2, 'date'] - df.loc[1, 'date']).value / 60e9

    for col in cols:
      for hour in hours:
        # New column name (Ex: sleep_6h_mean)
        cum_mean_col = col + '_' + str(hour) + 'h' + suffix
        df[cum_mean_col] = 0.0

        # Computing period: Number of instances to compute the mean score/duration of
        # period = (num of hours required / date frequency in hours)
        period = int(60 * hour / freq)

        # Summing scores/durations during the specified time interval (hours)
        for i in range(period):
          df[cum_mean_col] = df[cum_mean_col] + df[col].shift(i).fillna(0.0)

        # Dividing by hours for mean score/duration in one hour
        df[cum_mean_col] = df[cum_mean_col] / hour

    
    # Dropping status duration columns
    if self.name == 'status':
      cols = ['active', 'still', 'sleep', 'away']
      cols = df.columns[df.columns.isin(cols)]

    df = df.drop(columns = cols)

    return df

## Cleaner Class
def get_person_paths(person_id):
  '''
  Get file paths of the given person_id
  '''
  # This function assumes all batch dirs are in "epyu_data" dir
  batch_paths = os.listdir(r'EmPowerYu\Dataset\epyu_data')

  file_paths = []

  # Iterating through batch dirs to get all file paths
  for batch_path in batch_paths:
    batch_path = r'EmPowerYu/Dataset/epyu_data/' + batch_path

    # List of file names in this batch dir
    files = list(map(lambda x: batch_path + '/' + x, os.listdir(batch_path)))

    # List containing all file paths
    file_paths += files
    #print(file_paths)

  # Iterating through file paths to get the required person data
  person_paths = []
  for path in file_paths:
    id = re.search(r'epyu weights person (\d+)', path).group(1)
    if person_id != int(id):
      continue
    person_paths.append(path)

  return person_paths


def check_dt_format(data):
  '''
  This function drops date/time entries that do not comply
  with date/time formats.
  '''
  df = data[['date', 'time']].copy()

  # Regex to match dates
  date_re = re.compile(r'\d+(/|-){1}\d+(/|-){1}\d+')

  # Regex to match times
  time_re = re.compile(r'\d{1,2}:([0-5][0-9]):*([0-5][0-9])*( [A|P]M)*')

  # Regex to match 24:xx:xx
  time_24_re = re.compile(r'24:([0-5][0-9]):*([0-5][0-9])*( [A|P]M)*')

  # Check date entries that do not comply with date formats
  # and replace with NaN
  cond = df['date'].str.fullmatch(date_re).fillna(False)
  df.loc[~cond, 'date'] = np.nan

  # Check time entries that do not comply with time formats
  # and replace with nan
  cond = df['time'].str.fullmatch(time_re).fillna(False)
  df.loc[~cond, 'time'] = np.nan

  # Check time entries with 00:00 written as 24:00
  # and replace with 00:00
  cond = df['time'].str.fullmatch(time_24_re).fillna(False)
  df.loc[cond, 'time'] = df.loc[cond, 'time'].str.replace(r'^24', '00')

  return df

def rm_wrong_years(data):
  '''
  Fix wrong year entries
  This function assumes that there are segments with wrong year entries
  where there would be a date difference of +365 (for an added year)
  at the beginning of the segment and -365 difference at its end.
  '''

  df = data[['date']].copy()
  
  # Creating date difference column
  df['_diff'] = df['date'].diff()

  # Checking for date differences greater than +-350 days
  cond = (df['_diff'] > pd.Timedelta(days = 350)) | (df['_diff'] < -pd.Timedelta(days = 350))
  idx = df[cond].index

  if idx.shape[0] > 0:

    # If there are segments with wrong year entries, there would
    # be an even number of (+-350)+ day differences (start and end)
    # (There are two cases in person 21 data)

    if idx.shape[0]%2 != 0:
      print('   There is a year gap in the data.')

    else:
      for i in range(0, int(idx.shape[0]), 2):
        st = idx[i]
        end = idx[i+1]

        if ((df.loc[st, '_diff'] > pd.Timedelta(days = 350)) and (df.loc[end, '_diff'] < -pd.Timedelta(days = 350)))\
           or\
           ((df.loc[st, '_diff'] < -pd.Timedelta(days = 350)) and (df.loc[end, '_diff'] > pd.Timedelta(days = 350))):

          # Computing year difference
          wrong_year = df.loc[st, 'date'].year
          true_year = df.loc[st - 1, 'date'].year
          dt = wrong_year - true_year

          # Creating DateOffset object
          dt = pd.DateOffset(years = dt)

          # Modifying wrong date entries
          df.loc[st:end-1, 'date'] = df.loc[st:end-1, 'date'] - dt

          st_date = df.loc[st, 'date'].date().strftime(format = '%Y-%m-%d')
          end_date = df.loc[end, 'date'].date().strftime(format = '%Y-%m-%d')
          print('   Changed wrong year entries between dates: {}  -  {}'.format(st_date, end_date))
          print('     Replaced year {} with year {}'.format(wrong_year, true_year))

  return df.drop(columns = ['_diff'])

def clear_duplicate_dates(data, max_dt = pd.Timedelta(minutes = 60)):
    """
    Add pd.Timdelta() to duplicate dates to avoid coinciding dates.
    The function adds the defined max_dt if possible to the duplicate dates.
    If the difference between the duplicate dates and the following dates
    is less than the defined max_dt, a fraction of that difference is used instead of max_t.
    
    Ex:
    ---
    
    2021-08-15 17:55
    2021-08-15 17:55
    2021-08-15 17:55
    2021-08-15 21:00
    
    for max_dt = 60min, the result of the function would be:
    
    2021-08-15 17:55 + 0min
    2021-08-15 17:55 + 0.5*60min
    2021-08-15 17:55 + 60min
    2021-08-15 21:00
    
    2021-08-15 17:55
    2021-08-15 18:25
    2021-08-15 18:55
    2021-08-15 21:00
    """
    # Function to be passed to df.groupby('date').apply(__)
    def fun_1(x, max_dt):
        
        # Determining time interval to be added to the duplicate dates
        # min of:
        #        - defined maximum interval
        #        - difference between duplicate dates and the following date
        dt = min(max_dt, 0.8 * x.loc[x.index[-1], '__'])
        
        # Creating a vector of incremental time intervals to add to duplicate dates
        dt = np.linspace(0, 1, x.shape[0]) * dt
        
        # Adding Timedelta vector (dt)
        x['date'] = x['date'] + dt
        
        return x
    
    df = data[['date']].copy()
    
    # Creating column of time differences between each instance and the following
    df['__'] = pd.Timedelta(days = 1) - df['date'].diff(-1) - pd.Timedelta(days = 1)
    
    # Applying function
    df = df.groupby('date').apply(fun_1, max_dt)
    
    return df.drop(columns = ['__'])

def create_cat(data, cols, new_col, cases):
  """
  Create a single categorical column from several "dummy"
  columns (reverse get_dummies)

  Parameters:
  -----------

  cols: List of column names to apply function on.

  new_col: Name of new column

  cases: List of tuples where each tuple should have this format
          (name_of_category, list_of_corresponding_booleans)
          Ex:
              [('poop', [False, False, True, False]),
              ('before', [True, False, False, False]),
              ('p&p', [False, False, False, True]),
              ('pee', [False, True, False, False])]
  """

  df = data[cols].copy()

  # Function to pass to df.apply()
  def fun_1(x, cases):
    for i in cases:
      if i[1] == x.to_list():
        return i[0]
    else:
      return 'unknown'

    
  df[new_col] = df[cols].apply(fun_1, cases = cases, axis = 1)

  return df[new_col]


def get_time_features(data, cols = None):
  '''
  Create time/date related features
  
  Parameters:
  -----------

  data: Dataframe containing date column named 'date'

  cols: List of features to include, possible values:
        ['day', 'day_of_week', 'hour', 'minute', 'is_weekend', 'season']

        If cols is set to None all features are added
  '''
  df = data.copy()

  df['day'] = df['date'].dt.day
  df['day_of_week'] = df['date'].dt.dayofweek
  df['hour'] = df['date'].dt.hour
  df['minute'] = df['date'].dt.minute
  df['is_weekend'] = np.where((df['day_of_week'] == 5) | (df['day_of_week'] == 6), 1 ,0)

  # Creating season column
  df['season'] = df['date'].dt.month%12 //3 + 1
  df['season'] = df['season'].replace(dict(zip([1, 2, 3, 4], ['winter', 'spring', 'summer', 'autumn'])))

  _cols = ['day', 'day_of_week', 'hour', 'minute', 'is_weekend', 'season']

  if cols is not None:
    _cols = df.columns[df.columns.isin(cols)]

  df = df[_cols]

  return df


class Cleaner:
  '''
  Clean datasets received from EmpowerYu. This class can be used on all
  weight datasets except the two person 18 txt files, which have different
  formats.

  Parameters:
  -----------

  person_id: Id of person to extract his/her weight data

  paths: Paths to extract data from, if given person_id is ignored.
         (Use for person 18 tsv file)

  Attributes:
  -----------

  id: Person id

  paths: Paths used to extract data

  raw_df_list: List of dataframes as read from source without manipulation.

  Methods:
  --------

  clean(): Clean weight datasets of this person and concatenate
           them into one dataframe.

  get_behavioral_features(): Get the mean of status durations or meal scores
                             for a certain interval of hours preceding every
                             weight measurement.
  
  '''
  def __init__(self, person_id = None, paths = None):

    if person_id is None and paths is None:
      print('Enter person_id or file paths.')
      return None

    self.id = person_id

    if paths is not None:
      self.paths = paths
    else:
      self.paths = get_person_paths(person_id)

    self.raw_df_list = []

    for path in self.paths:
      self.raw_df_list.append(pd.read_csv(path, sep = '\t'))

  def clean(self,
            time_method = 'drop',
            group_event_cols = True,
            interpolate_duplicate_times = False,
            time_features = None,
            drop_wake_up = True,
            drop_comment = True,
            drop_notes = True):

    '''
    This method cleans the given person id's weight datasets and then combines
    them into one dataframe.

    Summary of the cleaning process:
      1. Missing date values are filled with their preceding date,
         as in all cases until now they have the same date as the
         measurements before and after them.
      
      2. Missing time values:
          a. time_method = 'drop':
            Missing times are entirely dropped
          b. time_method = 'ffill'
            Missing times are filled by the previous values for the cases
            where the previous values are from the same date and dropped otherwise.
      
      3. Out of sequence time values are assumed to be entered without sequence
         and are reordered. (All cases until now are within one day)

      4. If there is a sequence of values starting with a measurement
         that has a date difference greater than 350 days from the previous
         value and ending with a measurement that has a date difference greater
         than -350 (or vice versa), this segment is assumed to have a wrong year
         entry and is fixed accordingly. (See remove_wrong_years function for 
         more details on the fixing procedure). This is seen only in one of
         person 21's files, having two wrong year entries.
      
      5. Time intervals not exceeding 60 min, and not exceeding the difference
         between consecutive dates to maintain sequence, are added to coincident
         dates if interpolate_duplicate_times is set to True.
         (See clear_duplicate_dates function for more details on the
         interpolation procedure)

      6. Missing weights are dropped.

      7. Event columns (toilet, meals, and clothes) are each grouped into one
         column if group_event_cols is set to True. Cases where all columns are
         False or cases with multiple True values are marked as unknown.
         Ex: After meal and Before meal are True --> 'unknown'

      8. Comment, WakeUp Time, and Notes columns are dropped according to the
         input booleans.

    Parameters:
    -----------

    time_method: 'drop' to drop missing times
                 'ffill' if the previous value has the same date
                         --> fill with previous value
                         else drop

    group_event_cols: If True, toilet, meal, and clothes event columns are
                      grouped each in a single column.

    interpolate_duplicate_times: If True, timedelta will be added to measurements
                                 with coincident times.
                                 (Maximum added value = 60min)

    time_features: List of time features to include. Possible features:
                   ['day', 'day_of_week', 'hour', 'minute', 'is_weekend', 'season']
                    If False no features are added
                    If None all features are added (Default)
                                 
    '''

    clean_df_list = []

    for i, data in enumerate(self.raw_df_list):
      
      # Printing name of file being cleaned
      if re.search(r'epyu weights person .*', self.paths[i]):
        file_name = re.search(r'epyu weights person .*', self.paths[i]).group(0)
      else:
        file_name = self.paths[i]

      txt = 'Cleaning file: ' + file_name
      print(txt)
      print('-'*len(txt))

      ############################################
      ############################################
      # Dealing with dates and times
      ############################################
      ############################################

      txt = ' Cleaning date and time:'
      print(txt)
      print(' ' + '-'*len(txt))

      df = data.copy()

      # Converting non date/time entries to NaN
      df[['date', 'time']] = check_dt_format(df)

      # Printing number of missing dates and times
      print('   No. of missing dates: {}'.format(df['date'].isna().sum()))
      print('   No. of missing times: {}'.format(df['time'].isna().sum()))

      # Fill missing dates with the previous value
      print('   Filled {} missing dates.'.format(df['date'].isna().sum()))
      df['date'] = df['date'].fillna(method = 'ffill')


      # Dealing with missing times
      # Case 1: time_method == 'drop'
      if time_method == 'drop':
        print('   Dropped {} missing times.'.format(df['time'].isna().sum()))

        df = df.dropna(subset = ['time']).reset_index(drop = True)
        

      # Case 2: time_method == 'ffill'
      elif time_method == 'ffill':

        # Dropping missinpd.to_datetime(g times with) dates different from previous dates
        df['_diff'] = pd.to_datetime(df['date']).diff()
        cond = (df['_diff'] > pd.Timedelta(days = 0)) & (df['time'].isna())

        print('   Dropped {} missing times.'.format(cond.sum()))

        idx_to_drop = df[cond].index
        df = df.drop(index = idx_to_drop).reset_index(drop = True)
        df = df.drop(columns = ['_diff'])

        # Filling in other missing times
        print('   Filled {} missing times.'.format(df['time'].isna().sum()))
        df['time'] = df['time'].fillna(method = 'ffill')

      # Creating date pd.to_datetime(column of ty)pe: datetime64[ns]
      df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
      df = df.drop(columns = ['time'])

      # Fixing wrong year entries
      df['date'] = rm_wrong_years(df)

      # Resorting dataframe
      # (There are some out of sequence time entries)
      print('   Resorted dataframe by date.')
      df = df.sort_values(by = 'date').reset_index(drop = True)

      # Interpolate duplicate dates
      if interpolate_duplicate_times is True:
        df['date'] = clear_duplicate_dates(df, max_dt = pd.Timedelta(minutes = 60))
        print('   Interpolated duplicate times.')

      ############################################
      ############################################
      # Dealing with weights
      ############################################
      ############################################

      print('\n')
      txt = ' Cleaning weights:'
      print(txt)
      print(' ' + '-'*len(txt))

      if df['weight'].dtype == 'object':
        df['weight'] = pd.to_numeric(df['weight'], errors = 'coerce')
      
      # Dropping missing weights
      print('   Dropped {} missing weights.\n'.format(df['weight'].isna().sum()))
      df = df.dropna(subset = ['weight'])

      ############################################
      ############################################
      # Dealing with event columns
      ############################################
      ############################################

      

      if group_event_cols is True:
        print('\n')
        txt = ' Cleaning event columns:'
        print(txt)
        print(' ' + '-'*len(txt))

        # Toilet
        cases = [('poop', [False, False, True, False]),
                ('before', [True, False, False, False]),
                ('p&p', [False, False, False, True]),
                ('pee', [False, True, False, False])]

        cols = ['Before Toilet', 'After Pee', 'After Poop', 'After P&P']
        new_col = 'toilet'
        df[new_col] = create_cat(df, cols, new_col, cases)
        df = df.drop(columns = cols)

        # Meals
        cases = [('before', [True, False]),
                 ('after', [False, True])]
        cols = ['Before Meal', 'After Meal']
        new_col = 'meal'
        df[new_col] = create_cat(df, cols, new_col, cases)
        df = df.drop(columns = cols)

        # Clothes
        cases = [('night', [True, False, False]),
                 ('day', [False, True, False]),
                 ('no clothes', [False, False, True])]
        cols = ['Night Clothes', 'Day Clothes', 'No Clothes']
        new_col = 'clothes'
        df[new_col] = create_cat(df, cols, new_col, cases)
        df = df.drop(columns = cols)

        print('   Created grouped event columns.')
      
      # Drop Wake Up, Comment, and Notes columns
      cols_to_drop = np.array(['WakeUp Time', 'Comment', 'Notes'])
      cols_to_drop = cols_to_drop[[drop_wake_up, drop_comment, drop_notes]]

      df = df.drop(columns = cols_to_drop)
      
      txts = ['   Dropped ' + x + ' column.' for x in cols_to_drop]
      for txt in txts:
        print(txt)

      # Adding time features
      if time_features is not False:
        df = pd.concat([df, get_time_features(df, time_features)], axis = 1)

      clean_df_list.append(df)
      print('\n\n')

    

    
    clean_df = pd.concat(clean_df_list)
    clean_df = clean_df.sort_values(by = 'date').reset_index(drop = True)

    # Renaming id column
    clean_df = clean_df.rename(columns = {clean_df.columns[0]: 'id'})

    self.df = clean_df
    return clean_df

  ########################################
  # THIS PART IS NOT AVAILABLE IN data_cleaner.pynb
  ########################################

  def get_behavioral_features(self,
                              hours,
                              which = 'both',
                              cols = None,
                              suffix = '_mean',
                              weight_only = True):
    
    '''

    Get the mean of status durations or meal scores for a certain interval
    of hours preceding every weight measurement.

    Parameters:
    -----------

    hours: A list or range of hours to compute means. Must be a list of
           integers greater than zero.
           Ex: Passing [1, 2], the function will compute the mean score/status
           duration of the previous 1 hour and 2 hours.

    which: 'both', 'status', or 'meal'.
           Determines whethere to add status durations from home_status_event
           ('status'), kitchen score from meal_period ('meal'), or both ('both').

    cols: List of the home statuses to apply the function on.
          This could be one or more of the following:
          ['active', 'still', 'sleep', 'away']
          Function will use all possible columns if None is passed.


    suffix: Suffix of the new column containing means.
            (Added to the column original name)

    weight_only: Boolean. Default is True
                 If True, only weight and date columns are taken from the
                 weight data.
    '''

    # Defining bhv_list which contains the required behavioral data
    # 'meal', 'status', or both.
    if which == 'both':
      bhv_list = ['status', 'meal']

    elif which not in ['status', 'meal']:
      # Replace this with raise error
      print('Enter a valid value for which, possible values: "both", "meal", "status"')
      return None

    else:
      bhv_list = [which]

    # Checking if weight data has been cleaned
    # If not, weight data is cleaned using the default settings
    if hasattr(self, 'df'):
      df = self.df.copy()
    else:
      df = self.clean()
    
    # Dropping all columns except weight and date
    if weight_only is True:
      df = df[['id', 'date', 'weight']]

    # Add this to the original cleaner.clean() method
    df = df.rename(columns = {'omdena_person_id': 'id'})

    # Getting behavioral features using BehavioralDataCleaner
    for bhv in bhv_list:
      
      cl = BehavioralDataCleaner(person_id = self.id, name = bhv)

      # Making sure person has behavioral data
      if cl.clean() is None:
        print('Person {} has no behavioral data.'.format(self.id))
        return None
      
      # Creating behavioral features
      features = cl.get_cum_mean(hours = hours, suffix = suffix, cols = cols)

      # Merge features with weight data
      df = pd.merge_asof(df, features.drop(columns = ['id']), on = 'date')
    
    return df

  def get_sample(self, t1, t2, ratio):
    """
    Resample weight measurements taken between hours [t1, t2] (inclusive).
    The function returns one or two samples from every date in the time range given if possible,
    and a random weight measurement if no measurements are available in the given time range.
    The percentage of days having two samples is determined by the parameter "ratio"
    
    Parameters:
    -----------
    
    t1: Start hour of time range
    
    t2: End hour of time range
    
    ratio: Ratio of days with 2 measurement samples
    """
    def group_by_fun(group, t1, t2, ratio):
        """
        This function is passed to the "apply" method of a DataFrameGroupBy object.

        Parameters:
        -----------

        group: Parameter to be passed to the apply method
        """

        cond = group['date'].dt.hour.isin(np.arange(t1, t2 + 1))
        num_of_samples = 2 if np.random.rand() < ratio else 1
        if group[cond].shape[0] == 0:
            if group.shape[0] < 2:
                return group.sample(n = 1)
            else:
                return group.sample(n = num_of_samples, replace = False)
        elif group[cond].shape[0] < 2:
            return group[cond].sample(n = 1)
        else:
            return group[cond].sample(n = num_of_samples)

    # Checking if weight data has been cleaned
    # If not, weight data is cleaned using the default settings
    if hasattr(self, 'df'):
      df = self.df.copy()
    else:
      df = self.clean()
    
    return df.groupby(df['date'].dt.date, group_keys = False).apply(group_by_fun, t1 = t1, t2 = t2, ratio = ratio)


## Plot Status duration bars

def slice_dates(data, start_date, end_date, meal):
  df = data.copy()

  if meal is True:
    df = df.set_index('date').asfreq('30T').fillna(0)
  else:
    df = df.set_index('date')
  

  if start_date is not None:
    df = df.loc[start_date:]

  if end_date is not None:
    df = df.loc[:end_date]
  else:
    df = df.iloc[:-1]

  df = df.reset_index()

  return df

def plot_status_durations(data, meal = None, start_date = None, end_date = None, figsize = (30, 10)):
  '''
  Plot home status durations as horizontal bars.

  Parameters:
  -----------
  data: home_status_event dataframe cleaned with BehavioralDataCleaner

  meal: meal_period dataframe cleaned with BehavioralDataCleaner
        If set to None kitchen score is not plotted

  start_date: Start of date range to be plotted

  end_date: End of date range to be plotted
  '''
  df = data.copy()

  # Computing status durations
  df['duration'] = pd.Timedelta(days = 1) - df['date'].diff(-1) - pd.Timedelta(days = 1)

  # Slicing dates
  df = slice_dates(df, start_date, end_date, meal = False)

  if meal is not None:
    df1 = meal.copy()
    df1 = slice_dates(df1, start_date, end_date, meal = True)

    # Scaling kitchen scores
    df1['score'] = df1['score'] * 4 / df1['score'].max()

  # Creating y column for plt.barh(y = __)
  trans_dict = {'sleep': 1, 'still': 2, 'active': 3, 'away': 4}
  df['y'] = df['status'].replace(trans_dict)

  # Plotting
  fig, ax = plt.subplots(figsize = figsize)

  # Status durations
  ax.barh(y = df['y'], height = 0.2, width = df['duration'], left = df['date'],
          color = 'deepskyblue')
  #alt.Chart(df).mark_tick().encode(x='duration',y='y')
  
  # Kitchen score
  if meal is not None:
    ax.plot('date', 'score', data = df1, color = 'orange', alpha = 0.7, label = 'Kitchen score')

  # Setting xticks
  #start = '2021-06-08'
  #end = '2021-08-12'
  start = df.loc[0, 'date'].date() # pd.to_datetime('2020-12-01')# 
  end = df.loc[df.shape[0] - 1, 'date'].date() + pd.Timedelta(days = 1) #pd.to_datetime('2021-11-10') #

  xticks = pd.Series(pd.date_range(start, end, freq = '12H'))

  ax.set_xticks(xticks)
  ax.set_xticklabels(xticks.dt.strftime('%H:%M'), rotation = 45)

  ax.set_yticks(np.arange(1, 5))
  ax.set_yticklabels(trans_dict.keys(), fontsize = 20)

  title = 'Home status durations and kitchen score'
  ax.set_title(title, fontsize = 20)
  ax.set_xlabel('Time of day')

  ax.grid(axis = 'x')
  ax.legend()
  st.pyplot(fig)




st.write("""
# EmPowerYu Web App
This app reports the findings from the Omdena-EmPowerYu Project!
***
""")

st.header('Exploratory Data Analysis')

person_id = 21
cl = Cleaner(person_id)
df = cl.clean()

st.write('Clean data output of patient ', person_id,'; Total observations recorded: ',len(df))
## Get a cleaned dataset
st.write(df)

st.write("""***""")


## Get behavioral features
df_bf = cl.get_behavioral_features(hours = range(1, 3), which = 'both', cols = ['sleep'])
st.write("Get Behavioral data for patient",'; Total observations recorded: ',len(df_bf))
st.write(df_bf)


bcl = BehavioralDataCleaner(person_id, 'status')
df1 = bcl.clean()

bcl = BehavioralDataCleaner(person_id, 'meal')
df2 = bcl.clean()


start = '2021-07-08'
end = '2021-08-12'

st.write("""***""")

daily_avg = df.groupby(df['date'].dt.date).mean()

monthly_avg = df.set_index('date').resample(rule = '1m', label = 'left').mean()

fig = plt.figure(figsize = (20, 8))
plt.plot(daily_avg.index, daily_avg['weight'], label = 'Daily Avg')
plt.plot(monthly_avg.index, monthly_avg['weight'], linewidth = 4, label = 'Monthly Avg')

plt.title('Daily Weight Averages')
plt.xticks(monthly_avg.index + DateOffset(days = 1), rotation = 45)
plt.grid(axis = 'both')
plt.xlabel('Date')
plt.ylabel('Weight (lb)')
plt.legend()
st.pyplot(fig)

st.write('***')


## Removing Outliers
daily_median = df.groupby(df['date'].dt.date).median()
daily_avg = df.groupby(df['date'].dt.date).mean()


# If label = 'left', each point in the monthly avg line refers to the average of the following month
# If label = 'right', each point in the monthly avg line refers to the average of the previous month
# See df.resample() documentation for more details

weekly_median = df.set_index('date').resample(rule = '1w', label = 'left').mean()
weekly_avg = df.set_index('date').resample(rule = '1w', label = 'left').median()

monthly_median = df.set_index('date').resample(rule = '1m', label = 'left').mean()
monthly_avg = df.set_index('date').resample(rule = '1m', label = 'left').median()

fig, axs = plt.subplots(3, 1, figsize = (20, 24), tight_layout = True)

graph_dict = {'Daily': [daily_median, daily_avg],
             'Weekly': [weekly_median, weekly_avg],
             'Monthly': [monthly_median, monthly_avg]}

for i, (key, val) in enumerate(graph_dict.items()):

    ax = axs[i]
    
    ax.plot(val[1].index, val[1]['weight'], label = key + ' Avg', color = 'red', alpha = 0.8)
    ax.plot(val[0].index, val[0]['weight'], label = key + ' Median', color = 'steelblue')
    

    ax.set_title(key + ' Weight Averages')

    ax.set_xticks(monthly_avg.index + DateOffset(days = 1))
    ax.set_xticklabels((monthly_avg.index + DateOffset(days = 1)).date, rotation = 45)
    ax.grid(axis = 'both')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight (lb)')
    ax.legend()
st.pyplot(fig)

st.write('***')




st.write('***')


plot_status_durations(df1, df2, start_date = start, end_date = end)
#plot_status_durations(df1, df2, start_date = '2021-06-08', end_date = '2021-08-12')
#st.pyplot(fig)
#plt.show()

#st.write(df.loc[0, 'date'].date()) # pd.to_datetime('2020-12-01')# 
#st.write(df.loc[df.shape[0] - 1, 'date'].date() + pd.Timedelta(days = 1)) #pd.to_datetime('2021-11-10') #
