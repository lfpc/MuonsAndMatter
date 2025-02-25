combi = [ 70.  , 170.  , 208.  , 207.  , 281.  , 172.82, 212.54, 168.64,
    40.  ,  40.  , 150.  , 150.  ,   2.  ,   2.  ,  80.  ,  80.  ,
    150.  , 150.  ,   2.  ,   2.  ,  72.  ,  51.  ,  29.  ,  46.  ,
    10.  ,   7.  ,  54.  ,  38.  ,  46.  , 192.  ,  14.  ,   9.  ,
    10.  ,  31.  ,  35.  ,  31.  ,  51.  ,  11.  ,   3.  ,  32.  ,
    54.  ,  24.  ,   8.  ,   8.  ,  22.  ,  32.  , 209.  ,  35.  ,
    8.  ,  13.  ,  33.  ,  77.  ,  85.  , 241.  ,   9.  ,  26.  ]

sc_v6 = [231.00,  0., 353.08, 125.08, 184.83, 150.19, 186.81, 
         50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00, 0.00,
        0.,  0.,  0.,  0.,  0.,   0., 1., 0.,
        45.69,  45.69,  22.18,  22.18,  27.01,  16.24, 3.00, 0.00,
        0.,  0.,  0.,  0.,  0.,  0., 1., 0., 
        24.80,  48.76,   8.00, 104.73,  15.80,  16.78, 1.00, 0.00,
        3.00, 100.00, 192.00, 192.00,   2.00,   4.80, 1.00, 0.00,
        3.00, 100.00,   8.00, 172.73,  46.83,   2.00, 1.00, 0.00]

baseline = [205.,205.,280.,245.,305.,240.,87.,65.,
    35.,121,11.,2.,65.,43.,121.,207.,11.,2.,6.,33.,32.,13.,70.,11.,5.,16.,112.,5.,4.,2.,15.,34.,235.,32.,5.,
    8.,31.,90.,186.,310.,2.,55.]


optimal_oliver = [231.00, 208.0, 207.0, 281.0, 248.0, 305.0, 242.0,
                  50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00, 0.00,
                  72.0, 51.0, 29.0, 46.0, 10.0, 7.0, 1.0, 0.,
                  54.0, 38.0, 46.0, 192.0, 14.0, 9.0, 1.0, 0.,
                  10.0, 31.0, 35.0, 31.0, 51.0, 11.0, 1.0, 0.,
                  3.0, 32.0, 54.0, 24.0, 8.0, 8.0, 1.0, 0.,
                  22.0, 32.0, 209.0, 35.0, 8.0, 13.0, 1.0, 0.,
                  33.0, 77.0, 85.0, 241.0, 9.0, 26.0, 1.0, 0.]
oliver_scaled = [231.00, 170.89, 170.07, 230.87, 203.76, 250.59, 198.83,
                  50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00, 0.00,
                  72.00, 51.00, 29.00, 46.00, 10.00, 7.00, 1.00, 0.00,
                  54.00, 38.00, 46.00, 130.00, 14.00, 9.00, 1.00, 0.00,
                  10.00, 31.00, 35.00, 31.00, 51.00, 11.00, 1.00, 0.00,
                  5.00, 32.00, 54.00, 24.00, 8.00, 8.00, 1.00, 0.00,
                  22.00, 32.00, 130.00, 35.00, 8.00, 13.00, 1.00, 0.00,
                  33.00, 77.00, 85.00, 90.00, 9.00, 26.00, 1.00, 0.00]

params_per_magnet = ['CoreWidth_1', 'CoreWidth_2', 
                     'CoreHeight_1', 'CoreHeight_2', 
                     'GapWidth_1', 'GapWidth_1',
                     'RatioYokeCore', 'CentralGap']

new_parametrization = {'HA': [0, 7, 8, 9, 10, 11, 12, 13, 14],
                       'M1': [1, 15, 16, 17, 18, 19, 20, 21, 22],
                       'M2': [2, 23, 24, 25, 26, 27, 28, 29, 30], #SC
                       'M3': [3, 31, 32, 33, 34, 35, 36, 37, 38],
                       'M4': [4, 39, 40, 41, 42, 43, 44, 45, 46],
                       'M5': [5, 47, 48, 49, 50, 51, 52, 53, 54],
                       'M6': [6, 55, 56, 57, 58, 59, 60, 61, 62]}

SC_idx =  [new_parametrization['HA'][0]] + new_parametrization['M2'][:-2] + [new_parametrization['M3'][0]]+\
           new_parametrization['M4'] + new_parametrization['M5'] + new_parametrization['M6']



bounds_min = [51, 51, 51, 51, 51, 51, 51, 51,
 1, 1, 1, 1, 2, 2, 0, 0,
 1, 1, 1, 1, 2, 2, 0, 0,
 1, 1, 1, 1, 2, 2, 0, 0,
 1, 1, 1, 15, 15, 2, 0, 0,
 1, 1, 1, 1, 2, 2, 0, 0,
 1, 1, 1, 1, 2, 2, 0, 0,
 1, 1, 1, 1, 2, 2, 0, 0]

bounds_max = [401, 401, 401, 401, 401, 401, 401, 401,
 100, 100, 200, 200, 70, 70, 4, 70,
 100, 100, 200, 200, 70, 70, 4, 70,
 100, 100, 200, 200, 70, 70, 4, 70,
 100, 100, 200, 70, 70, 70, 4, 70,
 100, 100, 200, 200, 70, 70, 4, 70,
 100, 100, 200, 200, 70, 70, 4, 70,
 100, 100, 200, 200, 70, 70, 4, 70]
