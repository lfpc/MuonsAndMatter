combi = [ 70.  , 170.  , 208.  , 207.  , 281.  , 172.82, 212.54, 168.64,
    40.  ,  40.  , 150.  , 150.  ,   2.  ,   2.  ,  80.  ,  80.  ,
    150.  , 150.  ,   2.  ,   2.  ,  72.  ,  51.  ,  29.  ,  46.  ,
    10.  ,   7.  ,  54.  ,  38.  ,  46.  , 192.  ,  14.  ,   9.  ,
    10.  ,  31.  ,  35.  ,  31.  ,  51.  ,  11.  ,   3.  ,  32.  ,
    54.  ,  24.  ,   8.  ,   8.  ,  22.  ,  32.  , 209.  ,  35.  ,
    8.  ,  13.  ,  33.  ,  77.  ,  85.  , 241.  ,   9.  ,  26.  ]

sc_v6 = [ 40.0000, 231.0000,   0.0000, 353.0780, 125.0830, 184.8340, 150.1930,186.8120,  
         40.0000,  40.0000, 150.0000, 150.0000,   1.0000,   1.0000, 1.0000,  
         50.0000,  50.0000, 130.0000, 130.0000,   2.0000,   2.0000,1.0000, 
        72.0000,  51.0000,  29.0000,  46.0000,  10.0000,   7.0000,1.0000, 
        45.6888,  45.6888,  22.1839,  22.1839,  27.0063,  16.2448,3.0000,  
        10.0000,  31.0000,  35.0000,  31.0000,  51.0000,  11.0000,1.0000,  
        24.7961,  48.7639,   8.0000, 104.7320,  15.7991,  16.7793,1.0000,   
        3.0000, 100.0000, 192.0000, 192.0000,   2.0000,   4.8004,1.0000,   
        3.0000, 100.0000,   8.0000, 172.7290,  46.8285,   2.0000,1.0000]

baseline = [205.,205.,280.,245.,305.,240.,87.,65.,
    35.,121,11.,2.,65.,43.,121.,207.,11.,2.,6.,33.,32.,13.,70.,11.,5.,16.,112.,5.,4.,2.,15.,34.,235.,32.,5.,
    8.,31.,90.,186.,310.,2.,55.]

optimal_oliver =  [208.0, 207.0, 281.0, 248.0, 305.0, 242.0, 72.0, 51.0, 29.0, 46.0, 10.0, 7.0, 54.0,
                         38.0, 46.0, 192.0, 14.0, 9.0, 10.0, 31.0, 35.0, 31.0, 51.0, 11.0, 3.0, 32.0, 54.0, 
                         24.0, 8.0, 8.0, 22.0, 32.0, 209.0, 35.0, 8.0, 13.0, 33.0, 77.0, 85.0, 241.0, 9.0, 26.0]

param_test = [0,353.078,125.083,184.834,150.193,186.812,72,51,29,46,10,7,45.6888,
         45.6888,22.1839,22.1839,27.0063,16.2448,10,31,35,31,51,11,24.7961,48.7639,8,104.732,15.7991,16.7793,3,100,192,192,2,
         4.8004,3,100,8,172.729,46.8285,2]

magnets_params = {0: [],
                  1: [0,6,7,8,9,10,11],
                  2: [1, 12,13,14,15,16,17],
                  3: [2,18,19,20,21,22,23],
                  4: [3, 24,25,26,27,28,29],
                  5: [4, 30,31,32,33,34,35],
                  6: [5, 36,37,38,39,40,41]}