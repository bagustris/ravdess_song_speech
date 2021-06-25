#!/bin/python2
# python2 script
# Extract acoustic LLDs (MFCC and eGeMAPS sets from openSMILE)
# Output: csv files

import os
import time
import glob
import ntpath

# MODIFY HERE
feature_type  = 'egemaps'       # 'mfcc' or 'egemaps'        
folder_data   = '/media/bagustris/bagus/dataset/Audio_Speech_Actors_01-24/'  # folder with audio (.wav) files
subdirs       = os.listdir(folder_data)
exe_opensmile = '/home/bagustris/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'  # MODIFY this path to the folder of the SMILExtract (version 2.3) executable
path_config   = '/home/bagustris/opensmile-2.3.0/config/'                                      # MODIFY this path to the config folder of opensmile 2.3 - no POSIX here on cygwin (windows)

if feature_type=='mfcc':
    folder_output = './audio_features_mfcc/'  # output folder
    conf_smileconf = path_config + 'MFCC12_0_D_A.conf'  # MFCCs 0-12 with delta and acceleration coefficients
    opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsv 0 -timestampcsv 0 -headercsv 1'  # options from standard_data_output_lldonly.conf.inc
    outputoption = '-csvoutput'  # options from standard_data_output_lldonly.conf.inc
elif feature_type=='egemaps':
    folder_output = './audio_features_egemaps/'  # output folder
    conf_smileconf = path_config + 'gemaps/eGeMAPSv01a.conf'  # eGeMAPS feature set
    opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1'  # options from standard_data_output.conf.inc
    outputoption = '-lldcsvoutput'  # options from standard_data_output.conf.inc
else:
    print('Error: Feature type ' + feature_type + ' unknown!')


if not os.path.exists(folder_output):
    os.mkdir(folder_output)

for subdir in subdirs:
    for fn in glob.glob(os.path.join(folder_data, subdir, '*.wav')):
#for fn in os.listdir(folder_data):
        filename = ntpath.basename(fn)
        instname    = os.path.splitext(filename)[0]
        outfilename = folder_output + instname + '.csv'
        opensmile_call = exe_opensmile + ' ' + opensmile_options + ' -inputfile ' + fn + ' ' + outputoption + ' ' + outfilename + ' -instname ' + instname + ' -output ?'  # (disabling htk output)
        os.system(opensmile_call)
        time.sleep(0.01)

os.remove('smile.log')
