'''
Author: Dennis van der Meer
Email: dennis.vandermeer@angstrom.uu.se

This script untars the downloaded files from EUMETSAT and places them in a
directory that is named after the time stamp of the satellite image. The script
also keeps a list of the downloaded files that have already been untarred to
avoid duplicate work.

The code is adapted from Open Climate Fix.
'''
import tarfile
import re
import os
from datetime import datetime

#PATH = "/Volumes/G-Drive_USB-C/Data/EUMETSAT/"
PATH = r"D:\EUMETSAT"

# The directory containing the tar files downloaded from EUMETSAT
SRC_PATH = os.path.join(PATH, 'downloaded_tar')
DST_PATH = os.path.join(PATH, 'untarred')

# This is a list of the tar files which have already been processed.
# This is useful so that this script can re-start for where it left off,
# if need be.
LIST_OF_COMPLETED_FILES = os.path.join(PATH, 'completed_files.txt')

# Load list of completed files
if os.path.exists(LIST_OF_COMPLETED_FILES):
    with open(LIST_OF_COMPLETED_FILES, 'r') as fh:
        completed_files = fh.readlines()
    completed_files = [fname.strip() for fname in completed_files]
else:
    completed_files = []

# Filenames in the source path
filenames = os.listdir(SRC_PATH)
filenames = [ fname for fname in filenames if fname.endswith('.tar')]
print(len(filenames))

# Remove files which have previously been completed
filenames = list(set(filenames) - set(completed_files))
filenames.sort()
n = len(filenames)
print(n)

import shutil

def get_datetime(inner_nat_name):
    date_str = re.split('-|[.]',inner_nat_name)[5] # Always the sixth element we're after
    return datetime.strptime(date_str, "%Y%m%d%H%M%S")

for i, filename in enumerate(filenames):
    print(i+1, 'of', n, '= {:.0%}'.format((i+1)/n), ' : ', filename)
    full_filename = os.path.join(SRC_PATH, filename)
    nat = tarfile.open(full_filename)
    nat.extractall(PATH) # Move the file to the right directory
    for inner_nat in nat:
        print('\r', inner_nat.name, end='', flush=True)
        dt = get_datetime(inner_nat.name)
        dir_name = os.path.join(DST_PATH, dt.strftime("%Y/%m/%d/%H/%M"))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        shutil.move(os.path.join(PATH,inner_nat.name),dir_name)  # PATH+inner_nat.name
    with open(LIST_OF_COMPLETED_FILES, 'a') as fh:
        fh.write('{}\n'.format(filename))
    print()
