#!/usr/bin/bash

# Due to limitations in GitHub's LFS implementation, namely that files larger
# than 2GB are not well-supported, I have instead used this hacky script to use
# Google drive as a large file repository.  Once you have pulled this datashare
# repo, you can use this script to download the raw data files if you would
# like.

# This script is admiteddly hacky.  But it gets the job done.  Based on this
# magical incantation found here: https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99 
# I believe we are abusing a Google docs API and Google Drive sharing to wget
# read-only data from Google Drive.  Inelegant, but effective.
FILEID='1XMV-M1wfjeyYBKZx39m0RH5xISxOUvz0' # gs1826_10x_grid_data.Nov5.pk
OUT_FILENAME='gs1826_10x_grid_data.Nov5.pk'
INNER_WGET_TARGET='https://docs.google.com/uc?export=download&id='
INNER_WGET_TARGET+=${FILEID}
echo $INNER_WGET_TARGET
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${INNER_WGET_TARGET} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${OUT_FILENAME} && rm -rf /tmp/cookies.txt

#The safer variable version above is broken, hard-coding for now:
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XMV-M1wfjeyYBKZx39m0RH5xISxOUvz0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${OUT_FILENAME} && rm -rf /tmp/cookies.txt
