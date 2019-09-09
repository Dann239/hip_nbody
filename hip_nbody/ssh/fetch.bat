@echo off
pscp -P 2232 -i key_private.ppk -l pavlov pavlov@rasa202.asuscomm.com:work/data/* ../data/
pause -> nul