@echo off
pscp -P 2232 -i key_private.ppk -l pavlov ../kernel.cu pavlov@rasa202.asuscomm.com:work/