#!/bin/bash

{
    echo "audio_path,text_path,duration"
    cat radio_2.csv radio_pspeech_sample_manifest.csv 
} > combined_dataset.csv

echo "$(wc -l < combined_dataset.csv) строк"
