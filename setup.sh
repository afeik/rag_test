#!/bin/bash
python -m spacy download en_core_web_sm
python -m spacy link en_core_web_sm en_core_web_sm --force