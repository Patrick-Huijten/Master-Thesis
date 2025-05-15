import time
import fitz
import pandas as pd
import os
import string
import spacy
import torch
import logging
import re
import joblib
import numpy as np
from sklearn.svm import SVC
from spacy.util import is_package
from spacy.cli import download
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def classify(df, )