import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import talib
import tempfile
import time
from pathlib import Path
import uuid
import logging

import openai
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

import requests
from bs4 import BeautifulSoup
from xgboost import XGBClassifier

import subprocess
import json
