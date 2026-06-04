# from recbole.model.sequential_recommender.bert4rec import BERT4Rec
# from recbole.model.sequential_recommender.gru4rec import GRU4Rec
# from recbole.model.sequential_recommender.sasrec import SASRec
import importlib
import os
import sys

# 현재 디렉터리 기준으로 모든 .py 파일 자동 import
package_dir = os.path.dirname(__file__)

for file in os.listdir(package_dir):
    if file.endswith(".py") and file not in ["__init__.py", "__pycache__"]:
        module_name = file[:-3]
        module_path = f"{__name__}.{module_name}"
        try:
            importlib.import_module(module_path)
        except Exception as e:
            print(f"[WARN] Failed to import {module_name}: {e}", file=sys.stderr)
