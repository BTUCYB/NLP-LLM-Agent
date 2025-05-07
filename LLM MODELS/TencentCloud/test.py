import sys
import os
from distutils.sysconfig import get_python_lib

print(f"Python Path: {sys.executable}")
print(f"Site Packages: {get_python_lib()}")
print(f"Tencent Path: {os.path.join(get_python_lib(), 'tencentcloud', 'nlp', 'v20190423')}")

try:
    from tencentcloud.nlp.v20190423 import models
    print("Import SUCCESS")
except ImportError as e:
    print(f"Import FAILED: {str(e)}")