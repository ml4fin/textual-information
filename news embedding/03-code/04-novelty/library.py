import sys
import platform
import numpy
import pandas

print("Python version :", sys.version)
print("Executable     :", sys.executable)
print("Architecture   :", platform.machine())
print("NumPy          :", numpy.__version__)
print("Pandas         :", pandas.__version__)

try:
    import sklearn
    print("scikit-learn   :", sklearn.__version__)
except ImportError:
    print("scikit-learn   : NOT INSTALLED")

try:
    import psycopg2
    print("psycopg2       :", psycopg2.__version__)
except ImportError:
    print("psycopg2       : NOT INSTALLED")

try:
    import faiss
    print("FAISS          :", faiss.__version__)
except ImportError:
    print("FAISS          : NOT INSTALLED")