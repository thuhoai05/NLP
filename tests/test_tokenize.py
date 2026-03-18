import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from underthesea import word_tokenize

text = "Hà Nội là thủ đô của Việt Nam"
print(word_tokenize(text))