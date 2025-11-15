import os
import pprint

print("--- Python Executable ---")
print(os.sys.executable)
print("\n--- Environment PATH ---")

# 打印 PATH 环境变量，为了方便阅读，将每个路径拆分到新的一行
path_vars = os.environ.get('PATH', '').split(';')
pprint.pprint(path_vars)

print("\n--- All Environment Variables (for reference) ---")
pprint.pprint(dict(os.environ))