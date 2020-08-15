
import sys
import json

file_json = "templates/actData.json"
fp_in = open (file_json,encoding='utf-8')
json_str = fp_in.read()
fp_in.close()
dict_aa = json.loads(json_str)

print(dict_aa)