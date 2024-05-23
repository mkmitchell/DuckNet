import json, os, glob, sys

LOG_DIR           = 'latest_logs'
INPUT_DIR         = f'{LOG_DIR}/coverage_js/raw'
OUTPUT_DIR        = f'{LOG_DIR}/coverage_js/reports'



jsonfiles         = glob.glob(os.path.join(INPUT_DIR, '*.json'))
if len(jsonfiles)==0:
    print('No code coverage files found in '+INPUT_DIR)
    sys.exit()

jsondata_per_test = {}
jsondata_per_url  = {}
for js in jsonfiles:
    jsondata = json.load(open(js))
    jsondata = dict([ ( '/'.join(x['url'].split('/')[-2:]), x) for x in jsondata if 'thirdparty' not in x['url']])
    jsondata_per_test[js] = jsondata
    #urls                  = urls.union(jsondata.keys())
    for u in jsondata.keys():
        jsondata_per_url[u] = jsondata_per_url.get(u,[]) + [jsondata[u]]
urls = list(jsondata_per_url.keys())

def merge_ranges(*list_of_ranges):
    max_n         = max([ranges[-1][-1] for ranges in list_of_ranges])
    #sanity check
    assert max_n < 1e6
    
    covered_chars = [0]*(max_n+1)
    for ranges in list_of_ranges:
        for start,end in ranges:
            covered_chars[start:end] = [1]*(end-start)
    merged_starts   = [0]+[i+1 for i,x in enumerate(covered_chars[:-1]) if x==0 and covered_chars[i+1]==1]
    merged_ends     =     [i+1 for i,x in enumerate(covered_chars[:-1]) if x==1 and covered_chars[i+1]==0]
    return list(zip(merged_starts, merged_ends))

texts_per_url = {}
merged_ranges = {}
for url, data_list in jsondata_per_url.items():
    list_of_ranges = [[(r['start'], r['end']) for r in d['ranges']] for d in data_list]
    merged_ranges[url] = merge_ranges(*list_of_ranges)
    texts_per_url[url] = data_list[0]['text']


prcnt_text = ''
os.makedirs(OUTPUT_DIR, exist_ok=True)
for url, ranges in merged_ranges.items():
    ranges_flat = [0] + [i for I in ranges for i in I]
    splits      = [texts_per_url[url][i:j] for i,j in zip(ranges_flat, ranges_flat[1:])]
    splits      = [f'<span style="background:pink;">{s}</span>' if i%2==0 else s for i,s in enumerate(splits) if len(s)]
    html        = ('<html><body style="font-family:monospace; white-space:pre-wrap;">'+''.join(splits)+'</body></html>')
    basename    = os.path.basename(url) or '___'
    open(os.path.expanduser(f'{OUTPUT_DIR}/{url.replace("/","_")}.html'),'w').write(html)

    prcnt       = sum([end-start for start,end in ranges]) / len(texts_per_url[url]) * 100
    prcnt_text += f'[{prcnt:5.1f}%] {url}\n'

print()
print('Javascript Code Coverage:')
print(prcnt_text)
print()
open(os.path.join(OUTPUT_DIR, 'percentages.txt'), 'w').write(prcnt_text)
