import urllib.request, os

URLS = {
    'https://github.com/mkmitchell/DuckNet/releases/download/asset/basemodel.pt.zip'        : 'models/detection/basemodel.pt.zip',
}

for url, destination in URLS.items():
    print(f'Downloading {url} ...')
    with urllib.request.urlopen(url) as f:
        os.makedirs( os.path.dirname(destination), exist_ok=True )
        open(destination, 'wb').write(f.read())

