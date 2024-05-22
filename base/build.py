import os, shutil, sys, datetime, subprocess

NAME       = 'BASE'
build_name = f'%s_{NAME}'%(datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss') )
build_dir  = 'builds/%s'%build_name

rc = subprocess.call(f'''pyinstaller --noupx                                 \
                     --distpath {build_dir} main.py''', shell=True)
if rc:
    print('Build failed')
    sys.exit(rc)

from base.app import App
App().recompile_static(force=True)        #make sure the static/ folder is up to date
shutil.copytree('static', build_dir+'/static')

if 'linux' in sys.platform:
    #os.symlink('main/main', build_dir+'/main.run')  #has problems with libraries
    open(build_dir+'/main.sh','w').write('#!/bin/bash\ncd $(dirname ${BASH_SOURCE:-$0}) && main/main $@')
    os.chmod(build_dir+'/main.sh', int('755', base=8))  #chmod +x
else:
    open(build_dir+'/main.bat', 'w').write(r'main\main.exe'+'\npause')

shutil.rmtree('./build')
os.remove('./main.spec')

