from backend.app import App
from backend.cli import CLI

if __name__ == '__main__':
    ok = CLI.run()

    if not ok:
        #start UI
        App().run()



# directory layout
#
#-RootUI/
#   -main.py
#   -build.py
#   -backend/
#       -__init__.py
#       -root_detection.py
#       -root_tracking.py
#   -frontend/
#       -html/                 #jinja templates folder
#           -index.hmtl
#       -js/
#           -root_detection.js
#           -root_tracking.js
#       -css/
#           -??
#   -UI/                       #flask static folder, recompiled dynamically
#       -index.html
#       -...
#   -tests/
#       run_tests.sh
#       -testcases/
#           -test_basic.py
#           ...
#       -assets/
#           -img0.tiff
#       -logs/
#           -codecoverage/
#       -docker/
#           Dockerfile
#           -scripts/
#               -entrypoint.sh
#   -base/                     #submodule
#       -main.py               #minimal, for standalone testing only
#       -backend/
#           app.py             #main flask app
#       -frontend/
#           -thirdparty/
#              -jquery.min.js
#              -fomantic/
#           ...
#       -templates/
#       -UI/                   #flask static folder
#