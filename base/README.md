# DigIT-Base-UI
Base User Interface for the DigIT Projects
- [Root-Detector](https://github.com/alexander-g/Root-Detector)
- Wood Anatomy
- Pollen Detector
- Bat Detector

***

### Layout

```
RootDetector
+-- base/                     #this repository
|   +-- main.py               #minimal, for standalone testing only
|   +-- backend/              #python server code
|   |   +-- app.py            #base flask app
|   +-- frontend/             
|   |   +-- base/             #base javascript UI code
|   +-- templates/            #jinja HTML templates

+-- backend/                  #downstream backend overrides
|   +-- app.py                
+-- frontend/
|   +-- roots/                #downstream javascript UI overrides
+-- models/                   #pretrained models folder
+-- templates/
+-- static/                   #served HTML/JS/CSS, recompiled dynamically
+-- cache/                    #temporary, stores images/results for processing
```
