# YOLO Fracture Detection API

This project provides an API for fracture detection using the YOLO model.

## Dependencies

This program runs on Python 3.11

We make a virtual env using Pipenv for this repository.

First lets check if you have pipenv installed by running:
'''bash
pipenv --version
'''

If it is not installed. Install it using this guide for your specific computer setup. (https://pipenv.pypa.io/en/latest/installation.html)

Navigate to the project directory
'''bash
cd Applied-ML-Model
'''

Then all dependencies can be installed with:
'''bash
pipenv install
'''
This will install everything in the Pipfile and the versions specified in the Pipfile.lock

Activate the Pipenv by running:
'''bash
pipenv shell
'''


## Running the API
We can run the API without compiling, this is because we load a our trained model in the repository.
'''bash
uvicorn api.main:app
'''

The output should look something like this:
'''
INFO:     Started server process [57435]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:54360 - "GET / HTTP/1.1" 307 Temporary Redirect
INFO:     127.0.0.1:54360 - "GET /docs HTTP/1.1" 200 OK
INFO:     127.0.0.1:54360 - "GET /openapi.json HTTP/1.1" 200 OK
'''
Follow the link (by cmd + click). This should open a browser with the API.

We can kill the app by pressing ctrl + C.

## Training a new model
We can also train a new model, this will take some time!

For that we open the main.py in the root of the folder and adjust the parameters for training if needed.
To start training run:
'''bash
python main.py
'''
