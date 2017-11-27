# event_predictor
Tech meetups are events focussing on technology (usually Information Technology / software engineering / computer science), with a typical conference style setup, where there are presentations and the members that attend are audience. 

event_predictor is a model that can predict for a new, to be organised, meetup event how many members will RSVP to that meetup. This is useful for organisers to investigate whether minor changes to their event (e.g. event date / day of week) will have a positive influence on the number of expected attendees.

## Requirements
[Docker](https://docs.docker.com/engine/installation/).

## Installation
Clone this repo.
Build the docker image: `docker build -t event_predictor:latest .`
This will take a while when it first runs as it downloads an image with miniconda installed.

## Start the prediction service
`docker run -p 80:80 event_predictor python Scorer/Predict.py`

If you run the container more than once you might find that you need to change the port mapping. To get around this you can specify a new port on the host e.g. -p 81:80. You would then need to send prediction requests to this new port.

## Scoring a new meetup
POST a request to http://0.0.0.0:80/predict
The API expects one JSON record per query, e.g.:

`curl -H "Content-Type: application/json" -X POST -d '{"created": 1309088803000, "duration": null, "group_id": "INTERNATIONALS-in-Rotterdam", "rsvp_limit": null,  "time": 1310056200000, "topics": ["WCF", "Project Planning", "Leadership", "Software Engineering",   "Enterprise Architecture", "Software Development", ".NET", "IDesign Method", "Windows Azure Platform", "Software Architecture"]}' http://127.0.0.1:80/predict`

The required fields for each query are: `['rsvp_limit','duration','time','created','group_id','topics']`. 

*rsvp_limit* and *duration* are nullable but the nulls must be passed explicitly. If there are no topics, this must be passed as an empty list.

## Training on new data
To train a new model and then start the prediction service do:

`docker run -p 80:80 event_predictor /bin/bash -c "python Trainer/Train.py; python Scorer/Predict.py"`

The training process needs *events.json*, *users.json* and *groups.json* to be in the *Data* folder.
