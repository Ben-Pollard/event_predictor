# event_predictor
Tech meetups are events focussing on technology (usually Information Technology / software engineering / computer science), with a typical conference style setup, where there are presentations and the members that attend are audience. 

event_predictor is a model that can predict for a new, to be organised, meetup event how many members will RSVP to that meetup. This is useful for organisers to investigate whether minor changes to their event (e.g. event date / day of week) will have a positive influence on the number of expected attendees.

## Requirements
[Docker](https://docs.docker.com/engine/installation/).

## Installation
Clone this repo.
Build the docker image: `docker build -t event_predictor:latest .`
This will take a while when it first runs as it downloads an image with miniconda installed.

## How to start up the prediction server
`docker run -p 80:80 event_predictor python Scorer/Predict.py`

If you run the container more than once you might find that you need to change the port mapping. To get around this you can specify a new port on the host e.g. -p 81:80. You would then need to send prediction requests to this new port.

## How to use
POST a request to http://0.0.0.0:80/predict
The API expects one JSON record per query, e.g.:
`curl -H "Content-Type: application/json" -X POST -d '{"rsvp_limit":10}' http://0.0.0.0:80/predict`

The required field names for each query are:
* rsvp_limit: The maximum number of YES RSVPs that this event will allow

## How to change the dataset
