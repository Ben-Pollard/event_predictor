# event_predictor
Tech meetups are events focussing on technology (usually Information Technology / software engineering / computer science), with a typical conference style setup, where there are presentations and the members that attend are audience. 

event_predictor is a model that can predict for a new, to be organised, meetup event how many members will RSVP to that meetup. This is useful for organisers to investigate whether minor changes to their event (e.g. event date / day of week) will have a positive influence on the number of expected attendees.

## Requirements
[Docker](https://docs.docker.com/engine/installation/).

## Installation
Clone this repo.
Build the docker image: `docker build `

## How to start up the score server
`docker run `

## How to use
POST a request to http://0.0.0.0:80/predict
The API expects one JSON record per query, e.g.:
`curl -H "Content-Type: application/json" -X POST -d '{"rsvp_limit":10}' http://127.0.0.1:80/predict`
The required field names for each query are:

## How to change the dataset