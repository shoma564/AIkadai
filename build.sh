#!/bin/bash
docker-compose down
docker-compose build --no-cache aiflask
docker-compose up
