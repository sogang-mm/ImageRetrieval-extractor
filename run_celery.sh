#!/usr/bin/env bash
service rabbitmq-server restart
celery -A RetrievalExtractor worker -B -l INFO
