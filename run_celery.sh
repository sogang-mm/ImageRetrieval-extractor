#!/usr/bin/env bash
celery -A RetrievalExtractor worker -B -l INFO
