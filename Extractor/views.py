from django.shortcuts import render

from Extractor.models import *
from Extractor.serializers import *
from rest_framework import viewsets

class QueryViewSet(viewsets.ModelViewSet):
    queryset = QueryModel.objects.all()
    serializer_class = QuerySerializer

    def get_queryset(self):
        queryset = self.queryset
        queryset = queryset.order_by('-id')

        token = self.request.query_params.get('id', None)
        if token is not None:
            queryset = queryset.filter(token=token)

        return queryset