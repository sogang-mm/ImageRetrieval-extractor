from rest_framework import serializers
from Extractor.models import *


class QuerySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = QueryModel
        fields = ('image','feature','extract_time','uploaded_date','updated_date')
        read_only_fields = ('feature','extract_time')

