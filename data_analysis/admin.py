from django.contrib import admin
from .models import Dataset, Analysis, MLModel

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'uploaded_at')
    search_fields = ('name', 'description')
    list_filter = ('uploaded_at',)
    readonly_fields = ('uploaded_at',)

@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'analysis_type', 'created_at')
    list_filter = ('analysis_type', 'created_at')
    search_fields = ('dataset__name',)
    readonly_fields = ('created_at',)

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'dataset', 'model_type', 'created_at')
    list_filter = ('model_type', 'created_at')
    search_fields = ('name', 'dataset__name')
    readonly_fields = ('created_at', 'updated_at')
