from django.urls import path
from . import views
from .views import TransformationView, TransformationPreviewView, PreviewDataView

app_name = 'data_analysis'

urlpatterns = [
    path('', views.DatasetListView.as_view(), name='dataset-list'),
    path('dataset/upload/', views.DatasetUploadView.as_view(), name='dataset-upload'),
    path('dataset/<int:pk>/', views.DatasetDetailView.as_view(), name='dataset-detail'),
    path('dataset/<int:pk>/delete/', views.DatasetDeleteView.as_view(), name='dataset-delete'),
    path('dataset/<int:pk>/export/', views.DatasetExportView.as_view(), name='dataset-export'),
    path('dataset/get-sheets/', views.GetSheetsView.as_view(), name='get-sheets'),
    path('dataset/preview-data/', PreviewDataView.as_view(), name='preview-data'),
    path('analysis/', views.AnalysisListView.as_view(), name='analysis-list'),
    path('analysis/create/', views.AnalysisCreateView.as_view(), name='analysis-create'),
    path('analysis/<int:pk>/', views.AnalysisDetailView.as_view(), name='analysis-detail'),
    path('mlmodel/', views.MLModelListView.as_view(), name='mlmodel-list'),
    path('mlmodel/create/', views.MLModelCreateView.as_view(), name='mlmodel-create'),
    path('mlmodel/<int:pk>/', views.MLModelDetailView.as_view(), name='mlmodel-detail'),
    path('dataset/<int:pk>/transform/', TransformationView.as_view(), name='transform'),
    path('dataset/<int:pk>/transform-preview/', TransformationPreviewView.as_view(), name='transform-preview'),
    path('chatbot/', views.ChatbotView.as_view(), name='chatbot'),
] 