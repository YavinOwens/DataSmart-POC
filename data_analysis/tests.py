from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import Dataset, Analysis, MLModel
import pandas as pd
import numpy as np
import os
import json
from io import BytesIO
from openpyxl import Workbook

class DatasetModelTests(TestCase):
    def setUp(self):
        # Create a test file
        self.test_file = SimpleUploadedFile(
            name="test.csv",
            content=b"col1,col2\n1,2\n3,4",
            content_type="text/csv"
        )
        self.dataset = Dataset.objects.create(
            name="Test Dataset",
            description="Test Description",
            file=self.test_file
        )

    def test_dataset_creation(self):
        """Test dataset creation and basic attributes"""
        self.assertEqual(self.dataset.name, "Test Dataset")
        self.assertEqual(self.dataset.description, "Test Description")
        self.assertTrue(self.dataset.file.name.endswith('.csv'))
        self.assertIsNotNone(self.dataset.uploaded_at)
        self.assertIsInstance(self.dataset.metadata, dict)
        self.assertEqual(self.dataset.metadata['original_filename'], 'test.csv')

    def test_dataset_str(self):
        """Test string representation"""
        self.assertEqual(str(self.dataset), "Test Dataset")

    def test_read_file(self):
        """Test reading dataset file"""
        df = self.dataset.read_file()
        self.assertIsNotNone(df)
        self.assertEqual(list(df.columns), ['col1', 'col2'])
        self.assertEqual(len(df), 2)

class AnalysisModelTests(TestCase):
    def setUp(self):
        # Create test dataset and analysis
        self.csv_content = b'col1,col2\n1,2\n3,4'
        self.test_file = SimpleUploadedFile(
            "test.csv",
            self.csv_content,
            content_type="text/csv"
        )
        self.dataset = Dataset.objects.create(
            name="Test Dataset",
            file=self.test_file
        )
        self.analysis = Analysis.objects.create(
            dataset=self.dataset,
            analysis_type='STATS'
        )
    
    def test_analysis_creation(self):
        """Test analysis creation and basic attributes"""
        self.assertEqual(self.analysis.dataset, self.dataset)
        self.assertEqual(self.analysis.analysis_type, 'STATS')
        self.assertIsNone(self.analysis.results)

class MLModelTests(TestCase):
    def setUp(self):
        # Create test dataset and ML model
        self.csv_content = b'col1,col2,target\n1,2,0\n3,4,1'
        self.test_file = SimpleUploadedFile(
            "test.csv",
            self.csv_content,
            content_type="text/csv"
        )
        self.dataset = Dataset.objects.create(
            name="Test Dataset",
            file=self.test_file
        )
        self.model = MLModel.objects.create(
            name="Test Model",
            dataset=self.dataset,
            model_type='RF',
            features=['col1', 'col2'],
            target='target'
        )
    
    def test_model_creation(self):
        """Test ML model creation and basic attributes"""
        self.assertEqual(self.model.name, "Test Model")
        self.assertEqual(self.model.dataset, self.dataset)
        self.assertEqual(self.model.model_type, 'RF')
        self.assertEqual(self.model.features, ['col1', 'col2'])
        self.assertEqual(self.model.target, 'target')

class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Create test dataset
        self.csv_content = b'col1,col2\n1,2\n3,4'
        self.test_file = SimpleUploadedFile(
            "test.csv",
            self.csv_content,
            content_type="text/csv"
        )
        self.dataset = Dataset.objects.create(
            name="Test Dataset",
            file=self.test_file
        )
    
    def test_dataset_list_view(self):
        """Test dataset list view"""
        response = self.client.get(reverse('data_analysis:dataset-list'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'data_analysis/dataset_list.html')
        self.assertContains(response, "Test Dataset")
    
    def test_dataset_detail_view(self):
        """Test dataset detail view"""
        response = self.client.get(
            reverse('data_analysis:dataset-detail', args=[self.dataset.pk])
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'data_analysis/dataset_detail.html')
        self.assertContains(response, "Test Dataset")
    
    def test_dataset_upload_view(self):
        """Test dataset upload view"""
        response = self.client.get(reverse('data_analysis:dataset-upload'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'data_analysis/dataset_upload.html')
    
    def test_dataset_delete_view(self):
        """Test dataset delete view"""
        response = self.client.post(
            reverse('data_analysis:dataset-delete', args=[self.dataset.pk])
        )
        self.assertEqual(response.status_code, 302)  # Redirect after deletion
        self.assertEqual(Dataset.objects.count(), 0)

class DatasetUploadTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.upload_url = reverse('data_analysis:dataset-upload')
        
        # Create test files
        self.csv_file = SimpleUploadedFile(
            name="test.csv",
            content=b"col1,col2\n1,2\n3,4",
            content_type="text/csv"
        )
        
        # Create Excel file with multiple sheets
        wb = Workbook()
        ws1 = wb.active
        ws1.title = "Sheet1"
        ws1['A1'] = "col1"
        ws1['B1'] = "col2"
        ws1['A2'] = 1
        ws1['B2'] = 2
        
        ws2 = wb.create_sheet("Sheet2")
        ws2['A1'] = "col3"
        ws2['B1'] = "col4"
        ws2['A2'] = 3
        ws2['B2'] = 4
        
        excel_file = BytesIO()
        wb.save(excel_file)
        excel_file.seek(0)
        
        self.excel_file = SimpleUploadedFile(
            name="test.xlsx",
            content=excel_file.read(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    def test_upload_csv(self):
        """Test uploading CSV file"""
        data = {
            'name': 'Test CSV Dataset',
            'description': 'Test Description',
            'file': self.csv_file
        }
        response = self.client.post(self.upload_url, data)
        self.assertEqual(response.status_code, 302)  # Redirect on success
        self.assertEqual(Dataset.objects.count(), 1)
        dataset = Dataset.objects.first()
        self.assertEqual(dataset.name, 'Test CSV Dataset')

    def test_upload_excel_all_sheets(self):
        """Test uploading Excel file with all sheets"""
        data = {
            'name': 'Test Excel Dataset',
            'description': 'Test Description',
            'file': self.excel_file,
            'all_sheets': True
        }
        response = self.client.post(self.upload_url, data)
        self.assertEqual(response.status_code, 302)  # Redirect on success
        self.assertEqual(Dataset.objects.count(), 1)
        dataset = Dataset.objects.first()
        self.assertEqual(dataset.name, 'Test Excel Dataset')
        self.assertIn('Sheet1', dataset.metadata['sheets'])
        self.assertIn('Sheet2', dataset.metadata['sheets'])

    def test_upload_excel_selected_sheets(self):
        """Test uploading Excel file with selected sheets"""
        data = {
            'name': 'Test Excel Dataset',
            'description': 'Test Description',
            'file': self.excel_file,
            'all_sheets': False,
            'selected_sheets': ['Sheet1']
        }
        response = self.client.post(self.upload_url, data)
        self.assertEqual(response.status_code, 302)  # Redirect on success
        self.assertEqual(Dataset.objects.count(), 1)
        dataset = Dataset.objects.first()
        self.assertEqual(dataset.name, 'Test Excel Dataset')
        self.assertEqual(dataset.metadata['selected_sheets'], ['Sheet1'])

    def test_upload_excel_no_sheet_selection(self):
        """Test uploading Excel file without selecting sheets"""
        data = {
            'name': 'Test Excel Dataset',
            'description': 'Test Description',
            'file': self.excel_file,
            'all_sheets': False,
            'selected_sheets': []
        }
        response = self.client.post(self.upload_url, data)
        self.assertEqual(response.status_code, 200)  # Form invalid
        self.assertFormError(response, 'form', None, 'Please either select specific sheets or choose to load all sheets.')
