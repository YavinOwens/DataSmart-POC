from django.db import models
from django.core.validators import FileExtensionValidator
import pandas as pd
import os
import json
from datetime import datetime

class Dataset(models.Model):
    """
    Model for storing uploaded datasets and their metadata.
    Supports CSV and Excel files with sheet selection.
    """
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file = models.FileField(
        upload_to='datasets/',
        validators=[FileExtensionValidator(allowed_extensions=['csv', 'xlsx', 'xls'])]
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict)
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        # Initialize metadata if it's None
        if self.metadata is None:
            self.metadata = {}
            
        # Update metadata with file info
        if self.file and not self.file._committed:
            self.metadata.update({
                'original_filename': os.path.basename(self.file.name),
                'file_size': self.file.size,
                'file_type': os.path.splitext(self.file.name)[1].lower()[1:],
                'upload_date': datetime.now().isoformat()
            })
            
        super().save(*args, **kwargs)
    
    def read_file(self, sheet_name=None):
        """
        Read the dataset file into a pandas DataFrame.
        For Excel files, can specify sheet_name to read specific sheets.
        """
        if not self.file:
            return None
            
        file_ext = os.path.splitext(self.file.name)[1].lower()
        
        try:
            if file_ext in ['.xlsx', '.xls']:
                if sheet_name:
                    return pd.read_excel(self.file.path, sheet_name=sheet_name)
                else:
                    # If no sheet specified, read all sheets
                    return pd.read_excel(self.file.path, sheet_name=None)
            else:
                return pd.read_csv(self.file.path)
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    
    def get_sheet_names(self):
        """Get list of sheet names for Excel files."""
        if not self.file:
            return None
            
        file_ext = os.path.splitext(self.file.name)[1].lower()
        if file_ext in ['.xlsx', '.xls']:
            try:
                excel_file = pd.ExcelFile(self.file.path)
                return excel_file.sheet_names
            except Exception as e:
                raise ValueError(f"Error reading Excel file: {str(e)}")
        return None
    
    def get_preview(self, rows=5, sheet_name=None):
        """Get a preview of the dataset."""
        df = self.read_file(sheet_name)
        if isinstance(df, dict):  # Multiple sheets
            return {name: sheet.head(rows).to_dict('records') for name, sheet in df.items()}
        return df.head(rows).to_dict('records')
    
    def get_info(self, sheet_name=None):
        """Get dataset information including column types and statistics."""
        df = self.read_file(sheet_name)
        
        if isinstance(df, dict):  # Multiple sheets
            info = {}
            for name, sheet in df.items():
                info[name] = self._get_sheet_info(sheet)
            return info
        
        return self._get_sheet_info(df)
    
    def _get_sheet_info(self, df):
        """Helper method to get information for a single dataframe."""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_types': {
                col: str(df[col].dtype) for col in df.columns
            },
            'numeric_columns': list(numeric_cols),
            'categorical_columns': list(categorical_cols),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Basic statistics for numeric columns
        if len(numeric_cols) > 0:
            info['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Value counts for categorical columns
        info['categorical_stats'] = {
            col: df[col].value_counts().head().to_dict()
            for col in categorical_cols
        }
        
        return info

    def delete(self, *args, **kwargs):
        # Delete the file when the model instance is deleted
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)

class Analysis(models.Model):
    """
    Model representing an analysis performed on a dataset.
    
    Attributes:
        dataset (ForeignKey): The dataset being analyzed
        analysis_type (str): Type of analysis (STATS, CORR, DIST)
        results (JSONField): The analysis results stored as JSON
        created_at (datetime): When the analysis was created
    """
    ANALYSIS_TYPES = [
        ('STATS', 'Basic Statistics'),
        ('CORR', 'Correlation Analysis'),
        ('DIST', 'Distribution Analysis'),
    ]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    analysis_type = models.CharField(max_length=10, choices=ANALYSIS_TYPES)
    results = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_analysis_type_display()} for {self.dataset.name}"

    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('data_analysis:analysis-detail', kwargs={'pk': self.pk})

class MLModel(models.Model):
    """
    Model representing a machine learning model trained on a dataset.
    
    Attributes:
        name (str): The name of the model
        dataset (ForeignKey): The dataset used for training
        model_type (str): Type of model (RF, LR, LOG, SVM, DT)
        features (JSONField): List of feature column names
        target (str): Target column name
        metrics (JSONField): Model performance metrics
        created_at (datetime): When the model was created
        updated_at (datetime): When the model was last updated
    """
    MODEL_TYPES = [
        ('RF', 'Random Forest'),
        ('LR', 'Linear Regression'),
        ('LOG', 'Logistic Regression'),
        ('SVM', 'Support Vector Machine'),
        ('DT', 'Decision Tree'),
    ]

    name = models.CharField(max_length=255)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    model_type = models.CharField(max_length=10, choices=MODEL_TYPES)
    features = models.JSONField(help_text='List of feature column names')
    target = models.CharField(max_length=255, help_text='Target column name')
    metrics = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"
