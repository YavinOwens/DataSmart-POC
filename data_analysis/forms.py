from django import forms
from .models import Dataset, Analysis, MLModel
import pandas as pd
import os

class DatasetUploadForm(forms.ModelForm):
    """Form for uploading datasets with Excel sheet selection support."""
    all_sheets = forms.BooleanField(required=False, initial=True)
    selected_sheets = forms.MultipleChoiceField(required=False, choices=[])
    
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file', 'all_sheets', 'selected_sheets']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['selected_sheets'].choices = []
        
        # Add Bootstrap classes
        self.fields['name'].widget.attrs.update({'class': 'form-control'})
        self.fields['description'].widget.attrs.update({'class': 'form-control'})
        self.fields['file'].widget.attrs.update({
            'class': 'form-control',
            'accept': '.csv,.xlsx,.xls'
        })
        
        if self.is_bound and self.files:
            file = self.files.get('file')
            if file and file.name.endswith(('.xlsx', '.xls')):
                try:
                    if hasattr(file, 'temporary_file_path'):
                        excel_file = pd.ExcelFile(file.temporary_file_path())
                    else:
                        excel_file = pd.ExcelFile(file)
                    sheet_names = excel_file.sheet_names
                    self.fields['selected_sheets'].choices = [(name, name) for name in sheet_names]
                except Exception as e:
                    self.add_error('file', f'Error reading Excel file: {str(e)}')
        
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if not file:
            raise forms.ValidationError('Please select a file to upload.')
            
        # Check file extension
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in ['.csv', '.xlsx', '.xls']:
            raise forms.ValidationError('Only CSV and Excel files are supported.')
            
        # Check file size (max 10MB)
        if file.size > 10 * 1024 * 1024:
            raise forms.ValidationError('File size must be less than 10MB.')
            
        return file
        
    def clean(self):
        cleaned_data = super().clean()
        file = cleaned_data.get('file')
        all_sheets = cleaned_data.get('all_sheets')
        selected_sheets = cleaned_data.get('selected_sheets')
        
        if file and file.name.endswith(('.xlsx', '.xls')):
            if not all_sheets and not selected_sheets:
                self.add_error(None, 'Please either select specific sheets or choose to load all sheets.')
            
            # Validate file can be read
            try:
                if hasattr(file, 'temporary_file_path'):
                    excel_file = pd.ExcelFile(file.temporary_file_path())
                else:
                    excel_file = pd.ExcelFile(file)
                available_sheets = excel_file.sheet_names
                
                if not all_sheets and selected_sheets:
                    invalid_sheets = set(selected_sheets) - set(available_sheets)
                    if invalid_sheets:
                        self.add_error('selected_sheets', f'Invalid sheets selected: {", ".join(invalid_sheets)}')
            except Exception as e:
                self.add_error('file', f'Error reading Excel file: {str(e)}')
        
        return cleaned_data

class AnalysisForm(forms.ModelForm):
    class Meta:
        model = Analysis
        fields = ['dataset', 'analysis_type']
        widgets = {
            'dataset': forms.HiddenInput(),
            'analysis_type': forms.RadioSelect(attrs={'class': 'analysis-type-radio'})
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['analysis_type'].widget.attrs['class'] = 'form-check-input'
        self.fields['analysis_type'].help_text = {
            'STATS': 'Calculate basic statistics (mean, std, min, max, etc.) for numeric columns',
            'CORR': 'Calculate correlation matrix for numeric columns',
            'DIST': 'Generate histograms and distribution statistics for numeric columns'
        }

class MLModelForm(forms.ModelForm):
    features = forms.MultipleChoiceField(
        choices=[],
        widget=forms.SelectMultiple(attrs={
            'class': 'form-control',
            'size': '8',  # Show more options at once
        }),
        help_text='Select multiple features to use for training (Ctrl/Cmd + click)'
    )
    
    target = forms.ChoiceField(
        choices=[],
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Select the target variable to predict'
    )

    class Meta:
        model = MLModel
        fields = ['name', 'dataset', 'model_type', 'features', 'target']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'dataset': forms.Select(attrs={'class': 'form-control'}),
            'model_type': forms.Select(attrs={'class': 'form-control'})
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.dataset_id:
            dataset = Dataset.objects.get(pk=self.instance.dataset_id)
        elif 'initial' in kwargs and 'dataset' in kwargs['initial']:
            dataset = Dataset.objects.get(pk=kwargs['initial']['dataset'])
        else:
            return

        # Get columns from dataset
        df = dataset.read_file()
        columns = [(col, col) for col in df.columns]
        
        # Update choices for features and target
        self.fields['features'].choices = columns
        self.fields['target'].choices = columns

        # Add help text for model types
        self.fields['model_type'].help_text = {
            'RF': 'Random Forest - Good for both classification and regression, handles non-linear relationships',
            'LR': 'Linear Regression - Best for predicting continuous values with linear relationships',
            'LOG': 'Logistic Regression - Best for binary/multiclass classification with linear boundaries',
            'SVM': 'Support Vector Machine - Good for high-dimensional data, works well with clear margins',
            'DT': 'Decision Tree - Easy to interpret, handles non-linear relationships'
        }[self.instance.model_type] if self.instance and self.instance.model_type else ''

    def clean(self):
        cleaned_data = super().clean()
        features = cleaned_data.get('features', [])
        target = cleaned_data.get('target')

        if target in features:
            raise forms.ValidationError(
                "Target variable cannot be included in features"
            )

        return cleaned_data 

class TransformationForm(forms.Form):
    TRANSFORMATION_TYPES = [
        ('CLEAN', 'Data Cleaning'),
        ('ENCODE', 'Encoding'),
        ('SCALE', 'Scaling'),
        ('ENGINEER', 'Feature Engineering'),
    ]

    CLEANING_OPERATIONS = [
        ('drop_na', 'Drop Missing Values'),
        ('fill_na_mean', 'Fill Missing Values with Mean'),
        ('fill_na_median', 'Fill Missing Values with Median'),
        ('fill_na_mode', 'Fill Missing Values with Mode'),
        ('drop_duplicates', 'Drop Duplicate Rows'),
        ('remove_outliers', 'Remove Outliers (IQR method)'),
    ]

    ENCODING_OPERATIONS = [
        ('label', 'Label Encoding'),
        ('onehot', 'One-Hot Encoding'),
        ('ordinal', 'Ordinal Encoding'),
    ]

    SCALING_OPERATIONS = [
        ('standard', 'Standard Scaling (Z-score)'),
        ('minmax', 'Min-Max Scaling'),
        ('robust', 'Robust Scaling'),
        ('normalize', 'Normalization'),
    ]

    ENGINEERING_OPERATIONS = [
        ('polynomial', 'Polynomial Features'),
        ('interaction', 'Interaction Terms'),
        ('binning', 'Equal-width Binning'),
        ('log', 'Logarithmic Transform'),
        ('sqrt', 'Square Root Transform'),
        ('power', 'Power Transform'),
    ]

    transformation_type = forms.ChoiceField(
        choices=TRANSFORMATION_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Select the type of transformation to apply'
    )

    columns = forms.MultipleChoiceField(
        choices=[],
        widget=forms.SelectMultiple(attrs={
            'class': 'form-control',
            'size': '8'
        }),
        help_text='Select columns to transform (Ctrl/Cmd + click for multiple)'
    )

    operation = forms.ChoiceField(
        choices=[],
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Select the specific operation to apply'
    )

    # Additional parameters
    n_bins = forms.IntegerField(
        required=False,
        min_value=2,
        max_value=100,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Number of bins for binning operation'
    )

    polynomial_degree = forms.IntegerField(
        required=False,
        min_value=2,
        max_value=3,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Degree for polynomial features'
    )

    power_value = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Power value for power transform'
    )

    def __init__(self, *args, **kwargs):
        dataset = kwargs.pop('dataset', None)
        super().__init__(*args, **kwargs)
        
        if dataset:
            df_or_dict = dataset.read_file()
            
            # Handle multi-sheet Excel files: use the first sheet
            if isinstance(df_or_dict, dict):
                first_sheet_name = list(df_or_dict.keys())[0]
                df = df_or_dict[first_sheet_name]
                # Consider adding a message or handling sheet selection later
            elif isinstance(df_or_dict, pd.DataFrame):
                df = df_or_dict
            else:
                df = None # Or raise an error/handle appropriately
            
            if df is not None:
                self.fields['columns'].choices = [(col, col) for col in df.columns]
            else:
                # Handle case where df couldn't be loaded
                self.fields['columns'].choices = [] 
                # Optionally disable the field or add an error
        
        # Set initial operation choices (can be updated in clean method or JS)
        self.fields['operation'].choices = self.CLEANING_OPERATIONS

    def clean(self):
        cleaned_data = super().clean()
        transformation_type = cleaned_data.get('transformation_type')
        operation = cleaned_data.get('operation')

        # Update operation choices based on transformation type
        if transformation_type == 'CLEAN':
            self.fields['operation'].choices = self.CLEANING_OPERATIONS
        elif transformation_type == 'ENCODE':
            self.fields['operation'].choices = self.ENCODING_OPERATIONS
        elif transformation_type == 'SCALE':
            self.fields['operation'].choices = self.SCALING_OPERATIONS
        elif transformation_type == 'ENGINEER':
            self.fields['operation'].choices = self.ENGINEERING_OPERATIONS

        # Validate required parameters
        if operation == 'binning' and not cleaned_data.get('n_bins'):
            raise forms.ValidationError('Number of bins is required for binning operation')
        elif operation == 'polynomial' and not cleaned_data.get('polynomial_degree'):
            raise forms.ValidationError('Polynomial degree is required for polynomial features')
        elif operation == 'power' and not cleaned_data.get('power_value'):
            raise forms.ValidationError('Power value is required for power transform')

        return cleaned_data 