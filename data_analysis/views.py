from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView, View
from django.urls import reverse_lazy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from .models import Dataset, Analysis, MLModel
from .forms import DatasetUploadForm, AnalysisForm, MLModelForm, TransformationForm
import csv
from django.db import models
from django.core.files.base import ContentFile
from datetime import datetime
import os

class DatasetListView(ListView):
    """
    View for displaying a list of all datasets.
    Lists datasets in reverse chronological order.
    """
    model = Dataset
    template_name = 'data_analysis/dataset_list.html'
    context_object_name = 'datasets'
    ordering = ['-uploaded_at']

class DatasetDetailView(DetailView):
    """
    View for displaying detailed information about a dataset.
    Shows dataset preview, info, and recent analyses.
    """
    model = Dataset
    template_name = 'data_analysis/dataset_detail.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get enhanced dataset information
        context['dataset_info'] = self.object.get_info()
        context['preview'] = self.object.get_preview(rows=5)
        
        # Get recent analyses
        context['recent_analyses'] = self.object.analysis_set.all().order_by('-created_at')[:5]
        
        # Get analysis statistics
        analysis_stats = {
            'total': self.object.analysis_set.count(),
            'by_type': self.object.analysis_set.values('analysis_type').annotate(
                count=models.Count('id')
            ).order_by('-count')
        }
        context['analysis_stats'] = analysis_stats
        
        return context

class DatasetUploadView(CreateView):
    """
    View for uploading new datasets.
    Handles file validation and dataset creation.
    Supports Excel files with sheet selection.
    """
    model = Dataset
    form_class = DatasetUploadForm
    template_name = 'data_analysis/dataset_upload.html'
    success_url = reverse_lazy('data_analysis:dataset-list')

    def form_valid(self, form):
        try:
            dataset = form.save(commit=False)
            file_ext = os.path.splitext(dataset.file.name)[1].lower()
            
            # Handle Excel files with sheet selection
            if file_ext in ['.xlsx', '.xls']:
                all_sheets = form.cleaned_data.get('all_sheets')
                selected_sheets = form.cleaned_data.get('selected_sheets')
                
                if not all_sheets and not selected_sheets:
                    form.add_error(None, 'Please either select specific sheets or choose to load all sheets.')
                    return self.form_invalid(form)
                
                try:
                    if hasattr(dataset.file, 'temporary_file_path'):
                        excel_file = pd.ExcelFile(dataset.file.temporary_file_path())
                    else:
                        excel_file = pd.ExcelFile(dataset.file)
                        
                    available_sheets = excel_file.sheet_names
                    
                    if all_sheets:
                        dataset.metadata['sheets'] = available_sheets
                        dataset.metadata['selected_sheets'] = available_sheets
                    else:
                        # Validate selected sheets exist
                        invalid_sheets = set(selected_sheets) - set(available_sheets)
                        if invalid_sheets:
                            form.add_error('selected_sheets', f'Invalid sheets selected: {", ".join(invalid_sheets)}')
                            return self.form_invalid(form)
                            
                        dataset.metadata['sheets'] = available_sheets
                        dataset.metadata['selected_sheets'] = selected_sheets
                except Exception as e:
                    form.add_error('file', f'Error reading Excel file: {str(e)}')
                    return self.form_invalid(form)
            
            # Save the dataset
            dataset.save()
            messages.success(self.request, 'Dataset uploaded successfully!')
            return super().form_valid(form)
            
        except Exception as e:
            messages.error(self.request, f'Error uploading dataset: {str(e)}')
            return self.form_invalid(form)

    def form_invalid(self, form):
        for field, errors in form.errors.items():
            for error in errors:
                messages.error(self.request, f'{field}: {error}' if field != '__all__' else error)
        return super().form_invalid(form)

class DatasetDeleteView(DeleteView):
    """
    View for deleting datasets.
    Handles file cleanup and database record deletion.
    """
    model = Dataset
    template_name = 'data_analysis/dataset_confirm_delete.html'
    success_url = reverse_lazy('data_analysis:dataset-list')

    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Dataset deleted successfully!')
        return super().delete(request, *args, **kwargs)

class DatasetExportView(View):
    """
    View for exporting datasets as CSV files.
    Provides downloadable dataset files.
    """
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        df = dataset.read_file()
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{dataset.name}.csv"'
        
        df.to_csv(response, index=False)
        return response

class AnalysisCreateView(CreateView):
    """
    View for creating new analyses.
    Handles different types of analyses (STATS, CORR, DIST).
    """
    model = Analysis
    form_class = AnalysisForm
    template_name = 'data_analysis/analysis_create.html'

    def get_initial(self):
        initial = super().get_initial()
        dataset_id = self.request.GET.get('dataset')
        if dataset_id:
            initial['dataset'] = dataset_id
        return initial

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset_id = self.request.GET.get('dataset')
        if dataset_id:
            context['dataset'] = get_object_or_404(Dataset, pk=dataset_id)
        return context

    def form_valid(self, form):
        analysis = form.save(commit=False)
        dataset = analysis.dataset
        
        # Read the dataset file
        df_or_dict = dataset.read_file()
        
        # Handle multi-sheet Excel files: use the first sheet
        if isinstance(df_or_dict, dict):
            first_sheet_name = list(df_or_dict.keys())[0]
            df = df_or_dict[first_sheet_name]
            # Optionally, inform the user which sheet was used
            messages.info(self.request, f"Analyzing the first sheet: '{first_sheet_name}'")
        elif isinstance(df_or_dict, pd.DataFrame):
            df = df_or_dict
        else:
            messages.error(self.request, "Could not read the dataset file correctly.")
            return self.form_invalid(form)
        
        # Perform analysis based on the type
        if analysis.analysis_type == 'STATS':
            # Calculate basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            results = {}
            for col in numeric_cols:
                stats = df[col].describe()
                results[col] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'percentile_25': stats['25%'],
                    'percentile_50': stats['50%'],
                    'percentile_75': stats['75%'],
                    'max': stats['max']
                }
            analysis.results = results
            
        elif analysis.analysis_type == 'CORR':
            # Calculate correlation matrix for numeric columns
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            corr_matrix = numeric_df.corr()
            analysis.results = {
                'matrix': {
                    'values': corr_matrix.values.tolist(),
                    'columns': corr_matrix.columns.tolist()
                }
            }
            
        elif analysis.analysis_type == 'DIST':
            # Calculate distribution statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            results = {}
            for col in numeric_cols:
                hist, bin_edges = np.histogram(df[col].dropna(), bins='auto')
                results[col] = {
                    'bins': bin_edges.tolist(),
                    'counts': hist.tolist(),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'skewness': float(df[col].skew()),
                    'kurtosis': float(df[col].kurtosis())
                }
            analysis.results = results
        
        analysis.save()
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('data_analysis:analysis-detail', kwargs={'pk': self.object.pk})

class AnalysisDetailView(DetailView):
    """
    View for displaying analysis results.
    Shows visualizations and statistics based on analysis type.
    """
    model = Analysis
    template_name = 'data_analysis/analysis_detail.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Format the results based on analysis type
        analysis = self.object
        if analysis.analysis_type == 'STATS':
            # Basic statistics are already in the correct format
            pass
        elif analysis.analysis_type == 'CORR':
            # Correlation matrix is already in the correct format
            pass
        elif analysis.analysis_type == 'DIST':
            # Ensure histogram data is properly formatted for Plotly
            for column, stats in analysis.results.items():
                # Convert bin edges to bin centers for x-axis
                bins = stats['bins']
                stats['bins'] = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
                # Ensure counts match the number of bins
                stats['counts'] = stats['counts'][:len(stats['bins'])]
        
        return context

class MLModelCreateView(CreateView):
    """
    View for creating and training ML models.
    Handles model training, validation, and metrics calculation.
    """
    model = MLModel
    form_class = MLModelForm
    template_name = 'data_analysis/mlmodel_create.html'
    success_url = reverse_lazy('mlmodel-list')

    def form_valid(self, form):
        mlmodel = form.save(commit=False)
        dataset = mlmodel.dataset
        df = pd.read_csv(dataset.file.path)

        # Prepare features and target
        X = df[mlmodel.features]
        y = df[mlmodel.target]

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model and save metrics
        model = self.get_model_instance(mlmodel.model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate and save metrics
        metrics = self.calculate_metrics(y_test, y_pred, mlmodel.model_type)
        mlmodel.metrics = metrics
        mlmodel.save()

        messages.success(self.request, 'Model trained successfully!')
        return super().form_valid(form)

    def get_model_instance(self, model_type):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        models = {
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'LR': LinearRegression(),
            'LOG': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42),
            'DT': DecisionTreeClassifier(random_state=42)
        }
        return models[model_type]

    def calculate_metrics(self, y_true, y_pred, model_type):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import mean_squared_error, r2_score

        if model_type in ['RF', 'LOG', 'SVM', 'DT']:  # Classification
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
        else:  # Regression
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }

class MLModelDetailView(DetailView):
    """
    View for displaying ML model details.
    Shows model performance metrics and feature importance.
    """
    model = MLModel
    template_name = 'data_analysis/mlmodel_detail.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

class MLModelListView(ListView):
    """
    View for displaying a list of all ML models.
    Shows model types and performance metrics.
    """
    model = MLModel
    template_name = 'data_analysis/mlmodel_list.html'
    context_object_name = 'mlmodels'
    ordering = ['-created_at']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        for mlmodel in context['mlmodels']:
            if mlmodel.metrics:
                if 'accuracy' in mlmodel.metrics:  # Classification metrics
                    mlmodel.metric_display = f"Accuracy: {mlmodel.metrics['accuracy']:.2f}"
                elif 'r2' in mlmodel.metrics:  # Regression metrics
                    mlmodel.metric_display = f"R² Score: {mlmodel.metrics['r2']:.2f}"
        return context

class AnalysisListView(ListView):
    """
    View for displaying a list of all analyses.
    Can be filtered by dataset.
    """
    model = Analysis
    template_name = 'data_analysis/analysis_list.html'
    context_object_name = 'analyses'
    ordering = ['-created_at']

    def get_queryset(self):
        dataset_id = self.request.GET.get('dataset')
        queryset = super().get_queryset()
        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset_id = self.request.GET.get('dataset')
        if dataset_id:
            context['dataset'] = get_object_or_404(Dataset, pk=dataset_id)
        return context

class TransformationView(View):
    template_name = 'data_analysis/transformation_form.html'

    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        form = TransformationForm(dataset=dataset)
        return render(request, self.template_name, {
            'form': form,
            'dataset': dataset
        })

    def post(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        form = TransformationForm(request.POST, dataset=dataset)
        
        if form.is_valid():
            try:
                # Read the dataset
                df = dataset.read_file()
                
                # Get form data
                transformation_type = form.cleaned_data['transformation_type']
                operation = form.cleaned_data['operation']
                columns = form.cleaned_data['columns']
                
                # Apply transformation
                transformed_df = self.apply_transformation(
                    df,
                    transformation_type,
                    operation,
                    columns,
                    form.cleaned_data
                )
                
                # Save transformed dataset
                new_name = f"{dataset.name}_transformed"
                new_description = f"Transformed version of {dataset.name} using {operation}"
                
                # Convert DataFrame to CSV
                csv_buffer = transformed_df.to_csv(index=False)
                
                # Create new dataset
                new_dataset = Dataset.objects.create(
                    name=new_name,
                    description=new_description
                )
                
                # Save the CSV content to the new dataset's file field
                new_dataset.file.save(
                    f"{new_name}.csv",
                    ContentFile(csv_buffer.encode('utf-8'))
                )
                
                messages.success(request, 'Dataset transformed successfully!')
                return redirect('data_analysis:dataset-detail', pk=new_dataset.pk)
                
            except Exception as e:
                messages.error(request, f'Error transforming dataset: {str(e)}')
        
        return render(request, self.template_name, {
            'form': form,
            'dataset': dataset
        })

    def apply_transformation(self, df, transformation_type, operation, columns, params):
        if transformation_type == 'CLEAN':
            return self.apply_cleaning(df, operation, columns)
        elif transformation_type == 'ENCODE':
            return self.apply_encoding(df, operation, columns)
        elif transformation_type == 'SCALE':
            return self.apply_scaling(df, operation, columns)
        elif transformation_type == 'ENGINEER':
            return self.apply_engineering(df, operation, columns, params)
        return df

    def apply_cleaning(self, df, operation, columns):
        if operation == 'drop_na':
            return df.dropna(subset=columns)
        elif operation == 'fill_na_mean':
            return df.fillna(df[columns].mean())
        elif operation == 'fill_na_median':
            return df.fillna(df[columns].median())
        elif operation == 'fill_na_mode':
            return df.fillna(df[columns].mode().iloc[0])
        elif operation == 'drop_duplicates':
            return df.drop_duplicates(subset=columns)
        elif operation == 'remove_outliers':
            df_clean = df.copy()
            for col in columns:
                if df[col].dtype in ['int64', 'float64']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df_clean = df_clean[
                        (df_clean[col] >= Q1 - 1.5 * IQR) &
                        (df_clean[col] <= Q3 + 1.5 * IQR)
                    ]
            return df_clean
        return df

    def apply_encoding(self, df, operation, columns):
        df_encoded = df.copy()
        if operation == 'label':
            le = LabelEncoder()
            for col in columns:
                if df[col].dtype == 'object':
                    df_encoded[col] = le.fit_transform(df[col].astype(str))
        elif operation == 'onehot':
            for col in columns:
                if df[col].dtype == 'object':
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(col, axis=1, inplace=True)
        elif operation == 'ordinal':
            for col in columns:
                if df[col].dtype == 'object':
                    categories = df[col].unique()
                    category_dict = {cat: i for i, cat in enumerate(sorted(categories))}
                    df_encoded[col] = df[col].map(category_dict)
        return df_encoded

    def apply_scaling(self, df, operation, columns):
        df_scaled = df.copy()
        if operation == 'standard':
            scaler = StandardScaler()
        elif operation == 'minmax':
            scaler = MinMaxScaler()
        elif operation == 'robust':
            scaler = RobustScaler()
        elif operation == 'normalize':
            df_scaled[columns] = df_scaled[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            return df_scaled
        
        # Apply scaler to selected columns
        df_scaled[columns] = scaler.fit_transform(df[columns])
        return df_scaled

    def apply_engineering(self, df, operation, columns, params):
        df_engineered = df.copy()
        if operation == 'polynomial':
            degree = params.get('polynomial_degree', 2)
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(df[columns])
            feature_names = poly.get_feature_names_out(columns)
            
            # Add polynomial features to dataframe
            poly_df = pd.DataFrame(
                poly_features[:, len(columns):],
                columns=feature_names[len(columns):],
                index=df.index
            )
            df_engineered = pd.concat([df_engineered, poly_df], axis=1)
            
        elif operation == 'interaction':
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    col1, col2 = columns[i], columns[j]
                    new_col = f"{col1}_{col2}_interaction"
                    df_engineered[new_col] = df[col1] * df[col2]
                    
        elif operation == 'binning':
            n_bins = params.get('n_bins', 5)
            for col in columns:
                if df[col].dtype in ['int64', 'float64']:
                    df_engineered[f"{col}_binned"] = pd.qcut(
                        df[col],
                        q=n_bins,
                        labels=[f"bin_{i+1}" for i in range(n_bins)]
                    )
                    
        elif operation == 'log':
            for col in columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Handle negative values
                    min_val = df[col].min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                        df_engineered[f"{col}_log"] = np.log(df[col] + shift)
                    else:
                        df_engineered[f"{col}_log"] = np.log(df[col])
                        
        elif operation == 'sqrt':
            for col in columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Handle negative values
                    min_val = df[col].min()
                    if min_val < 0:
                        shift = abs(min_val)
                        df_engineered[f"{col}_sqrt"] = np.sqrt(df[col] + shift)
                    else:
                        df_engineered[f"{col}_sqrt"] = np.sqrt(df[col])
                        
        elif operation == 'power':
            power = params.get('power_value', 2)
            for col in columns:
                if df[col].dtype in ['int64', 'float64']:
                    df_engineered[f"{col}_power_{power}"] = np.power(df[col], power)
                    
        return df_engineered

class TransformationPreviewView(View):
    def post(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        form = TransformationForm(request.POST, dataset=dataset)
        
        if form.is_valid():
            try:
                # Read the dataset
                df = dataset.read_file()
                
                # Get form data
                transformation_type = form.cleaned_data['transformation_type']
                operation = form.cleaned_data['operation']
                columns = form.cleaned_data['columns']
                
                # Apply transformation
                transformer = TransformationView()
                transformed_df = transformer.apply_transformation(
                    df,
                    transformation_type,
                    operation,
                    columns,
                    form.cleaned_data
                )
                
                # Generate preview
                preview_df = transformed_df.head()
                preview_html = preview_df.to_html(
                    classes=['table', 'table-striped', 'table-sm'],
                    float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x
                )
                
                # Calculate summary of changes
                n_rows_before = len(df)
                n_rows_after = len(transformed_df)
                n_cols_before = len(df.columns)
                n_cols_after = len(transformed_df.columns)
                
                message = (
                    f"Rows: {n_rows_before} → {n_rows_after}, "
                    f"Columns: {n_cols_before} → {n_cols_after}"
                )
                
                return JsonResponse({
                    'preview_table': preview_html,
                    'message': message
                })
                
            except Exception as e:
                return JsonResponse({'error': str(e)})
        
        return JsonResponse({'error': 'Invalid form data'})

class GetSheetsView(View):
    def post(self, request):
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)
            
        file = request.FILES['file']
        if not file.name.endswith(('.xlsx', '.xls')):
            return JsonResponse({'error': 'Not an Excel file'}, status=400)
            
        try:
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names
            
            # Get preview of first sheet
            df = pd.read_excel(file, sheet_name=sheet_names[0], nrows=5)
            preview_html = df.to_html(
                classes=['table', 'table-striped', 'table-bordered', 'table-hover', 'table-sm'],
                float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x,
                index=False
            )
            
            return JsonResponse({
                'sheets': sheet_names,
                'preview': preview_html,
                'columns': list(df.columns),
                'num_rows': len(df)
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

class PreviewDataView(View):
    """View for previewing dataset contents before upload."""
    def post(self, request):
        try:
            file = request.FILES.get('file')
            if not file:
                return JsonResponse({'error': 'No file provided'}, status=400)

            sheet_name = request.POST.get('sheet_name')
            
            # Read the first few rows for preview
            if file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, sheet_name=sheet_name, nrows=5)
            else:
                df = pd.read_csv(file, nrows=5)
            
            # Convert DataFrame to HTML table with Bootstrap classes
            preview_html = df.to_html(
                classes=['table', 'table-striped', 'table-bordered', 'table-hover', 'table-sm'],
                float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x,
                index=False
            )
            
            return JsonResponse({
                'preview': preview_html,
                'columns': list(df.columns),
                'num_rows': len(df)
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
