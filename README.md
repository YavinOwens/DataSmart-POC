# Data Analysis Web Application

A Django-based web application for data analysis and machine learning. This application allows users to upload datasets, perform various analyses, and train machine learning models through a modern web interface.

## Features

- **Dataset Management**
  - Upload CSV and Excel files
  - View dataset statistics and previews
  - Export datasets
  - Manage multiple datasets

- **Data Analysis**
  - Basic statistics (mean, std, min, max, etc.)
  - Correlation analysis
  - Distribution analysis with visualizations
  - Interactive data exploration

- **Machine Learning**
  - Train various ML models:
    - Random Forest (Classification/Regression)
    - Linear/Logistic Regression
    - Support Vector Machines
    - Decision Trees
  - Model performance metrics
  - Feature selection
  - Model comparison

## Technology Stack

- Python 3.9+
- Django 4.2
- Pandas for data manipulation
- Scikit-learn for machine learning
- Plotly.js for visualizations
- Material Design-inspired UI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YavinOwens/django_webapp.git
cd data-analysis-webapp
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Apply database migrations:
```bash
python manage.py migrate
```

5. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

6. Run the development server:
```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`.

## Project Structure

```
crud_project/
├── data_analysis/          # Main application
│   ├── models.py          # Database models
│   ├── views.py           # View logic
│   ├── forms.py           # Form definitions
│   ├── urls.py           # URL routing
│   ├── templates/        # HTML templates
│   └── static/           # Static files (CSS, JS)
├── crud_project/         # Project settings
├── manage.py            # Django management script
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Usage

1. **Upload Dataset**
   - Click "Upload Dataset" button
   - Enter dataset name and description
   - Upload CSV or Excel file
   - View dataset statistics

2. **Perform Analysis**
   - Select a dataset
   - Choose analysis type
   - View results with visualizations
   - Export results if needed

3. **Train ML Models**
   - Select a dataset
   - Choose model type
   - Select features and target
   - Train and evaluate model

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Django framework and community
- Scikit-learn for machine learning functionality
- Plotly.js for interactive visualizations
- Material Design for UI inspiration