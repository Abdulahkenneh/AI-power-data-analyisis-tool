# Standard library imports
import os
import re
import csv
import json
import logging
import io
import base64
import traceback
from datetime import datetime
from pathlib import Path
from uuid import UUID
from celery.result import AsyncResult

# Data science and machine learning imports
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Third-party libraries
import openai
import chardet
from fuzzywuzzy import process
from pandera import Column, DataFrameSchema
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from dotenv import load_dotenv

# Django framework imports
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.core.files.storage import default_storage
from django.db.models import Q
from django.http import JsonResponse, HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render

# Local application imports
from .forms import DataUploadForm
from .models import (
    Ai_generated_code,
    ChatMessage,
    ErrorLog,
    UserFile,
    UserProfile,
    UserloadedFile,
)
from .tasks import process_user_query  # Celery task

# Ensure matplotlib compatibility in some environments
matplotlib.use("Agg")

# Additional imports for data handling
from io import BytesIO
import openai
from django.core.cache import cache
from django.shortcuts import render
from django.http import JsonResponse

# UUID and path handling
from pathlib import Path
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .models import UserFile
import pandas as pd
import pyreadstat
import csv

# Handling of chat messages, error logs, and JSON data
from .models import ChatMessage, ErrorLog
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
import json




MAX_ATTEMPTS = 3
import logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


load_dotenv()
openai.api_key = os.getenv('API_SECRET_Key')



@login_required(login_url='userlogs/login')
def dashboard(request):
    form = DataUploadForm(request.POST or None, request.FILES or None)
    user_profile, created = UserProfile.objects.get_or_create(user=request.user)

    if request.method == 'POST' and form.is_valid():
        uploaded_file = form.cleaned_data.get('uploaded_files')
        if uploaded_file:
            # Check file limit
            if user_profile.files.count() >= 4:
                form.add_error('uploaded_files', 'You can only upload a maximum of 4 files.')
            else:
                UserFile.objects.create(user_profile=user_profile, uploaded_file=uploaded_file)
                return redirect('Dataspy:userprofile')

    return render(request, 'Dataspy/dashboard.html', {'form': form})


def get_ai_fix_suggestion(error_message):
    """
    Get a fix suggestion from OpenAI based on the error message.
    """
    prompt = f"Given the following error message, suggest Python code to fix it:\n\n{error_message}\n\nProvide only Python code to resolve the issue."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        code = response['choices'][0]['message']['content'].strip()
        return code
    except openai.error.RateLimitError:
        return "AI service quota exceeded. Please check OpenAI billing details."


def analyze_and_load(file_path, file_extension):
    """
    Tries to load the dataset based on the file extension, identifies issues,
    and dynamically generates AI code to fix them.
    """
    fix_attempts = 0
    solutions = []

    while fix_attempts < 5:
        fix_attempts += 1
        try:
            # Try to load the dataset based on extension
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
            return df, "Success"

        except Exception as e:
            error_message = str(e)
            solutions.append(f"Error detected: {error_message}")

            # Get the AI-generated fix suggestion for the error
            fix_code = get_ai_fix_suggestion(error_message)
            solutions.append(f"AI generated fix:\n{fix_code}")

            # Try to apply the AI fix dynamically
            try:
                exec(fix_code)  # Execute the generated code to apply the fix
                if file_extension == '.csv':
                    df = pd.read_csv(file_path)
                elif file_extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path, engine='openpyxl')
                return df, "Fix applied successfully"
            except Exception as fix_error:
                solutions.append(f"Failed after AI fix attempt: {str(fix_error)}")

    # Return the solutions if no fix works
    return None, solutions




def userprofile_view(request):
    """
    View to handle user profile, file upload, and AI suggestions.
    """
    # Get all files uploaded by the current user
    user_files = UserFile.objects.filter(user=request.user)
    form = DataUploadForm(request.POST or None, request.FILES or None)

    if request.method == 'POST' and form.is_valid():
        uploaded_file = form.cleaned_data.get('uploaded_files')
        if uploaded_file:
            # Check if the user has uploaded less than 4 files
            if user_files.count() >= 4:
                form.add_error('uploaded_files', 'You can only upload a maximum of 4 files.')
            else:
                # Create a new UserFile object for the uploaded file
                UserFile.objects.create(user=request.user, uploaded_file=uploaded_file)
                form.add_error('uploaded_files', 'File uploaded successfully.')

        if not form.errors:
            return redirect('Dataspy:userprofile')

    return render(request, 'Dataspy/userprofile.html', {
        'form': form,
        'user_files': user_files,
        # Add AI suggestions or other context here if needed
    })

def visualizations(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        uploaded_file = request.FILES['dataset']
        try:
            dataset = pd.read_csv(uploaded_file)
        except Exception as e:
            return JsonResponse({'error': f'Failed to load dataset: {str(e)}'}, status=400)

        request.session['dataset'] = dataset.to_json()
        return JsonResponse({'message': 'Dataset uploaded successfully. You can now visualize columns.'}, status=200)

    elif request.method == 'POST' and 'command' in request.POST:
        command = request.POST['command'].strip()
        dataset_json = request.session.get('dataset')

        if not dataset_json:
            return JsonResponse({'error': 'No dataset loaded'}, status=400)

        dataset = pd.read_json(dataset_json)

        if 'visualize' in command.lower():
            column_match = re.search(r"visualize the (.+) column", command.lower())
            if column_match:
                column_name = column_match.group(1).strip()

                if column_name not in dataset.columns:
                    return JsonResponse({'error': f'Column "{column_name}" not found in the dataset.'}, status=400)

                column_data = dataset[column_name]
                if column_data.dtype in ['int64', 'float64']:
                    fig = px.histogram(dataset, x=column_name, title=f'{column_name} Distribution')
                elif column_data.dtype == 'object':
                    fig = px.bar(column_data.value_counts().reset_index(), x='index', y=column_name,
                                 title=f'{column_name} Count')
                else:
                    return JsonResponse({'error': 'Unsupported column data type for visualization'}, status=400)

                chart_json = fig.to_json()
                return JsonResponse({'chart': chart_json}, status=200)

            return JsonResponse({'error': 'Could not interpret the command. Please specify a column to visualize.'}, status=400)

        return JsonResponse({'error': 'Unknown command. Please try again.'}, status=400)

    return render(request, 'Dataspy/visualizations.html')


def detect_file_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    """
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result.get('encoding', 'utf-8')
    
def try_reading_with_delimiters(file_path, encoding, error_messages):
    """
    Try reading the CSV file with detected or common delimiters.
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            sample = file.read(1024)
            detected_delimiter = ','  # Default to CSV comma delimiter
            df = pd.read_csv(file_path, encoding=encoding, delimiter=detected_delimiter)
            error_messages.append(f"File loaded successfully with detected delimiter: '{detected_delimiter}'.")
            return df
    except Exception as detection_error:
        error_messages.append(f"Delimiter detection failed: {detection_error}")

    delimiters = [',', ';;', '\t', '|',';']
    for delimiter in delimiters:
        try:
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
            error_messages.append(f"File loaded successfully with delimiter: '{delimiter}'.")
            return df
        except Exception as e:
            error_messages.append(f"Failed with delimiter '{delimiter}': {e}")
    return None



def home(request):
    return render(request,'Dataspy/home.html')

# Dashboard view

def insights_generation(request):
    return render(request, 'Dataspy/insights_generation.html')

def code_generation(request):
    return render(request, 'Dataspy/code_generation.html')

def reports(request):
    return render(request, 'Dataspy/reports.html')

def export_options(request):
    return render(request, 'Dataspy/export_options.html')

# Support views
def help_documentation(request):
    return render(request, 'Dataspy/help_documentation.html')

def feedback_community(request):
    return render(request, 'Dataspy/feedback_community.html')

# User settings view
def user_settings(request):
    return render(request, 'Dataspy/user_settings.html')

# Admin panel view
def admin_panel(request):
    return render(request, 'Dataspy/admin_panel.html')

def tryer(request):
    return render(request, 'Dataspy/tryer.html')

def clean_data(request):
    return render(request, 'Dataspy/datacleaning.html')



def detect_header_row(df):
    """
    Dynamically detect the header row of a dataset.
    Assumes the header row has the maximum number of unique string values.
    """
    for i in range(min(10, len(df))):  # Inspect the first 10 rows or fewer
        if df.iloc[i].nunique() > len(df.columns) * 0.7:  # Heuristic for header detection
            return i
    return 0  # Default to the first row if no clear header row is detected



def execute_generated_code(df, code):
    """Helper function to execute generated Python code safely."""
    local_vars = {"df": df, "pd": pd}

    # Execute the code
    exec(code, {}, local_vars)

    # Dynamically capture updated DataFrame or other variables
    if "df" in local_vars and isinstance(local_vars["df"], pd.DataFrame):
        return local_vars["df"]

    # Return any other variables if necessary
    return None


def generate_plotly_table(df):
    """Helper function to generate a Plotly table for a DataFrame."""
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left')
    )])
    return fig.to_html(full_html=False)


from django.shortcuts import render

import plotly.io as pio

def execute_generated_plotly_code(df, generated_code):
    """Executes Plotly code and returns the rendered HTML for the chart."""
    try:
        # Restricted execution environment
        restricted_globals = {
            'df': df,  # Pass the DataFrame
            'pd': pd,  # Allow Pandas usage
            'plotly': __import__('plotly'),  # Allow Plotly usage
        }
        exec(generated_code, restricted_globals)

        # Look for any figure created in the executed code
        for var_name, var_value in restricted_globals.items():
            if isinstance(var_value, (restricted_globals['plotly'].graph_objs.Figure, )):
                # Convert Plotly figure to HTML
                return pio.to_html(var_value, full_html=False)
        raise Exception("No Plotly figure found in the executed code.")
    except Exception as e:
        raise Exception(f"Plot execution failed: {str(e)}")


def execute_generated_plotly_code(df, generated_code):
    """Executes Plotly code and returns the rendered HTML for the chart."""
    try:
        # Restricted execution environment
        restricted_globals = {
            'df': df,  # Pass the DataFrame
            'pd': pd,  # Allow Pandas usage
            'plotly': __import__('plotly'),  # Allow Plotly usage
        }
        exec(generated_code, restricted_globals)

        # Look for any figure created in the executed code
        for var_name, var_value in restricted_globals.items():
            if isinstance(var_value, (restricted_globals['plotly'].graph_objs.Figure, )):
                # Convert Plotly figure to HTML
                return pio.to_html(var_value, full_html=False)
        raise Exception("No Plotly figure found in the executed code.")
    except Exception as e:
        raise Exception(f"Plot execution failed: {str(e)}")


def generate_plotly_table(df):
    """Generates a Plotly table from the DataFrame."""
    df_clean =df
    fig = go.Figure(data=[go.Table(
        header=dict(values=df_clean.columns.tolist()),  # Set the header as column names
        cells=dict(values=[df_clean[col].values.tolist() for col in df_clean.columns])  # Convert each column to a list
    )])
    # Return the figure as HTML
    return fig.to_html()






def generate_dynamic_schema(dataframe):
    """
    Generate a Pandera schema dynamically based on the dataframe.
    """
    column_types = {
        np.dtype('int64'): Column(int, nullable=True),
        np.dtype('float64'): Column(float, nullable=True),
        np.dtype('object'): Column(str, nullable=True),
        np.dtype('datetime64[ns]'): Column(pd.Timestamp, nullable=True),
    }

    schema_dict = {
        col: column_types.get(dtype, Column(str, nullable=True))  # Default to string if dtype is unknown
        for col, dtype in dataframe.dtypes.items()
    }

    return DataFrameSchema(schema_dict, coerce=True, strict=False)



from .models import UserProfile
from django.contrib.auth import get_user_model

def return_dataframe(user):
    """
    Load a dataframe from the most recently uploaded file for a specific user.

    Args:
        user (User): A Django User object.

    Returns:
        tuple: A pandas DataFrame and the file path of the uploaded file.

    Raises:
        ValueError: If there is an error loading the file or unsupported file format.
    """
    try:
        # Fetch the most recent file uploaded by the user
        user_file = UserFile.objects.filter(user=user).latest('date_uploaded')
        print(user_file)

        # Get the file path
        file_path = user_file.uploaded_file.path
        print(file_path)

        # Determine the file extension
        file_extension = file_path.split('.')[-1].lower()

        # Load the file into a DataFrame
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file_path, engine='openpyxl')
        elif file_extension == 'csv':
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')
        elif file_extension in ['sav', 'zsav']:
            result = pyreadstat.read_sav(file_path)
            df = result[0] if isinstance(result, tuple) else result
        elif file_extension in ['dta']:
            result = pyreadstat.read_dta(file_path)
            df = result[0] if isinstance(result, tuple) else result
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Check if the dataframe is empty
        if df.empty or len(df.columns) == 0:
            raise ValueError("The dataset has no rows or columns.")

        # Clean up column names
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns.tolist()]

        return df, file_path  # Ensure only the dataframe and file path are returned

    except UserFile.DoesNotExist:
        raise ValueError("No uploaded files found for the user.")
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")

def fetch_user_messages(request):
    """Helper function to retrieve user messages."""
    if request.user.is_authenticated:
        # Filter the messages by the authenticated user
        return ChatMessage.objects.all()
def fetch_user_errors(request):
    """Helper function to retrieve error logs."""
    return ErrorLog.objects.all().order_by('timestamp')



def generate_chatgpt_prompt(df, file_path, user_prompt):
    """Helper function to generate the ChatGPT prompt."""
    columns_list = df.columns[:min(10, len(df.columns))].tolist()
    return f"""
    You are a data assistant specialized in Python-based data analysis. Provide concise, actionable responses based on the dataset and user query.

    Dataset:
    - File: {file_path}
    - Columns: {columns_list}
    - Types: {df.dtypes.to_dict()}
    - Sample: {df.head(min(3, len(df))).to_dict(orient='records')}

    Query: '{user_prompt}'

    Compulsory Instructions :
    -1. For code or analysis tasks, respond **only with executable Python code**.
    -2. For guidance or insights, respond with **concise HTML-formatted text** (no code unless requested).
    -3. Use **Plotly only** for visualizations (ie, plot,chat,graph ,etc); avoid Matplotlib or other libraries.
    -4. Ensure Plotly graphs do not open in a new tab using: **`pio.to_html(fig, full_html=False, config={{'displayModeBar': False, 'autoOpen': False}})`**
    """


from django.db import transaction

def handle_error(error_message, generated_code, generated_response, user_message=None):
    print(generated_response,generated_response)

    """Log errors and create an ErrorLog entry."""
    logger.error(error_message)  # Log error to the system log
    if user_message and isinstance(user_message, ChatMessage):
        try:
            with transaction.atomic():
                ErrorLog.objects.create(chat_message=user_message, error_message=error_message)
        except Exception as e:
            logger.error(f"Error saving ErrorLog: {str(e)}")
    else:
        logger.error("Invalid user_message, cannot create ErrorLog.")




def process_generated_code(df, generated_response):
    """Helper function to execute the code generated by ChatGPT."""
    try:
        updated_df = execute_generated_code(df, generated_response)
        return updated_df
    except Exception as e:
        handle_error(f"Code execution error: {str(e)}")
        raise


def get_cached_response(cache_key):
    """Try to get a cached response for the given cache key."""
    return cache.get(cache_key)

def set_cache_response(cache_key, data, timeout=60*15):
    """Set the response data in cache."""
    cache.set(cache_key, data, timeout)




#this is the real view
def data_query(request):
    """Main view for handling data analysis queries."""
   # chat_messages = fetch_user_messages(request)
    chat_messages = ChatMessage.objects.all().order_by('timestamp')[:10]
    usererrors = fetch_user_errors(request)
    user_message=None
    messages = []  # This is your custom list


    last_plot_message = ChatMessage.objects.filter(
        Q(user=request.user) & ~Q(plotly__isnull=True)
    ).order_by('-timestamp').first()

    messages = [
        {
            "user": chat_message.user.username if chat_message.user else "Anonymous",
            "user_request": chat_message.user_request,
            "code": chat_message.code,
            "ai_response": chat_message.ai_response,
            "timestamp": chat_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format the timestamp
        }
        for chat_message in chat_messages
    ]

    # Pass data to the template
    context = {"messages": messages,'last_plot_message':last_plot_message}


    user_message = None
    generated_response = None
    error_message = None

    if request.method == "POST":
        if request.user.is_authenticated:
            user = request.user
        try:
            user_prompt = request.POST.get('prompt')



            if not user_prompt:
                context['messages'].append({'user': 'user', 'text': 'No prompt provided.'})
                return render(request, 'Dataspy/data_query1.html', context)

            # Add the user prompt to the messages list for chat history
            context['messages'].append({'user': 'user', 'text': user_prompt})


            # Process the query asynchronously using Celery
            task = process_user_query.delay(user_prompt,request.user.id)
            # Send immediate response indicating the task has been queued
            context['messages'].append({'user': 'chatgpt', 'text': 'Your query is being processed in the background. Please wait...'})
            context['task_id'] = task.id  # Add task ID to context
            return render(request, 'Dataspy/data_query1.html', context)

        except Exception as e:
            context['messages'].append({'user': 'chatgpt', 'text': error_message})

    return render(request, 'Dataspy/data_query1.html', context)



def auto_updata(request):
    user_message = None
    # Check if the user is authenticated
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'User not authenticated', 'messages': []})

    # Retrieve all messages for the authenticated user
    messages = ChatMessage.objects.filter(user=request.user)
    last_message = messages.last()
    # If there's a last message, retrieve its user_request
    if last_message:
        user_message = last_message.user_request
    # Return the JSON response with the list of messages and the user's input message
    return JsonResponse({
        'messages': list(messages.values('user_request', 'ai_response', 'timestamp','code','plotly','matplot')),  # Adjust fields as needed
        'user_message': user_message
    })



def task_status(request, task_id):
    if isinstance(task_id, UUID):
        task_id = str(task_id)  # Convert UUID to string
    logging.debug(f"Task ID: {task_id}")
    task = AsyncResult(task_id)

    if task.ready():  # Check if the task has completed
        result = task.result  # Retrieve the result if the task is finished
        return JsonResponse({'status': 'SUCCESS', 'result': result})
    else:
        # If the task is still pending, return a pending status
        return JsonResponse({'status': 'PENDING'})


def custom_analysis(request):
    return render(request, 'Dataspy/custom_analysis.html')

def detect_file_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    """
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result.get('encoding', 'utf-8')  # Default to utf-8 if detection fails

def get_ai_fix_suggestion(error_message):
    """
    Get a fix suggestion from OpenAI based on the error message.
    """
    prompt = f"Given the following error message, response with only Python code and no comment in code to fix it:\n\n{error_message}\n\n Again ,Provide only Python code to resolve the issue!"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        code = response['choices'][0]['message']['content'].strip()
        return code
    except openai.error.RateLimitError:
        return "AI service quota exceeded. Please check OpenAI billing details."




def try_reading_with_delimiters(file_path, encoding, error_messages):
    """
    Try reading the CSV file with detected or common delimiters.
    """
    detected_delimiter = None

    # Step 1: Attempt to detect delimiter using csv.Sniffer
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            sample = file.read(1024)  # Read a small portion of the file
            detected_delimiter = csv.Sniffer().sniff(sample).delimiter
            if len(detected_delimiter) != 1:
                raise ValueError("Detected delimiter is invalid.")
            df = pd.read_csv(file_path, encoding=encoding, delimiter=detected_delimiter)
            error_messages.append(f"File loaded successfully with detected delimiter: '{detected_delimiter}'.")
            return df, detected_delimiter
    except Exception as detection_error:
        error_messages.append(f"Delimiter detection failed: {detection_error}")
        detected_delimiter = None

    # Step 2: Fall back to common delimiters if detection fails
    delimiters = [',', ';', '\t', '|']  # Common delimiters
    for delimiter in delimiters:
        try:
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
            error_messages.append(f"File loaded successfully with fallback delimiter: '{delimiter}'.")
            return df, delimiter
        except Exception as e:
            error_messages.append(f"Failed with delimiter '{delimiter}': {e}")

    # If all attempts fail, raise an error
    raise ValueError("Failed to read the file with all tested delimiters.")



def get_dataset_summary(df):
    """
    Generate a summary of the dataset, including basic statistics.
    """
    summary = {}
    summary['shape'] = df.shape  # Number of rows and columns
    summary['info'] = df.info()  # Basic info like datatypes and non-null counts
    summary['head'] = df.head(5).to_dict(orient='records')  # First 5 rows of data
    summary['statistics'] = df.describe().to_dict()  # Basic statistics for numerical columns
    summary['missing_values'] = df.isnull().sum().to_dict()  # Count of missing values in each column
    return summary
def detect_file_encoding(file_path):
    """
    Detect the encoding of a file using the chardet library.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read a small portion of the file
    result = chardet.detect(raw_data)
    return result['encoding']

def get_ai_summary(dataset_summary):
    """
    Generate a text summary of the dataset using OpenAI's GPT model.
    """
    summary_prompt = (
        f"Provide a concise summary of the following dataset details in less than 100 tokens:\n"
        f"Shape: {dataset_summary['shape']}\n"
        f"Missing Values: {dataset_summary['missing_values']}\n"
    )

    statistics = dataset_summary.get('statistics', {})
    if statistics:
        summary_prompt += "Basic Statistics:\n"
        for column, stats in statistics.items():
            summary_prompt += (
                f"{column}: Mean: {stats.get('mean')}, Std Dev: {stats.get('std')}, "
                f"Min: {stats.get('min')}, Median: {stats.get('percent_50')}, Max: {stats.get('max')}\n"
            )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes datasets."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        return f"AI summary generation failed: {str(e)}"

def convert_datetime_to_string(obj):
    """
    Recursively converts datetime objects in a dictionary or list to string format.
    """
    if isinstance(obj, dict):
        return {key: convert_datetime_to_string(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_string(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj



def display_spread_sheet(request):
    # Get all user files for the logged-in user
    user_files = UserFile.objects.filter(user=request.user)
    error_messages = []
    dataset_summary = {'statistics': {}, 'missing_values': {}, 'shape': ''}
    table_json = None
    detected_delimiter = None

    if user_files.exists():
        try:
            # Get the latest file uploaded
            latest_file = user_files.latest('date_uploaded')
            file_path = latest_file.uploaded_file.path
            file_extension = os.path.splitext(file_path)[1].lower()

            # Check for supported file formats
            if file_extension not in ['.xlsx', '.xls', '.csv']:
                raise ValueError("Unsupported file format. Please upload an Excel or CSV file.")

            # Load the file into a DataFrame
            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_extension == '.csv':
                detected_encoding = detect_file_encoding(file_path)
                try:
                    # Try reading the file with multiple delimiters
                    df, detected_delimiter = try_reading_with_delimiters(file_path, detected_encoding, error_messages)
                    if df is None:
                        raise ValueError("Could not determine delimiter automatically.")
                except Exception as e:
                    # Fallback: Try reading with a comma delimiter if delimiter detection fails
                    error_messages.append(f"Delimiter detection failed: {str(e)}. Attempting default delimiter (comma).")
                    try:
                        df = pd.read_csv(file_path, encoding=detected_encoding, delimiter=',')
                        detected_delimiter = ','
                    except Exception as e:
                        raise ValueError(f"Error loading CSV file: {str(e)}")

            # If DataFrame is loaded successfully
            if df is not None:
                save_delimiter = detected_delimiter or ','
                updated_file_path = os.path.splitext(file_path)[0] + "_updated.csv"
                df.to_csv(updated_file_path, index=False, sep=save_delimiter)

                # Delete the original file before updating
                if os.path.exists(file_path):
                    os.remove(file_path)

                # Update the UserFile model with the new file path
                latest_file.uploaded_file = updated_file_path
                latest_file.save()
                df = df.where(pd.notnull(df), None)
                table_json = df.to_dict(orient='records')
                dataset_summary['shape'] = df.shape
                dataset_summary['missing_values'] = df.isnull().sum().to_dict()
                dataset_summary['statistics'] = df.describe(include='all').to_dict()

        except Exception as e:
            error_messages.append(str(e))
    else:
        error_messages.append("No files found for the user.")

    # Initialize ai_summary
    ai_summary = None

    # Check if the "Generate Summary" button was clicked (POST request)
    if request.method == "POST" and 'generate_summary' in request.POST:
        # Generate AI summary only if the button was clicked
        ai_summary = get_ai_summary(dataset_summary)

    # Convert datetime objects to strings (if any)
    dataset_summary = convert_datetime_to_string(dataset_summary)
    if table_json:
        table_json = json.dumps(table_json)

    return render(request, 'Dataspy/display_spread_sheet.html', {
        'user_files': user_files,
        'table_json': json.dumps(table_json) if table_json else None,
        'error_messages': error_messages,
        'ai_summary': ai_summary,
        'dataset_summary': dataset_summary,
    })

def del_file(request, id):
    if request.method == 'POST':
        file = get_object_or_404(UserFile, id=id,user=request.user)
        file.delete()
        return redirect('Dataspy:userprofile')
    return HttpResponseForbidden("Invalid request method.")