# Standard library imports
import re
import io
import base64

# Third-party libraries
import openai
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import mpld3
import plotly.io as pio

# Django imports
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth import get_user_model
from django.core.serializers import serialize

# Local application imports
from .models import UserFile, ChatMessage

# Celery task decorators
from celery import shared_task

# Ensure that this import is made only when required to avoid circular imports
from .views import *

@shared_task()
def process_user_query(user_prompt, user_id):
    from .views import return_dataframe, generate_chatgpt_prompt
    """Background task to handle data query processing"""
    context = {'messages': []}
    fig_html = None
    plot_data = None
    plot_html = None  # Initialize plot_html to avoid uninitialized variable error

    try:
        # Retrieve the user object using the user_id
        user = get_user_model().objects.get(id=user_id)

        # Process uploaded file and generate ChatGPT response
        df, file_path = return_dataframe(user)  # Pass the 'user' object directly
        if df is None or df.empty or len(df.columns) == 0:
            raise ValueError("Invalid dataset. Please check the uploaded file.")

        # Generate a prompt for ChatGPT based on the dataframe and user prompt
        chatgpt_prompt = generate_chatgpt_prompt(df, file_path, user_prompt)

        # Make a request to OpenAI's API to get the AI-generated response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": chatgpt_prompt},
                {"role": "user", "content": "You are a friendly data analyst assistant. Always respond in the language of the user's input. Do not response in diffrent langange other than the user input language. Do not engage with requests that have malicious intent or violate ethical guidelines"}

            ]
        )
        generated_response = response["choices"][0]["message"]["content"].strip()

        # Handle code extraction and execution from response
        if generated_response.startswith("```python") and generated_response.endswith("```"):
            generated_response_code = generated_response.strip("```python").strip("```").strip()
        else:
            code_pattern = r"```python(.*?)```"
            python_code_matches = re.findall(code_pattern, generated_response, flags=re.DOTALL)
            generated_response_code = python_code_matches[0].strip() if python_code_matches else None

        if generated_response_code:
            generated_response_code = fix_file_paths(generated_response_code)
            exec_namespace = {}

            try:
                # Execute the code and capture the output
                exec(generated_response_code, {}, exec_namespace)

                # Handle Matplotlib plot
                if 'plt' in exec_namespace and isinstance(plt, type):
                    plot_html = mpld3.fig_to_html(plt.gcf())  # Convert current figure to HTML
                    context['plot_html'] = plot_html
                    print("Matplotlib plot saved as HTML")
                    plt.close()

                # Handle Plotly plot
                if 'fig' in exec_namespace:
                    fig = exec_namespace['fig']
                    # Ensure 'fig' is a valid Plotly Figure object
                    if isinstance(fig, go.Figure):
                        # Convert the figure to HTML
                        plot_html = pio.to_html(fig, full_html=False,
                                                config={'displayModeBar': False, 'autoOpen': False})
                        context['plot_html'] = plot_html



            except Exception as e:
                context['error'] = f"Error executing Python code: {e}"
                print(f"Execution error: {e}")

        # Save the user message in ChatMessage
        ChatMessage.objects.create(
            user_request=user_prompt,
            user=user,
            ai_response=generated_response,
            code=generated_response_code,
            plotly=plot_html  # plot_html is always initialized

        )


        context['messages'].append({'user': 'chatgpt', 'text': generated_response})

    except Exception as e:
        context['error'] = str(e)
        raise ValueError(f"Error processing user query: {str(e)}")

    return context



def fix_file_paths(code_block):
    """Fix Windows-style file paths to prevent unicode escape errors"""
    # Use a regex to find any file paths and replace backslashes with forward slashes
    code_block = re.sub(r'([A-Za-z]:\\[^\n]*)', lambda x: x.group(0).replace('\\', '/'), code_block)
    return code_block
