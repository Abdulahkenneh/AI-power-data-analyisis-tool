
from django.core.serializers import serialize
from django.contrib.auth import get_user_model
from celery import shared_task
import openai
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt  # Ensure this import is here for plotting
from .views import return_dataframe, generate_chatgpt_prompt
from django.core.exceptions import ObjectDoesNotExist

import base64
import io
import openai
import matplotlib.pyplot as plt
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth import get_user_model
from .models import UserFile
from .views import *
from celery import shared_task
import pandas as pd

from django.core.exceptions import ObjectDoesNotExist
from .models import ChatMessage  # Ensure the correct import for your model
import openai
import io
import base64
import matplotlib.pyplot as plt

import plotly

import plotly

#
# #
# #
# # @shared_task()
# # def process_user_query(user_prompt, user_id):
# #     """Background task to handle data query processing"""
# #     generated_response_code = None
# #
# #     try:
# #         # Fetch user messages and errors (replace with actual functions)
# #         context = {'messages': []}
# #
# #         # Retrieve the user object using the user_id
# #         try:
# #             user = get_user_model().objects.get(id=user_id)
# #         except ObjectDoesNotExist:
# #             raise ValueError(f"User with ID {user_id} does not exist.")
# #
# #         # Process uploaded file and generate ChatGPT response
# #         df, file_path = return_dataframe(user)  # Pass the 'user' object directly
# #         if df is None or df.empty or len(df.columns) == 0:
# #             raise ValueError("Invalid dataset. Please check the uploaded file.")
# #
# #         # Generate a prompt for ChatGPT based on the dataframe and user prompt
# #         chatgpt_prompt = generate_chatgpt_prompt(df, file_path, user_prompt)
# #
# #         # Make a request to OpenAI's API to get the AI-generated response
# #         response = openai.ChatCompletion.create(
# #             model="gpt-4",
# #             messages=[{"role": "user", "content": chatgpt_prompt}]
# #         )
# #         generated_response = response["choices"][0]["message"]["content"].strip()
# #
# #         # Initialize generated_response_code to None
# #         if generated_response.startswith("```python") and generated_response.endswith("```"):
# #             generated_response_code = generated_response.strip("```python").strip("```").strip()
# #             exec_namespace = {}
# #
# #             # Execute the code and capture the output
# #             exec(generated_response_code, {}, exec_namespace)
# #
# #             # If a plot is generated, encode it to base64 for rendering
# #             if 'plt' in exec_namespace:
# #                 buf = io.BytesIO()
# #                 plt.savefig(buf, format='png')  # Ensure plt is defined here
# #                 buf.seek(0)
# #                 plot_data = base64.b64encode(buf.read()).decode('utf-8')
# #                 buf.close()
# #                 plt.close()
# #
# #                 context['plot_image_base64'] = plot_data  # Add plot to context
# #
# #             # If the 'df' variable is modified in the executed code, update context
# #             updated_df = exec_namespace.get('df', None)
# #             if updated_df is not None:
# #                 # Ensure the dataframe is serializable (convert to HTML)
# #                 context['dataframe'] = updated_df.to_html(classes="table table-striped")
# #
# #         # Save the user message in ChatMessage
# #         ChatMessage.objects.create(
# #             user_request=user_prompt,
# #             user=user,
# #             ai_response=generated_response,
# #             code=generated_response_code  # Only set if it's not None
# #         )
# #
# #         # Append the generated response to the messages context
# #         context['messages'].append({'user': 'chatgpt', 'text': generated_response})
# #
# #         # If a dataframe exists, convert it to HTML (this is already handled above)
# #         if 'dataframe' in context:
# #             context['dataframe'] = context['dataframe']  # Ensure it's HTML format
# #
# #     except Exception as e:
# #         # Log the error and handle it appropriately
# #         context['error'] = str(e)
# #         raise ValueError(f"Error processing user query: {str(e)}")
#
#
#
#
#
#
#
#
#
#
#
#
# #
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.preprocessing import StandardScaler
# # import sklearn
# # from sklearn.metrics import accuracy_score
# # from sklearn.model_selection import train_test_split
# # from django.core.serializers import serialize
# # from django.contrib.auth import get_user_model
# # from celery import shared_task
# # import openai
# # import io
# # import base64
# # import pandas as pd
# # import matplotlib.pyplot as plt  # Ensure this import is here for plotting
# # from .views import return_dataframe, generate_chatgpt_prompt
# # from django.core.exceptions import ObjectDoesNotExist
# # import re
# # from .models import ChatMessage  # Ensure the correct import for your model
# #
# # @shared_task()
# # def process_user_query(user_prompt, user_id):
# #     """Background task to handle data query processing"""
# #     generated_response_code = None
# #     context = {'messages': []}
# #
# #     try:
# #         # Retrieve the user object using the user_id
# #         try:
# #             user = get_user_model().objects.get(id=user_id)
# #         except ObjectDoesNotExist:
# #             raise ValueError(f"User with ID {user_id} does not exist.")
# #
# #         # Process uploaded file and generate ChatGPT response
# #         df, file_path = return_dataframe(user)  # Pass the 'user' object directly
# #         if df is None or df.empty or len(df.columns) == 0:
# #             raise ValueError("Invalid dataset. Please check the uploaded file.")
# #
# #         # Generate a prompt for ChatGPT based on the dataframe and user prompt
# #         chatgpt_prompt = generate_chatgpt_prompt(df, file_path, user_prompt)
# #
# #         # Make a request to OpenAI's API to get the AI-generated response
# #         response = openai.ChatCompletion.create(
# #             model="gpt-4",
# #             messages=[{"role": "user", "content": chatgpt_prompt}]
# #         )
# #         generated_response = response["choices"][0]["message"]["content"].strip()
# #
# #         # Enhanced handling for extracting Python code
# #         if generated_response.startswith("```python") and generated_response.endswith("```"):
# #             # Extract code if it's the whole response (starts and ends with Python code block)
# #             generated_response_code = generated_response.strip("```python").strip("```").strip()
# #         else:
# #             # Use regex to find Python code anywhere in the response
# #             code_pattern = r"```python(.*?)```"
# #             python_code_matches = re.findall(code_pattern, generated_response, flags=re.DOTALL)
# #
# #             if python_code_matches:
# #                 # Flatten the list (since re.findall returns a list of matches) and strip extra spaces
# #                 generated_response_code = python_code_matches[0].strip()
# #
# #         # If we have extracted Python code, execute it
# #         if generated_response_code:
# #             exec_namespace = {}
# #
# #             try:
# #                 # Execute the code and capture the output
# #                 exec(generated_response_code, {}, exec_namespace)
# #
# #                 # If a plot is generated, encode it to base64 for rendering
# #                 if 'plt' in exec_namespace:
# #                     buf = io.BytesIO()
# #                     plt.savefig(buf, format='png')  # Ensure plt is defined here
# #                     buf.seek(0)
# #                     plot_data = base64.b64encode(buf.read()).decode('utf-8')
# #                     buf.close()
# #                     plt.close()
# #
# #                     context['plot_image_base64'] = plot_data  # Add plot to context
# #
# #                 # If the 'df' variable is modified in the executed code, update context
# #                 updated_df = exec_namespace.get('df', None)
# #                 if updated_df is not None:
# #                     # Ensure the dataframe is serializable (convert to HTML)
# #                     context['dataframe'] = updated_df.to_html(classes="table table-striped")
# #
# #             except Exception as e:
# #                 # Handle errors during code execution
# #                 context['error'] = f"Error executing Python code: {e}"
# #                 raise ValueError(f"Error executing Python code: {e}")
# #
# #         # Save the user message in ChatMessage
# #         ChatMessage.objects.create(
# #             user_request=user_prompt,
# #             user=user,
# #             ai_response=generated_response,
# #             code=generated_response_code  # Only set if it's not None
# #         )
# #
# #         # Append the generated response to the messages context
# #         context['messages'].append({'user': 'chatgpt', 'text': generated_response})
# #
# #         # If a dataframe exists, convert it to HTML (this is already handled above)
# #         if 'dataframe' in context:
# #             context['dataframe'] = context['dataframe']  # Ensure it's HTML format
# #
# #     except Exception as e:
# #         # Log the error and handle it appropriately
# #         context['error'] = str(e)
# #         raise ValueError(f"Error processing user query: {str(e)}")
#
# from celery import shared_task
# import plotly
# import openai
# import io
# import base64
# import matplotlib.pyplot as plt
# from django.contrib.auth import get_user_model
# from .models import ChatMessage
# from .views import return_dataframe, generate_chatgpt_prompt
# import re
# import plotly.io as pio
# import mpld3
# pio.renderers.default = "json"
# # @shared_task()
# # def process_user_query(user_prompt, user_id):
# #     """Background task to handle data query processing"""
# #     context = {'messages': []}
# #     fig_html = None
# #     plot_data = None
# #     plot_html = None
# #
# #     try:
# #         # Retrieve the user object using the user_id
# #         user = get_user_model().objects.get(id=user_id)
# #
# #         # Process uploaded file and generate ChatGPT response
# #         df, file_path = return_dataframe(user)  # Pass the 'user' object directly
# #         if df is None or df.empty or len(df.columns) == 0:
# #             raise ValueError("Invalid dataset. Please check the uploaded file.")
# #
# #         # Generate a prompt for ChatGPT based on the dataframe and user prompt
# #         chatgpt_prompt = generate_chatgpt_prompt(df, file_path, user_prompt)
# #
# #         # Make a request to OpenAI's API to get the AI-generated response
# #         response = openai.ChatCompletion.create(
# #             model="gpt-4",
# #             messages=[{"role": "user", "content": chatgpt_prompt}]
# #         )
# #         generated_response = response["choices"][0]["message"]["content"].strip()
# #
# #         # Handle code extraction and execution from response
# #         if generated_response.startswith("```python") and generated_response.endswith("```"):
# #             generated_response_code = generated_response.strip("```python").strip("```").strip()
# #         else:
# #             code_pattern = r"```python(.*?)```"
# #             python_code_matches = re.findall(code_pattern, generated_response, flags=re.DOTALL)
# #             generated_response_code = python_code_matches[0].strip() if python_code_matches else None
# #
# #         if generated_response_code:
# #             generated_response_code = fix_file_paths(generated_response_code)
# #             exec_namespace = {}
# #
# #             try:
# #                 pio.renderers.default = "json"
# #                 # Execute the code and capture the output
# #                 exec(generated_response_code, {}, exec_namespace)
# #
# #                 # Handle Matplotlib plot
# #                 if 'plt' in exec_namespace and isinstance(plt, type):
# #                     plot_html = mpld3.fig_to_html(plt.gcf())  # Convert current figure to HTML
# #                     context['plot_html'] = plot_html
# #                     print("Matplotlib plot saved as HTML")
# #                     plt.close()
# #
# #                 # Handle Plotly plot
# #                 elif 'fig' in exec_namespace and isinstance(exec_namespace['fig'], plotly.graph_objs._figure.Figure):
# #                     fig = exec_namespace['fig']
# #                     plot_html = pio.to_html(fig, full_html=False, config={'displayModeBar': False, 'autoOpen': False})
# #                     context['plot_html'] = plot_html
# #                     print("Plotly plot saved as HTML")
# #
# #             except Exception as e:
# #                 context['error'] = f"Error executing Python code: {e}"
# #                 print(f"Execution error: {e}")
# #
# #         # Save the user message in ChatMessage
# #         ChatMessage.objects.create(
# #             user_request=user_prompt,
# #             user=user,
# #             ai_response=generated_response,
# #             code=generated_response_code,
# #             plotly=plot_html  # plot_html is always initialized
# #         )
# #         context['messages'].append({'user': 'chatgpt', 'text': generated_response})
# #
# #     except Exception as e:
# #         context['error'] = str(e)
# #         raise ValueError(f"Error processing user query: {str(e)}")
# #
# #     return context
#
#
#
#
#
#
#
#
#
# from django.contrib.auth import get_user_model
# from .models import ChatMessage
# from .views import return_dataframe, generate_chatgpt_prompt
# import openai
# import re
# import plotly
# import mpld3
# import plotly.io as pio
#
# @shared_task()
# def process_user_query(user_prompt, user_id):
#     """Background task to handle data query processing"""
#     context = {'messages': []}
#     plot_html = None
#
#     try:
#         # Retrieve the user object using the user_id
#         user = get_user_model().objects.get(id=user_id)
#
#         # Process uploaded file and generate ChatGPT response
#         df, file_path = return_dataframe(user)  # Pass the 'user' object directly
#         if df is None or df.empty or len(df.columns) == 0:
#             raise ValueError("Invalid dataset. Please check the uploaded file.")
#
#         # Generate a prompt for ChatGPT based on the dataframe and user prompt
#         chatgpt_prompt = generate_chatgpt_prompt(df, file_path, user_prompt)
#
#         # Make a request to OpenAI's API to get the AI-generated response
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": chatgpt_prompt}]
#         )
#         generated_response = response["choices"][0]["message"]["content"].strip()
#
#         # Extract Python code from the response
#         if generated_response.startswith("```python") and generated_response.endswith("```"):
#             generated_response_code = generated_response.strip("```python").strip("```").strip()
#         else:
#             code_pattern = r"```python(.*?)```"
#             python_code_matches = re.findall(code_pattern, generated_response, flags=re.DOTALL)
#             generated_response_code = python_code_matches[0].strip() if python_code_matches else None
#
#         if generated_response_code:
#             generated_response_code = fix_file_paths(generated_response_code)
#
#             # Execute the code using the subprocess function
#             execution_result = execute_python_code(generated_response_code)
#
#             if execution_result['status'] == 'SUCCESS':
#                 # Handle Matplotlib plot (if applicable)
#                 if '<div class="mpld3"' in execution_result['output']:
#                     plot_html = execution_result['output']
#                     context['plot_html'] = plot_html
#                     print("Matplotlib plot processed")
#
#                 # Handle Plotly plot (if applicable)
#                 elif 'plotly' in execution_result['output']:
#                     plot_html = execution_result['output']
#                     context['plot_html'] = plot_html
#                     print("Plotly plot processed")
#
#                 else:
#                     # Handle other text outputs
#                     context['messages'].append({'user': 'chatgpt', 'text': execution_result['output']})
#             else:
#                 context['error'] = execution_result['error']
#                 print(f"Code execution error: {execution_result['error']}")
#
#         # Save the user message in ChatMessage
#         ChatMessage.objects.create(
#             user_request=user_prompt,
#             user=user,
#             ai_response=generated_response,
#             code=generated_response_code,
#             plotly=plot_html  # plot_html is always initialized
#         )
#         context['messages'].append({'user': 'chatgpt', 'text': generated_response})
#
#     except Exception as e:
#         context['error'] = str(e)
#         raise ValueError(f"Error processing user query: {str(e)}")
#
#     return context
#
#
# def fix_file_paths(code_block):
#     """Fix Windows-style file paths to prevent unicode escape errors"""
#     # Use a regex to find any file paths and replace backslashes with forward slashes
#     code_block = re.sub(r'([A-Za-z]:\\[^\n]*)', lambda x: x.group(0).replace('\\', '/'), code_block)
#     return code_block
#
#
# import subprocess
#
#
# def execute_python_code(input_code):
#     """
#     Executes Python code using subprocess and returns the result.
#
#     Args:
#         input_code (str): The Python code to execute.
#
#     Returns:
#         dict: A dictionary with 'output', 'error', and 'status'.
#     """
#     try:
#         # Run the Python code in a subprocess
#         result = subprocess.run(
#             ['python3', '-c', input_code],  # Executes the Python code passed as a string
#             stdout=subprocess.PIPE,  # Capture standard output
#             stderr=subprocess.PIPE,  # Capture standard error
#             text=True  # Return output as a string
#         )
#
#         # Check for errors in stderr and return the appropriate result
#         if result.stderr:
#             return {
#                 'output': None,
#                 'error': result.stderr,  # Return the error if there is any
#                 'status': 'FAILED'
#             }
#         else:
#             return {
#                 'output': result.stdout,  # Return the successful output
#                 'error': None,
#                 'status': 'SUCCESS'
#             }
#     except Exception as e:
#         # Return any exception that occurs during the subprocess execution
#         return {
#             'output': None,
#             'error': str(e),  # Return exception message as error
#             'status': 'ERROR'
#         }





import re
import openai
import mpld3
import plotly.io as pio
import matplotlib.pyplot as plt
from django.contrib.auth import get_user_model
from .models import ChatMessage

@shared_task()
def process_user_query(user_prompt, user_id):
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
