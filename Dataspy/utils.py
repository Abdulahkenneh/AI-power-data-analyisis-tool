# utils.py

import pandas as pd
import plotly.graph_objects as go


def execute_generated_code(df, generated_response):
    """
    Executes the code generated from the user's prompt and applies transformations to the DataFrame.

    Args:
        df (pandas.DataFrame): The original data to be processed.
        generated_response (str): The code (e.g., Python or DataFrame transformation logic) to execute on the DataFrame.

    Returns:
        pandas.DataFrame: The updated DataFrame after applying the generated code.
    """
    try:
        # Execute the generated code on the dataframe (for safety, we might want to limit what code is allowed to run)
        # Use `exec` cautiously, ideally with proper validation and sandboxing
        # For now, assuming generated_response is a valid Python expression for DataFrame manipulation
        exec(generated_response)  # exec is used to execute the code dynamically
        return df
    except Exception as e:
        # Log or raise the error if something goes wrong
        raise ValueError(f"Error executing generated code: {str(e)}")


def generate_plotly_table(df):
    """
    Generates a Plotly table from a pandas DataFrame for visualization.

    Args:
        df (pandas.DataFrame): The processed DataFrame to be displayed as a table.

    Returns:
        plotly.graph_objects.Figure: The Plotly table figure.
    """
    # Generate a Plotly table from the DataFrame
    table = go.Figure(data=[go.Table(
        header=dict(values=df.columns),
        cells=dict(values=[df[col] for col in df.columns])
    )])
    return table
