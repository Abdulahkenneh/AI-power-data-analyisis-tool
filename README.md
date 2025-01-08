# AI-power-data-analysis-tool

**AI-power-data-analysis-tool** is a powerful data analysis and visualization platform built with Django. It provides various features such as data cleaning, reporting, insights generation, code generation, and custom analysis, all integrated into a user-friendly dashboard. This tool is designed for individuals and organizations looking to gain valuable insights from their data, automate analysis processes, and generate visualizations for better decision-making.

> **Note**: This project is still under development. Features are continuously being added, and the application is being improved for stability, performance, and user experience.

---

## Features

### Dashboard
- **Dashboard**: A central hub to navigate and manage all data-related tasks and insights.
- **User Profile**: View and update personal information and settings.
- **Data Cleaning**: Clean and preprocess your data for analysis.
- **Spreadsheet Display**: View and interact with spreadsheets containing your data.

### Data Analysis
- **Data Query**: Query your data for detailed analysis and insights.
- **Auto Update**: Automatically update data for real-time analysis.
- **Task Status**: Check the status of ongoing tasks and operations.
- **Data Deletion**: Easily delete files from the system.
- **Insights Generation**: Generate insights based on your data analysis.
- **Code Generation**: Automatically generate analysis code based on your data.
- **Visualizations**: Create visual representations (graphs, charts) from your data.

### Reporting
- **Reports**: View and generate comprehensive reports based on your data.
- **Export Options**: Export data and reports in various formats.

### Support & Settings
- **Help Documentation**: Access detailed documentation for using the platform.
- **Community Feedback**: Provide feedback and suggestions to improve the system.
- **User Settings**: Manage user preferences and settings.
  
### Admin Panel
- **Admin Panel**: Access exclusive admin tools for managing the platform and user data.
- **Custom Analysis**: Perform custom data analysis based on unique requirements.

---

## Installation

Follow these steps to get your development environment set up and run the project locally.

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/AI-power-data-analysis-tool.git
   cd AI-power-data-analysis-tool



Create a virtual environment

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Set up the database

Run migrations to set up the database tables:

bash
Copy code
python manage.py migrate
Create a superuser

Create an admin user to access the admin panel:

bash
Copy code
python manage.py createsuperuser
Run the server

Start the Django development server:

bash
Copy code
python manage.py runserver
Access the application

Open your browser and go to http://127.0.0.1:8000/ to access the app. The admin panel can be accessed at http://127.0.0.1:8000/admin/.

API Endpoints
Data Analysis
/data_query/: Perform data queries for analysis.
/updata/: Automatically update data.
/api/task-status/<uuid:task_id>/: Check the status of a task.
/insights/: Generate insights from the data.
/code_generation/: Generate code based on analysis.
Reporting
/reports/: View and generate reports.
/export/: Export data and reports.
User Profile & Settings
/userprofile/: View and edit user profile.
/user_settings/: Update user preferences.










