Here’s the updated content of your file with corrected step numbering, a **Table of Contents**, and improved formatting for clarity:

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
   - [Step 1: Install PostgreSQL](#step-1-install-postgresql)
   - [Step 2: Clone the Repository](#step-2-clone-the-repository)
   - [Step 3: Set Up the Python Environment](#step-3-set-up-the-python-environment)
   - [Step 4: Configure the Environment](#step-4-configure-the-environment)
   - [Step 5: Install Dependencies](#step-5-install-dependencies)
   - [Step 6: Run Migrations](#step-6-run-migrations)
4. [Running the Project](#running-the-project)
5. [Create Regular Daily Forecast](#create-regular-daily-forecast)
6. [License](#license)
7. [Contact](#contact)

---

## Overview
The project contains a Python implementation for the VMD-TiDE model. This repository requires setting up a new environment with Python 3.10 and installing dependencies from `requirements.txt`. Follow the instructions below to set up and run the project.

### Эксперимент F10.7 + SN в Google Colab

[Открыть тетрадь из fork в Colab](https://colab.research.google.com/github/Yuri-ga1/VMD-TiDE/blob/master/notebooks/F107_SN.ipynb)


---

## Prerequisites
Before you begin, ensure that you have the following installed on your system:
- **Git** (for cloning the repository)
- **Python 3.10** (required for this project)

---

## Installation Steps

### Step 1: Install PostgreSQL
Install PostgreSQL and create a database for the project:
```bash
sudo apt install postgresql postgresql-contrib -y
sudo systemctl start postgresql
sudo systemctl enable postgresql
```
Switch to the `postgres` user and create a database:
```bash
sudo -i -u postgres
createdb forecast_project
createuser --pwprompt forecast_user
```
Grant privileges to the user:
```sql
psql
GRANT ALL PRIVILEGES ON DATABASE forecast_project TO forecast_user;
\q
psql -U postgres -d forecast_project
CREATE SCHEMA f107;
GRANT USAGE ON SCHEMA f107 TO forecast_user;
GRANT CREATE ON SCHEMA f107 TO forecast_user;
\q
exit
```

---

### Step 2: Clone the Repository
To get started, clone the project repository using the following commands:
```bash
mkdir -m 755 projects
cd projects
git clone https://git.iszf.irk.ru/diegon/f107.git
cd f107
```

---

### Step 3: Set Up the Python Environment
Create and activate a Python virtual environment:
```bash
python3 -m venv forecast
source forecast/bin/activate
```

---

### Step 4: Configure the Environment
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Update the `.env` file with your environment-specific settings.

---

### Step 5: Install Dependencies
Install the required dependencies listed in the `requirements.txt` file:
```bash
sudo apt install python3-pip
sudo apt install python3-venv
pip install --upgrade pip
pip install -r requirements.txt
```
This will install all the necessary packages and libraries for the project.

---

### Step 6: Run Migrations
Run the database migrations:
```bash
python3 migrate.py
```

---

## Running the Project
How to train and predict with the model can be found in the `example.ipynb` Jupyter Notebook. Alternatively, you can run scripts from the terminal.

Run the Decomposition of Original Signal:
```bash
python3 data_decompose.py
```

Run training and prediction for a 7-day horizon:
```bash
python3 regular_job_7.py
```

Run training and prediction for a 27-day horizon:
```bash
python3 regular_job_27.py
```

---

## Create Regular Daily Forecast

### Step 1: Open the Crontab Editor
Open the crontab file for editing using the following command:
```bash
crontab -e
```
This will open the crontab editor in your default text editor.

---

### Step 2: Add Cron Jobs
Add the following lines to the crontab file to schedule the scripts:
```cron
# Run Decomposition of Original Signal at 01:00 AM daily
0 1 * * * /usr/bin/python3 /path/to/regular_job.py >> /var/log/regular_job.log 2>&1

# Run training and prediction for 7-day horizon at 02:00 AM daily
0 2 * * * /usr/bin/python3 /path/to/regular_job_7.py >> /var/log/regular_job_7.log 2>&1

# Run training and prediction for 27-day horizon at 03:00 AM daily
0 3 * * * /usr/bin/python3 /path/to/regular_job_27.py >> /var/log/regular_job_27.log 2>&1
```

---

### Step 3: Create Log Files (Optional)
Ensure the log files exist and are writable by the user running the cron jobs:
```bash
sudo touch /var/log/regular_job.log
sudo touch /var/log/regular_job_7.log
sudo touch /var/log/regular_job_27.log
sudo chmod 644 /var/log/regular_job.log /var/log/regular_job_7.log /var/log/regular_job_27.log
```
Alternatively, you can specify different log paths or omit logging entirely if not needed.

---

### Step 4: Save and Exit
Save the crontab file and exit the editor. The cron daemon will automatically reload the updated configuration.

---

### Step 5: Verify Cron Jobs
To verify that the cron jobs have been added successfully, list the current cron jobs:
```bash
crontab -l
```
You should see the three scheduled jobs listed.

---

## License
This project is licensed under GPL-3.0. See the `LICENSE` file for more details.

---

## Contact
For any questions or issues, please contact Yaroslav Egorov at egorov@iszf.irk.ru.

---