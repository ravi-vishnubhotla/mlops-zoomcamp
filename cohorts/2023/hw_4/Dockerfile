FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

# RUN pip list
RUN pip freeze

# Copy predict.py to /app
COPY [ "predict.py", "./" ]

# list files in the /app directory
RUN ls -la /app

# execute predict.py with arguments --year and --month
CMD [ "python", "predict.py", "--year", "2022" , "--month", "4" ] # can be overwritten with docker run --env YEAR=2022 --env MONTH=4
