# 1. use the official Python image as the base image
FROM python:3.11-slim

# 2. set the working directory in the container
WORKDIR /code

# 3. copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# 4. install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#5. copy the rest of the application code into the container
COPY . /code

# 6. expose the port that the application will run on
EXPOSE 8000

# 7. define the command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]