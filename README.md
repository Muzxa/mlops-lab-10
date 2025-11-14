# MLOps Lab 10
## Installation

Clone this project into your local repository

```bash
  git clone https://github.com/Muzxa/mlops-lab-10.git
```
Navigate to the project directory
```bash
    cd mlops-lab-10
```
Make sure the docker daemon is running in the background
```bash
    docker --version
```
Run the following command
```bash
    docker compose up --build
```
This will spool up all the docker containers required to run the file. This may take a few minutes.

You can access the Flask App at localhost:500 and the MLFlow UI at localhost:5001

If you would like to train a model, run

```bash
    docker compose up trainer
```
This will train a fresh model and store it in mlflow

Note: If you are following these steps for the first time, train the model before using the FLask app!