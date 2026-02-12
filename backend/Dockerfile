FROM python:3.12.3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create database tables on startup
CMD ["sh", "-c", "python init_db.py && gunicorn app.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080"]