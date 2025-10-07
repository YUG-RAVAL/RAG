# Use Python 3.10 slim image as base
FROM python:3.10.12

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies and gunicorn
RUN pip install --no-cache-dir -r requirements.txt gunicorn eventlet

# Copy the rest of the application
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Create directory for Chroma DB
RUN mkdir -p chroma_db

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0

# Expose port 8080 for the Flask application
EXPOSE 8080

# Create a script to run the application
RUN echo '#!/bin/bash\n\
echo "Starting application..."\n\
if [ -n "$OPENAI_API_KEY" ]; then\n\
  echo "OPENAI_API_KEY status: available (masked for security)"\n\
else\n\
  echo "OPENAI_API_KEY status: missing"\n\
fi\n\
echo "FLASK_APP: ${FLASK_APP}"\n\
echo "FLASK_ENV: ${FLASK_ENV}"\n\
echo "FLASK_DEBUG: ${FLASK_DEBUG}"\n\
echo "PYTHONUNBUFFERED: ${PYTHONUNBUFFERED}"\n\
exec gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:8080 --log-level debug --error-logfile - --access-logfile - --capture-output run:app\n\
' > /app/start.sh && chmod +x /app/start.sh

# Command to run the application with gunicorn
CMD ["/app/start.sh"]
