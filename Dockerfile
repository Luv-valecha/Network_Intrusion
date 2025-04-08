FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy everything into the container
COPY . .

# Install dependencies from API folder
RUN pip install -r requirements.txt

# Expose port (Flask default is 5000)
EXPOSE 8080

# Run with waitress
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "API.wsgi:app"]
