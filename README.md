# RAG-BE\
## First Time Setup Instructions

To set up the RAG-BE project on another machine, follow these steps:

### Prerequisites

1. **Python 3.8+**: Ensure you have Python 3.8 or higher installed. You can download it from [python.org](https://www.python.org/downloads/).

2. **Virtual Environment**: It's recommended to use a virtual environment to manage dependencies. You can create one using `venv` or `virtualenv`.

3. **PostgreSQL**: Ensure you have PostgreSQL installed and running. You can download it from [postgresql.org](https://www.postgresql.org/download/).

4. **Redis**: Ensure you have Redis installed and running. You can download it from [redis.io](https://redis.io/download).

### Steps

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/RAG-BE.git
    cd RAG-BE
    ```

2. **Create and Activate Virtual Environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**:
    - Copy the `.env` file from the repository and update it with your own credentials and API keys.
    - Ensure the following variables are set:
        ```sh
        OPENAI_API_KEY=your_openai_api_key
        ANTHROPIC_API_KEY=your_anthropic_api_key
        DEEPSEEK_API_KEY=your_deepseek_api_key
        GOOGLE_API_KEY=your_google_api_key
        DB_USER=your_db_user
        DB_PASSWORD=your_db_password
        DB_HOST=your_db_host
        DB_PORT=your_db_port
        DB_NAME=your_db_name
        SECRET_KEY=your_secret_key
        GOOGLE_CLIENT_ID=your_google_client_id
        GOOGLE_CLIENT_SECRET=your_google_client_secret
        JWT_SECRET_KEY=your_jwt_secret_key
        VERIFY_USER_URL=your_verify_user_url
        ```

5. **Initialize the Database**:
    - Ensure PostgreSQL is running.
    - Create the database specified in the `.env` file.

6. **Run the Application**:
    ```sh
    python app.py
    ```

7. **Access the Application**:
    - Open your browser and navigate to `http://localhost:5000` to access the application.

8. **Build and Run with Docker**:
    - Ensure Docker is installed and running on your machine.
    - Build the Docker image using the provided Dockerfile in the codebase:
        ```sh
        docker build -t rag-be .
        ```
    - Run the Docker container using the environment variables specified in your `.env` file:
        ```sh
        docker run -d -p 5000:5000 --name rag-be-container --env-file .env rag-be
        ```
    - This will start the application inside a Docker container and map port 5000 of the container to port 5000 on your host machine.

### Additional Notes

- **Logging**: The application uses logging to provide insights into its operations. Check the logs for any issues during setup or runtime.
- **API Documentation**: Refer to the LangGraph and LlamaIndex documentation for more details on the APIs used in this project:
  - [LangGraph Documentation](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
  - [LlamaIndex Documentation](https://docs.cloud.llamaindex.ai/)

If you encounter any issues during setup, please refer to the logs or reach out to the project maintainers for assistance.
