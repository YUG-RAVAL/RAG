from rag_chatbot import create_app

app, socketio = create_app()

if __name__ == '__main__':
    socketio.run(
        app,
        debug=True,
        host="0.0.0.0",
        port=8080,
        allow_unsafe_werkzeug=True
    )  # Use socketio.run instead of app.run