from app import app

@app.route('/')
def home():
    return "Hello, Flask!"


# Define as rotas da sua aplicação Flask. Aqui você cria as URLs e as funções que respondem a essas URLs.
