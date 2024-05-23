from flask import Flask, request, jsonify
import sympy as sp
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

def newton_raphson(f, x, x0, tol=1e-6, max_iter=100):
    df = sp.diff(f, x)
    f_func = sp.lambdify(x, f)
    df_func = sp.lambdify(x, df)
    for i in range(max_iter):
        fx = f_func(x0)
        dfx = df_func(x0)
        if dfx == 0:
            raise ValueError("Derivative is zero, no solution found.")
        x_new = x0 - fx / dfx
        if abs(x_new - x0) < tol:
            return x_new
        x0 = x_new
    raise ValueError("Maximum iterations exceeded, no solution found.")


@app.route("/")
def start():
    return "The Server is Running"

@app.route('/api/newton_raphson', methods=['POST'])
def api_newton_raphson():
    try:
        data = request.json
        if data is None:
            raise ValueError("No JSON data provided.")
        func_str = data.get('function')
        x_symbol = data.get('variable', 'x')
        x = sp.symbols(x_symbol)
        x0 = data.get('x0')
        if func_str is None or x0 is None:
            raise ValueError("Missing required fields in JSON data.")
        tol = data.get('tol', 1e-6)
        max_iter = data.get('max_iter', 100)

        f = sp.sympify(func_str)

        # Compute the root using Newton-Raphson method
        root = newton_raphson(f, x, x0, tol, max_iter)
        result = {"root": root}
    except Exception as e:
        result = {"error": str(e)}

    return jsonify(result)


