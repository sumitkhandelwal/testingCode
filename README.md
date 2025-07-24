<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Multi-Database App</title>
    {% load static %}
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
    <style>
        /* Simple inline CSS for quick styling - move to style.css later */
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .login-container {
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            width: 350px;
            text-align: center;
        }
        h2 {
            margin-bottom: 20px;
            color: #333;
        }
        .form-group {
            margin-bottom: 15px;
            text-align: left;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: bold;
        }
        .form-group input[type="text"],
        .form-group input[type="password"] {
            width: calc(100% - 20px); /* Account for padding */
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box; /* Include padding and border in the element's total width and height */
        }
        .form-group button {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        .form-group button:hover {
            background-color: #0056b3;
        }
        .errorlist {
            color: red;
            list-style-type: none;
            padding: 0;
            margin-top: -10px;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .messages {
            list-style-type: none;
            padding: 0;
            margin-bottom: 15px;
        }
        .messages li {
            padding: 10px;
            margin-bottom: 5px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .messages .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .messages .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>Login to Application</h2>

        {# Display messages from Django's messages framework #}
        {% if messages %}
            <ul class="messages">
                {% for message in messages %}
                    <li{% if message.tags %} class="{{ message.tags }}"{% endif %}>{{ message }}</li>
                {% endfor %}
            </ul>
        {% endif %}

        <form method="post">
            {% csrf_token %} {# Django's CSRF protection #}

            {# Display non-field errors (e.g., "invalid credentials") #}
            {% if form.non_field_errors %}
                <ul class="errorlist">
                    {% for error in form.non_field_errors %}
                        <li>{{ error }}</li>
                    {% endfor %}
                </ul>
            {% endif %}

            <div class="form-group">
                <label for="{{ form.username.id_for_label }}">Username:</label>
                {{ form.username }}
                {% if form.username.errors %}
                    <ul class="errorlist">
                        {% for error in form.username.errors %}
                            <li>{{ error }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
            </div>

            <div class="form-group">
                <label for="{{ form.password.id_for_label }}">Password:</label>
                {{ form.password }}
                {% if form.password.errors %}
                    <ul class="errorlist">
                        {% for error in form.password.errors %}
                            <li>{{ error }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
            </div>

            {# Hidden input for 'next' URL parameter if you want to redirect after login #}
            {% if next %}
                <input type="hidden" name="next" value="{{ next }}" />
            {% endif %}

            <div class="form-group">
                <button type="submit">Login</button>
            </div>
        </form>
    </div>
</body>
</html>
