/* General Body Styling - Appian Style */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    margin: 0;
    background-color: #f7f7f7; /* Light gray background */
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    color: #333;
}

/* Header Styling */
.header {
    display: flex;
    align-items: center;
    padding: 12px 30px;
    background-color: #ffffff;
    border-bottom: 1px solid #e0e0e0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.logo {
    width: 40px;
    height: 40px;
    margin-right: 15px;
}

.header-text {
    color: #2d2d2d;
    font-size: 1.1em;
    font-weight: 600;
}

/* Main Login Container */
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-grow: 1;
    padding: 20px;
}

.login-form {
    background-color: #ffffff;
    padding: 40px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    width: 100%;
    max-width: 400px;
    text-align: center;
    border-top: 4px solid #005a9c; /* Appian-like blue accent */
}

.login-form h2 {
    margin-bottom: 25px;
    color: #333;
    font-weight: 600;
}

/* Form Group Styling */
.form-group {
    margin-bottom: 20px;
    text-align: left;
}

.form-group label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: #555;
    font-size: 0.9em;
}

.form-group input {
    width: 100%;
    padding: 12px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box;
    transition: border-color 0.3s, box-shadow 0.3s;
}

.form-group input:focus {
    outline: none;
    border-color: #005a9c;
    box-shadow: 0 0 0 2px rgba(0, 90, 156, 0.2);
}

/* Button Styling */
.login-btn {
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 4px;
    background-color: #005a9c; /* Primary Appian blue */
    color: white;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: background-color 0.3s;
}

.login-btn:hover {
    background-color: #004b80; /* Darker blue on hover */
}

/* Links */
.links {
    margin-top: 20px;
    font-size: 0.9em;
}

.links a {
    color: #005a9c;
    text-decoration: none;
}

.links a:hover {
    text-decoration: underline;
}

/* Responsive Design */
@media (max-width: 480px) {
    .login-form {
        padding: 25px;
        border-top-width: 3px;
    }
    .header {
        padding: 10px 15px;
    }
    .logo {
        width: 35px;
        height: 35px;
    }
    .header-text {
        font-size: 1em;
    }
}




// You can add form validation or other interactions here later.
document.addEventListener('DOMContentLoaded', (event) => {
    console.log('Login page is loaded and script is running!');
});



from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

def login_view(request):
    """Handles user login."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
            return redirect('login')
            
    return render(request, 'myapp/login.html')

@login_required
def dashboard_view(request):
    """Displays the user's dashboard."""
    return render(request, 'myapp/dashboard.html')

def logout_view(request):
    """Logs the user out."""
    logout(request)
    return redirect('login')





from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),
]



from django.apps import AppConfig


class CoreappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'




{% load static %}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - My Sample Website</title>
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
</head>
<body>

    <header class="header">
        <img src="https://placehold.co/600x600/005a9c/ffffff?text=Logo" alt="Logo" class="logo">
        <span class="header-text">This is my sample Website</span>
    </header>

    <main class="login-container">
        <div class="login-form">
            <h2>Member Login</h2>
            
            {% if messages %}
                <div class="messages" style="margin-bottom: 15px;">
                    {% for message in messages %}
                        <p style="color: #d93025; font-size: 0.9em; background-color: #fce8e6; padding: 10px; border-radius: 4px;">{{ message }}</p>
                    {% endfor %}
                </div>
            {% endif %}

            <form action="{% url 'login' %}" method="POST">
                {% csrf_token %}
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit" class="login-btn">Login</button>
            </form>
            <div class="links">
                <a href="#">Forgot Password?</a>
            </div>
        </div>
    </main>

    <script src="{% static 'js/script.js' %}"></script>
</body>
</html>




{% load static %}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard</title>
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
</head>
<body>
    <header class="header">
        <img src="https://placehold.co/600x600/005a9c/ffffff?text=Logo" alt="Logo" class="logo">
        <span class="header-text">This is my sample Website</span>
    </header>
    <main class="login-container">
        <div class="login-form">
            <h2>Welcome, {{ user.username }}!</h2>
            <p>You have successfully logged in to the dashboard.</p>
            <a href="{% url 'logout' %}" class="login-btn" style="text-decoration: none; display: block; margin-top: 20px; text-align: center;">Logout</a>
        </div>
    </main>
</body>
</html>
