from django.shortcuts import render
from django.shortcuts import render,redirect
from django.urls import reverse
from .forms import CustomUserCreationForm,LoginForm
from Dataspy.models import *
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import uuid


def register(request):
    """View for user registration."""
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            profile = UserProfile(user=user)
            profile.save()

            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            messages.success(request, 'Registration successful. You are now logged in.')
            return redirect('Dataspy:dashboard')
    else:
        form = CustomUserCreationForm()
    return render(request, 'logusers/register.html', {'form': form})


def login_view(request):
    # Get the URL to redirect to after login
    next_url = request.GET.get('next', reverse('Dataspy:dashboard'))

    if request.method == 'POST':
        form = LoginForm(data=request.POST)  # Pass form data
        if form.is_valid():
            # Extract data from the form
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            # Authenticate the user
            user = authenticate(request, username=username, password=password)

            if user is not None:
                # Log in the user
                login(request, user)  # Ensure user is passed here
                return redirect(next_url)  # Redirect to the original page or home
            else:
                # Add a non-field error to the form for feedback
                form.add_error(None, 'Invalid email/username or password')
    else:
        form = LoginForm()

    return render(request, 'userlogs/login.html', {'form': form})

@login_required(login_url="/login/")
def logout_view(request):
    logout(request)
    return redirect(reverse('Dataspy:dashboard'))  # Redirect to the homepage after logout




