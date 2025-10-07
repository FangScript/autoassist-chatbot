import os
from app import app as application  # Vercel expects WSGI callable named `app` or `application`

# Optional: ensure Flask runs in production mode on Vercel
os.environ.setdefault("FLASK_ENV", "production")

# Expose `app` for some Python runtimes that look for it
app = application


