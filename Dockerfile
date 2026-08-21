FROM python:3.12-slim

WORKDIR /app

# Install dependencies from the fully-pinned lock, never the >= ranges in
# requirements.txt. The deploy path rebuilds with --no-cache, so installing
# the ranges re-resolved every package on every rebuild; see the header of
# requirements.lock.txt for the four major-version drifts that caused.
# requirements.txt is copied too, purely so the image records the declared
# intent alongside the resolved set.
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

# Copy application code
COPY . .

# Create the volume mount points. Both are bind-mounted at runtime and are now
# excluded from the build context, so create them explicitly rather than relying
# on COPY having brought them in.
RUN mkdir -p logs data

# Run the bot
CMD ["python", "main.py"]
