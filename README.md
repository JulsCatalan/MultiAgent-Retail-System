# SF - FastAPI Application

A FastAPI application with a simple ping endpoint.

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

### Install uv

If you don't have `uv` installed, you can install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Sync Dependencies

To install all project dependencies defined in `pyproject.toml`, run:

```bash
uv sync
```

This will:

- Create a virtual environment (if it doesn't exist)
- Install all dependencies from `pyproject.toml`
- Update `uv.lock` if needed

## Running the Application

### Option 1: Direct Python execution

```bash
python app/main.py
```

### Option 2: Using uvicorn directly

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option 3: Using uv run

```bash
uv run python app/main.py
```

Or:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The application will start on `http://0.0.0.0:8000` (or `http://localhost:8000`).

## API Endpoints

### GET /ping

Returns a simple pong message.

**Example:**

```bash
curl http://localhost:8000/ping
```

**Response:**

```json
{
  "message": "pong"
}
```

## Development

### Adding Dependencies

To add a new dependency:

```bash
uv add <package-name>
```

This will automatically update `pyproject.toml` and `uv.lock`.

### Updating Dependencies

To update all dependencies to their latest compatible versions:

```bash
uv sync --upgrade
```

### Viewing Dependencies

To see the dependency tree:

```bash
uv tree
```
