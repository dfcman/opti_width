@echo off
echo [INFO] Creating virtual environment...
python -m venv .venv

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing requirements...
pip install -r requirements.txt

echo [INFO] Setup complete.
pause
