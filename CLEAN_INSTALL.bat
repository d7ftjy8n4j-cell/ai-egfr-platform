@echo off
echo ========================================
echo Cleaning and reinstalling dependencies
echo ========================================

echo.
echo Step 1: Uninstalling conflicting packages...
pip uninstall -y markdown-it-py rich

echo.
echo Step 2: Installing from requirements.txt...
pip install -r requirements.txt

echo.
echo Step 3: Verifying installations...
python -c "import streamlit; print(f'[OK] Streamlit: {streamlit.__version__}')"
python -c "import rich; print(f'[OK] Rich: {rich.__version__}')"
python -c "import numpy; print(f'[OK] Numpy: {numpy.__version__}')"

echo.
echo Step 4: Testing streamlit app...
echo Run: streamlit run app.py

echo.
echo ========================================
echo Done!
echo ========================================
