@echo off
echo ========================================
echo Fixing rich version for streamlit
echo ========================================

echo.
echo Step 1: Uninstalling current rich...
pip uninstall -y rich

echo.
echo Step 2: Installing rich 13.7.1...
pip install rich==13.7.1

echo.
echo Step 3: Verifying installation...
python -c "import rich; print(f'[OK] Rich version: {rich.__version__}')"

echo.
echo Step 4: Checking streamlit compatibility...
python -c "import streamlit; print(f'[OK] Streamlit version: {streamlit.__version__}')"

echo.
echo ========================================
echo Done! Dependencies are now compatible.
echo ========================================
