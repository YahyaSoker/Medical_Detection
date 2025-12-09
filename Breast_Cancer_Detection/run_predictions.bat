@echo off
echo Setting OpenMP environment variable to avoid runtime conflicts...
set KMP_DUPLICATE_LIB_OK=TRUE

echo.
echo Starting YOLO Prediction Pipeline...
echo ====================================

python main.py

echo.
echo Press any key to exit...
pause >nul
