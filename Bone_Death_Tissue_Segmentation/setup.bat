@echo off
echo ========================================
echo    KEMIK SEGMENTASYON MODELI KURULUMU
echo ========================================
echo.

echo [1/2] Python kontrolu yapiliyor...
python --version >nul 2>&1
if errorlevel 1 (
    echo HATA: Python yuklu degil!
    echo Lutfen Python 3.8+ yukleyin: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo OK - Python bulundu

echo.
echo [2/3] Pip guncelleniyor...
python -m pip install --upgrade pip

echo.
echo [3/3] Gerekli kutuphaneler yukleniyor...
echo Bu islem 5-10 dakika surebilir...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo HATA: Kutuphane yukleme basarisiz!
    echo Lutfen internet baglantinizi kontrol edin.
    pause
    exit /b 1
)

echo.
echo ========================================
echo    KURULUM TAMAMLANDI!
echo ========================================
pause


