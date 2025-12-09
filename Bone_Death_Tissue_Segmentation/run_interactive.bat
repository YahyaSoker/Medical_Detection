@echo off
setlocal enabledelayedexpansion
title Kemik Segmentasyon Modeli
color 0B

echo ========================================
echo    KEMIK SEGMENTASYON MODELI
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo Mevcut modeller:
echo.
set /a count=0
for %%f in (*.pth) do (
    set /a count+=1
    echo !count!. %%~nxf
)

echo.
set /p model_num="Model secimi (1-%count%): "
if "%model_num%"=="" set model_num=1

set /a current=0
for %%f in (*.pth) do (
    set /a current+=1
    if !current!==%model_num% (
        set selected_model=%%f
    )
)

echo.
echo Secilen model: %selected_model%
echo.

echo Mevcut test goruntuleri:
echo.
set /a count=0
for %%f in (merged\*.jpg) do (
    set /a count+=1
    echo !count!. %%~nxf
)

echo.
set /p img_num="Goruntu secimi (1-%count%): "
if "%img_num%"=="" set img_num=1

set /a current=0
for %%f in (merged\*.jpg) do (
    set /a current+=1
    if !current!==%img_num% (
        set selected_image=%%f
    )
)

echo.
echo Secilen goruntu: %selected_image%
echo.

echo Python script calistiriliyor...
echo.

python simple_test.py "%selected_image%" "%selected_model%"

echo.
echo ========================================
echo    TEST TAMAMLANDI!
echo ========================================
echo.

if exist "simple_test_result.png" (
    echo OK - simple_test_result.png olusturuldu
) else (
    echo HATA - simple_test_result.png olusturulamadi!
)

if exist "detailed_analysis.png" (
    echo OK - detailed_analysis.png olusturuldu
) else (
    echo HATA - detailed_analysis.png olusturulamadi!
)

echo.
echo Cikmak icin herhangi bir tusa basin...
pause >nul
