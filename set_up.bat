@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%env"

echo ==============================================
echo   Configurando entorno virtual del proyecto
Echo ==============================================

echo Directorio del proyecto: %PROJECT_DIR%

echo.
echo [1/3] Creando (si no existe) el entorno virtual en %VENV_DIR%
if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Error al crear el entorno virtual.
        exit /b 1
    )
) else (
    echo El entorno virtual ya existe, se reutilizara.
)

echo.
echo [2/3] Activando entorno virtual...
call "%VENV_DIR%\Scripts\activate"
if errorlevel 1 (
    echo No se pudo activar el entorno virtual.
    exit /b 1
)

echo.
echo [3/3] Instalando dependencias del proyecto
python -m pip install --upgrade pip
pip install --no-cache-dir -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 (
    echo Hubo errores durante la instalacion de dependencias.
    exit /b 1
)

echo.
echo Entorno configurado correctamente. Puedes comenzar a trabajar.
exit /b 0
