@echo off
REM ---------------------------------------------------------------------
REM  Windows PC에서 직접 .exe를 빌드할 때 사용합니다.
REM  GitHub Actions를 쓰시면 이 파일은 필요 없습니다.
REM
REM  사전 준비: Python 3.10~3.12 설치 (설치 시 "Add to PATH" 체크)
REM  사용법  : 이 폴더에서 build.bat 더블클릭 또는 실행
REM ---------------------------------------------------------------------
setlocal

echo [1/4] 가상환경 생성...
if not exist .venv (
    python -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat || goto :fail

echo [2/4] 의존성 설치...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || goto :fail
pip install pyinstaller || goto :fail

echo [3/4] 실행 파일 빌드... (수 분 걸립니다)
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name CMP_PoreAnalyzer ^
  --collect-all skimage ^
  --collect-all tkinterdnd2 ^
  --hidden-import PIL._tkinter_finder ^
  --exclude-module matplotlib ^
  --exclude-module pytest ^
  --exclude-module IPython ^
  main.py || goto :fail

echo [4/4] 동작 확인...
python tests\make_sample.py sample.png || goto :fail
dist\CMP_PoreAnalyzer.exe --cli sample.png --csv out.csv || goto :fail
if not exist out.csv goto :fail

echo.
echo ====================================================
echo  빌드 완료: dist\CMP_PoreAnalyzer.exe
echo ====================================================
pause
exit /b 0

:fail
echo.
echo *** 빌드 실패 — 위 메시지를 확인하십시오. ***
pause
exit /b 1
