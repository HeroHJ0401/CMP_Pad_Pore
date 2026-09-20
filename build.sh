#!/usr/bin/env bash
# ---------------------------------------------------------------------
#  macOS / Linux에서 직접 빌드할 때 사용합니다.
#  GitHub Actions를 쓰시면 이 파일은 필요 없습니다.
#
#  사용법: chmod +x build.sh && ./build.sh
# ---------------------------------------------------------------------
set -euo pipefail

echo "[1/4] 가상환경 생성..."
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/4] 의존성 설치..."
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt
pip install pyinstaller

echo "[3/4] 실행 파일 빌드... (수 분 걸립니다)"
pyinstaller --noconfirm --clean --onefile --windowed \
  --name CMP_PoreAnalyzer \
  --osx-bundle-identifier com.cmp.poreanalyzer \
  --collect-all skimage \
  --collect-all tkinterdnd2 \
  --hidden-import PIL._tkinter_finder \
  --exclude-module matplotlib \
  --exclude-module pytest \
  --exclude-module IPython \
  main.py

echo "[4/4] 동작 확인..."
python tests/make_sample.py sample.png
if [ -d dist/CMP_PoreAnalyzer.app ]; then
    BIN="dist/CMP_PoreAnalyzer.app/Contents/MacOS/CMP_PoreAnalyzer"
else
    BIN="dist/CMP_PoreAnalyzer"
fi
"$BIN" --cli sample.png --csv out.csv
test -f out.csv

echo
echo "===================================================="
if [ -d dist/CMP_PoreAnalyzer.app ]; then
    echo " 빌드 완료: dist/CMP_PoreAnalyzer.app"
    echo " 로컬 빌드본은 격리 속성이 없어 바로 실행됩니다."
else
    echo " 빌드 완료: dist/CMP_PoreAnalyzer"
fi
echo "===================================================="
