# CMP Pad 개공률 분석기

CMP 패드 표면 SEM 이미지를 이진화하여 개구부를 분할하고, 원형도·solidity·**개공률(open-pore fraction)** 을 산출하는 데스크톱 프로그램입니다.

이미지를 창에 끌어다 놓으면 계산됩니다. Python 설치는 필요 없습니다.

---

## 1. 실행 파일 받기

GitHub Actions가 각 OS의 러너에서 빌드하므로, 어느 환경에서 푸시하든 두 플랫폼 실행 파일이 모두 나옵니다.

| 플랫폼 | 러너 | 산출물 |
|---|---|---|
| Windows | `windows-latest` | `CMP_PoreAnalyzer.exe` |
| macOS (Apple Silicon) | `macos-14` | `CMP_PoreAnalyzer.app` (zip) |

| 방법 | 절차 |
|---|---|
| **Actions 탭** | Actions → `Build executables` → 최신 실행 → Artifacts에서 원하는 플랫폼 다운로드 |
| **Releases** | `git tag v1.0.0 && git push origin v1.0.0` → Releases에 두 파일 자동 첨부 |
| **로컬 빌드** | Windows는 `build.bat`, macOS는 `./build.sh` |

### 첫 실행 시 보안 경고

두 빌드 모두 **코드 서명이 없습니다.** OS가 경고를 띄우는 것은 정상입니다.

**Windows** — SmartScreen 경고에서 `추가 정보` → `실행`.
사내 PC 정책으로 차단되면 서명이 필요하므로 IT에 문의하십시오.

**macOS** — 처음에는 "확인되지 않은 개발자" 경고가 뜹니다. 둘 중 하나로 해결하십시오.

```bash
# 방법 1: 격리 속성 제거 (권장)
xattr -dr com.apple.quarantine /경로/CMP_PoreAnalyzer.app

# 방법 2: Finder에서 앱을 우클릭 → 열기 → 열기
```

압축은 반드시 Finder에서 풀거나 `unzip`을 쓰십시오. 번들 내부 심볼릭 링크가 깨지면 실행되지 않습니다.

> 실행 파일은 단일 파일 형태라 **80–120 MB** 정도이고, 실행할 때마다 내부를 임시 폴더에 풀기 때문에 **첫 창이 뜨기까지 3–10초** 걸립니다. 정상 동작입니다. 과학 계산 라이브러리(numpy·scipy·scikit-image)를 통째로 담고 있어 크기를 더 줄이기는 어렵습니다.

---

## 2. 분석 파이프라인

논문에 기술한 절차와 동일합니다.

```
그레이스케일 정규화
  → Gaussian 평활 (σ = 1.2 px)
  → 고정 임계 이진화 (정규화 밝기 0.45 미만 = 어두운 영역)
  → 형태학적 opening (disk r = 2 px)
  → 연결 성분 라벨링
  → 등가직경 2 µm 미만 객체 제외
  → 프레임 접촉 객체 제외
  → 원형도 = 4πA/P² , solidity = A / A_convex hull
  → 개공률 = (solidity ≥ 0.90 인 객체의 면적 합) / 시야 면적
```

### 왜 단순 암부면적률을 쓰지 않는가

단순히 어두운 픽셀의 비율을 세면 **결론과 반대 방향**이 나옵니다. 어두운 영역에는 노출된 기공뿐 아니라 찢긴 리가먼트 사이의 **그림자**가 섞여 있어, 표면이 거칠수록 값이 커지기 때문입니다.

이 프로그램은 두 값을 모두 보여줍니다.

- `개공률` — solidity 기준을 통과한 볼록 개구부만 합산한 **주 지표**
- `암부면적 %` — 그림자를 포함한 전체 어두운 면적, **참고용**

### 절대값 사용에 대한 경고

임계값과 solidity 기준을 바꾸면 개공률 **절대값은 크게 변합니다**. 동일 설정에서 두 조건을 비교한 **대소 관계와 비율**만 사용하시고, 절대값을 인용하실 때는 반드시 측정 조건을 병기하십시오.

`강건성 스윕` 버튼이 임계 0.35–0.65 × solidity 0.85–0.95의 21개 조합을 계산해 주므로, 순서가 뒤집히지 않는지 직접 확인하실 수 있습니다.

---

## 3. 사용법

### GUI

1. 이미지를 창에 끌어다 놓거나 `이미지 추가` 클릭 (여러 장 동시 가능)
2. 필요하면 분석 조건 수정 — 특히 **픽셀 크기(µm/px)** 는 배율에 맞게 반드시 확인
3. `분석 실행`
4. `CSV 저장` / `오버레이 저장` / `강건성 스윕`

오버레이에서 **녹색**은 solidity 기준을 통과한 볼록 개구부, **적색**은 오목하거나 제외된 객체입니다. 분할이 타당한지 눈으로 먼저 확인하십시오.

### 명령줄 (배치 처리)

Windows:

```bat
CMP_PoreAnalyzer.exe --cli "C:\images\*.png" --csv result.csv --overlay-dir ov
CMP_PoreAnalyzer.exe --cli MID.png IN.png --threshold 0.45 --solidity 0.90 --sweep
```

macOS — 앱 번들 안의 실행 파일을 직접 호출하십시오:

```bash
CMP_PoreAnalyzer.app/Contents/MacOS/CMP_PoreAnalyzer --cli "*.png" --csv result.csv
```

주요 옵션: `--pixel-size` `--sigma` `--threshold` `--min-diam` `--solidity`
`--opening-radius` `--crop-bottom` `--no-autocrop` `--keep-border` `--sweep`

---

## 4. 분석 조건

| 항목 | 기본값 | 설명 |
|---|---|---|
| 픽셀 크기 | 0.404 µm/px | **배율이 다르면 반드시 수정** |
| Gaussian σ | 1.2 px | 실질 판정 단위는 약 ±1.2 µm 이웃의 가중평균 |
| 이진화 임계 | 0.45 | 정규화 밝기, 이 값 미만이 어두운 영역 |
| 최소 등가직경 | 2.0 µm | 유효 최소 스케일 (0.404 µm/px에서 약 5 px) |
| Solidity 기준 | 0.90 | 이 값 이상만 개공률 분자에 포함 |
| Opening 반경 | 2 px | 폭 약 1.6 µm 미만의 가는 연결부 절단 |
| 하단 크롭 | 0 (자동) | SEM 정보바 감지 시 자동 제거, 수동 지정 가능 |
| 프레임 접촉 제외 | 켜짐 | 잘린 객체는 solidity를 신뢰할 수 없음 |

---

## 5. 해석 시 유의사항

- **프레임 접촉 객체 제외로 개공률은 과소평가됩니다.** 큰 객체일수록 프레임에 닿을 확률이 높기 때문이며, 제외된 면적 비율을 결과에 함께 표시합니다. 논문에 한 줄 밝히시는 것이 정직합니다.
- **"개공률"은 표준 용어와 다른 조작적 정의입니다.** 정의를 병기하지 않으면 독자가 전체 암부면적률을 기대하고 읽습니다.
- **조건당 시야 1장은 부족합니다.** 한 시야 안의 객체들은 공간적으로 상관되어 있어, 객체를 표본 단위로 삼은 신뢰구간은 실제보다 좁습니다. 조건당 5시야 이상 확보하신 뒤 시야를 단위로 다시 계산하시는 것이 안전합니다.
- 분할된 어두운 영역에는 노출된 기공과 표면 그림자가 함께 포함됩니다.

`pore_analyzer/core.py`의 `bootstrap_median_diff()`로 두 조건의 중앙값 차이에 대한 95% 신뢰구간을 구할 수 있습니다. 위 한계가 그대로 적용되므로 주의해서 사용하십시오.

---

## 6. 구조

```
main.py                     GUI / CLI 진입점
pore_analyzer/core.py       알고리즘 (GUI 의존성 없음)
pore_analyzer/gui.py        tkinter GUI
pore_analyzer/cli.py        명령줄 인터페이스
tests/smoke_test.py         CI 검증 (합성 이미지)
tests/make_sample.py        합성 테스트 이미지 생성
.github/workflows/build.yml Windows/macOS 실행 파일 자동 빌드
build.bat                   로컬 Windows 빌드
build.sh                    로컬 macOS/Linux 빌드
```

모든 연산은 로컬에서 수행되며 이미지는 외부로 전송되지 않습니다.
