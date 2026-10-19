# MNIST Handwritten Digit Recognition

MNIST 손글씨 숫자 데이터로 얕은 MLP의 SGD·Adam 학습을 비교하고, 깊은 MLP를 학습한 뒤 마우스로 쓴 숫자를 인식하는 프로젝트입니다.

원본 `Play_7_2.py`~`Play_7_5.py`, `Play_7_7.py`를 `reference/`에 그대로 보존했습니다. 실제 실행은 정리된 `train.py`와 `drawing_demo.py`를 사용합니다. 원본 저장소: [Image-processing-4](https://github.com/pod369/Image-processing-4/tree/b2f9a5b94bbe353cce22ce469e358bb2715759ce).

## 구현 내용

| 실험 | 네트워크 | 최적화 방법 | 학습률 | 손실함수 |
|---|---|---|---|---|
| `shallow_sgd` | 784 → 512(tanh) → 10(softmax) | SGD | 0.01 | MSE |
| `shallow_adam` | 784 → 512(tanh) → 10(softmax) | Adam | 0.001 | MSE |
| `deep_mlp` | 784 → 1024 → 512 → 512 → 10 | Adam | 0.0001 | Categorical cross entropy |

- 학습용 60,000장 안에서 클래스별 검증 분할을 만듭니다. 기본값은 학습 약 54,000장, 검증 약 6,000장입니다.
- 공식 시험용 10,000장은 학습이나 체크포인트 선택에 사용하지 않고, 검증 손실이 가장 낮은 저장 모델을 평가할 때만 사용합니다.
- SGD·Adam은 **동일한 초기 가중치, 동일한 데이터 분할과 seed**를 사용합니다. 학습률은 원본의 값으로 서로 다르므로, 결과는 해당 학습률을 포함한 설정 비교입니다.
- 원본에서 누락된 import를 보완하고, 저장 형식을 `.keras`로 변경했습니다.
- 입력 글씨를 자르고 20×20 영역에 맞춘 뒤 28×28 영상의 질량 중심에 정렬합니다. 빈 칸은 숫자로 인식하지 않습니다.
- 원본의 Windows 전용 `winsound` 의존성은 새 프로그램에서 제거했습니다.

## 설치와 실행

Python 3.11을 권장합니다. 아래 명령은 저장소 루트에서 시작합니다.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --model all --epochs 50
python drawing_demo.py --model outputs/deep_mlp.keras
```

MNIST를 처음 실행하면 TensorFlow가 데이터셋을 내려받습니다. CPU에서도 실행 가능하지만 학습 시간이 걸립니다. 창을 표시할 수 없는 서버/Colab에서는 마우스 인식 GUI를 실행할 수 없습니다.

짧은 실행 확인용 학습:

```bash
python train.py --model comparison --epochs 2 --limit-train 2000 --limit-validation 500 --output outputs/quick_demo
```

이 설정은 전체 학습 실험을 대체하지 않습니다. 깊은 모델만 학습하려면 `--model deep`, 한 칸만 쓰려면 `drawing_demo.py --boxes 1`을 사용합니다. 재실행 시 같은 `--output` 폴더의 파일이 교체되므로, 다른 실험은 별도 출력 경로를 지정하세요.

## 마우스 인식

왼쪽 버튼으로 각 칸에 **숫자 하나씩** 씁니다. `r`: 인식, `e`: 지우기, `s`: 전처리 영상 확인, `q` 또는 Esc: 종료. 숫자 옆의 비율은 모델의 softmax 점수이며 보정된 확률이 아닙니다. 실제 마우스 글씨는 MNIST와 모양·획이 달라 오인식할 수 있습니다.

## 생성 결과 및 제출 자료

실제로 학습한 후 지정한 `outputs/`에 아래 파일이 생성됩니다.

- `shallow_sgd.keras`, `shallow_adam.keras`, `deep_mlp.keras`: 실행한 실험의 검증 최적 모델
- `metrics.json`: 실제 시험 정확도, 시험 손실, 검증 최적 epoch, 데이터 수와 설정
- `history.json`, `learning_curves.png`: 학습/검증 정확도 및 손실 곡선
- `*_confusion_matrix.png`: 공식 시험 데이터의 클래스별 혼동행렬
- `split_indices.npz`: 실제 사용한 학습·검증 인덱스

얕은 모델의 MSE와 깊은 모델의 cross entropy 손실은 수치 자체를 직접 비교하지 않습니다. 정확도와 혼동행렬을 함께 해석하세요. seed와 결정적 연산 설정을 사용하지만, 다른 TensorFlow·장치 환경에서 완전히 같은 값이 보장되는 것은 아닙니다.

**이번 저장소 정리에서는 대규모 학습을 수행하지 않았으며, 학습된 가중치나 새 정확도 수치는 포함하지 않았습니다.** 제출 자료에는 직접 실행한 학습 곡선, 시험 정확도, 마우스 인식 화면을 추가하면 됩니다.

## AI-Ex 포트폴리오

2026년 2학기 AI-Ex 포트폴리오의 개별 프로젝트입니다. 원본 파일은 `reference/`에 보존했으며, 정리한 실행 코드와 원본을 구분했습니다. 원본 기준 커밋은 `b2f9a5b94bbe353cce22ce469e358bb2715759ce`입니다. 원본 재배포 라이선스는 별도로 추가하지 않았습니다.

실행 예시와 결과 설명은 이 저장소의 루트 구조에 맞췄습니다. `results/`에 포함된 측정 기록은 2026-10-14 실행 당시의 자료로, 과거 통합 폴더 경로가 메타데이터에 남아 있을 수 있습니다.

| 관련 프로젝트 |
|---|
| [SIFT Matching and Homography](https://github.com/pod369/SIFT-Matching-and-Homography) |
| [Superpixel and Normalized Cut](https://github.com/pod369/Superpixel-and-Normalized-Cut) |
| [Edge and Contour Detection](https://github.com/pod369/Edge-and-Contour-Detection) |
| [Image and Video Effects](https://github.com/pod369/Image-and-Video-Effects) |
| [YOLO Hangul Text Detection](https://github.com/pod369/YOLO-Hangul-Text-Detection) |
