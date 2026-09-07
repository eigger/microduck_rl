# 처음 시작하기 — 맥에서 microduck_rl 돌려보기

GPU 없는 맥(Apple Silicon)에서 시뮬레이터를 띄우고, 학습된 정책을 받아서
움직여보고, 학습이 무엇인지까지 한 바퀴 도는 기록입니다. 2026-09-07에 실제로
한 순서 그대로 적었고, **막혔던 지점과 예상이 틀렸던 부분도 그대로** 남겼습니다.
매끄러운 설명서보다 그쪽이 다음 사람에게 쓸모 있어서입니다.

측정은 전부 Apple M4 Mac mini(성능코어 6 + 효율코어 4, 16 GB)에서 했습니다.

전제 지식은 없어도 됩니다. RL을 처음 보는 사람 기준으로 썼습니다.

---

## 0. 알고 시작하면 좋은 것

**학습(training)과 실행(inference)은 완전히 다른 일입니다.**

- **학습** — 정책을 만드는 과정. MuJoCo Warp가 돌리고 **CUDA GPU가 필요**합니다.
  맥에서도 되긴 하지만 느립니다 (5장).
- **실행** — 만들어진 정책(`.onnx`)을 돌려보는 것. **CPU MuJoCo로 충분**하고
  맥에서 잘 돕니다.

처음이면 **실행부터** 하세요. 학습은 그 다음입니다.

---

## 1. 환경 준비

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
uv sync
```

`uv`가 없으면 [먼저 설치](https://docs.astral.sh/uv/)하세요. `uv sync`는 torch,
mujoco 등 수 GB를 받으므로 몇 분 걸립니다. 끝나면 `.venv/`가 생깁니다.

> 맥에서 `uv sync`는 그냥 됩니다. warp가 CUDA 없이 빌드된 CPU 전용 휠로 깔릴 뿐,
> 설치가 실패하지는 않습니다.

---

## 2. 시뮬레이터 띄우기

### macOS에서는 `mjpython`을 써야 합니다 (첫 번째 함정)

MuJoCo 뷰어는 macOS에서 메인 스레드를 요구합니다. 그래서 문서에 적힌 대로

```bash
uv run duck-body        # ← 맥에서는 창이 안 뜹니다
```

이렇게 하면 **에러도 없이 조용히 헤드리스로 떨어집니다.** 로그에
`== no viewer (...); running headless`가 찍히는데 놓치기 쉽습니다.

맥에서는 `mujoco` 패키지가 같이 깔아주는 `mjpython`으로 실행하세요:

```bash
.venv/bin/mjpython -m mjlab_microduck.sim.body_server --keyframe HOME
```

창이 뜨고 덕이 서 있으면 성공입니다. **덕은 가만히 있습니다** — 이 프로세스는
몸통만 시뮬레이션하고, 관절을 움직이는 건 접속하는 데몬(`robotd`, 별도 레포)입니다.

---

## 3. 학습된 정책 받아서 움직이기

혼자 움직이는 걸 보려면 정책 파일이 필요합니다. 공식 세트가 Hugging Face에 있습니다.

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('policies', exist_ok=True)
for f in ['manifest.json','alpha_walking.onnx','alpha_stand.onnx','roulade.onnx']:
    shutil.copy(hf_hub_download('pollen-robotics/microduck-policies', f), 'policies/'+f)
    print('ok', f)
"
```

로그인 없이 받아집니다. 각 정책은 약 793 KB입니다.

`manifest.json`을 열어보면 이 정책들의 계약이 적혀 있습니다:

```json
"obs_len": 61, "action_len": 14, "control_hz": 50
```

**관측 61개를 받아서 행동 14개를 내놓는 신경망, 초당 50번.** 이게 정책의 전부입니다.

### 걷게 하기

```bash
.venv/bin/mjpython scripts/infer_policy.py \
  --walking policies/alpha_walking.onnx \
  --standing policies/alpha_stand.onnx \
  --roulade policies/roulade.onnx \
  --new-cmd-obs
```

터미널에 이렇게 뜨면 정상입니다:

```
Observation size: 61 (expected: 61)
Walking policy: loaded
Standing policy: loaded  (body pose: z=±30mm, pitch/roll=±30°)
roulade policy: loaded  (press R, auto-return after 2.0s)
```

조작:

| 키 | 동작 |
|---|---|
| ↑ / ↓ | 전진 속도 증감 |
| ← / → | 옆걸음 |
| A / E | 좌/우 회전 |
| SPACE | 정지 (속도 0 → standing 정책으로 자동 전환) |
| B | 몸통 자세 모드 |
| R | 룰라드(앞구르기) |
| P | 랜덤 밀기 — 넘어뜨려보기 |
| Q | 종료 |

### 함정 세 가지

**① `--new-cmd-obs`를 빼면 안 됩니다.** 이 플래그가 없으면 관측이 51차원으로
만들어져서 61차원 정책과 안 맞습니다. `manifest.json`의 `obs_len: 61`이 기준입니다.

**② 키는 뷰어 창이 아니라 터미널에 입력합니다.** 뷰어 창에서 키를 누르면
MuJoCo 자체 단축키(프레임 표시 등)가 먹습니다.

**③ 느린 속도 명령에서는 안 걷습니다.** 저는 처음에 `--lin-vel-x 0.15`로 띄우고
"정책이 고장났나" 했습니다. 로그가 이랬거든요:

```
achieved/cmd  fwd=+0.00/+0.15   trunk_z=115.5 mm    ← 제자리
```

0.3을 주니 걸었습니다:

```
achieved/cmd  fwd=+0.11/+0.30   trunk_z=117~120 mm  ← 보행 상하동
```

**"안 움직인다 = 고장"이 아닙니다.** 게이트가 시작되는 문턱이 있고, 그 아래에서는
서 있는 게 정상 동작입니다. 명령 0.30에 실제 0.11이 나오는 미달도 정상 범위입니다.

> 이게 이 레포의 AGENTS.md가 반복하는 원칙의 첫 실전 사례입니다 —
> **"실패했다"고 판단하기 전에 먼저 측정하라.**

---

## 4. 학습이 뭔지 — 최소한의 개념

**정책(Policy)** 이 학습의 대상입니다. 숫자 61개 → 숫자 14개 신경망 하나.
`.onnx`로 내보내 실기에 올리는 것도 이겁니다.

**관측 61개** — 정책이 보는 세상. 48개는 자기 몸 상태(관절 14개의 각도·속도,
중력이 몸통 기준 어느 방향인지, 직전에 자기가 뭘 했는지), 13개는 **명령**
(속도 3 + 머리자세 4 + 몸통자세 6). **카메라도 지도도 없습니다.** 앞이 안 보이는
채로 관절 감각만으로 걷습니다.

**행동 14개** — 서보 14개의 목표 각도. 초당 50번.

**에피소드** — 한 판. 넘어지거나 시간이 다 되면 리셋.

**리워드** — 매 스텝 점수 하나. **이게 유일한 학습 신호입니다.**

**Iteration** — 로그의 `Learning iteration` 한 줄. 4096마리가 동시에 24스텝씩,
즉 **98,304 스텝**을 모아서 신경망을 한 번 업데이트합니다.

가장 중요한 감각 하나:

> **RL에게 "어떻게"를 가르치지 않습니다. "무엇이 좋은지"만 말하고, 방법은 정책이
> 발명합니다.**

걷기 정책에 "왼발 들고 오른발 들고"를 알려준 적이 없습니다. 실제 리워드는 이렇습니다:

| 항 | 가중치 |
|---|---:|
| `track_linear_velocity` | +2.0 |
| `track_angular_velocity` | +2.0 |
| `air_time` (발이 공중에 머무는 시간) | +3.0 |
| `upright` | +2.0 |
| `pose` | +1.0 |
| `foot_slip` | −0.1 |
| `action_rate_l2` | −0.1 → −1.0 (커리큘럼) |

보행은 이 조건들을 만족시키려다 **발명된** 겁니다.

이게 힘이자 함정입니다. 조건을 만족하는 다른 방법이 있으면 정책은 그걸 합니다.
AGENTS.md에 "엉덩이로 뛰기", "머리로 삼각대 서기" 같은 실패가 적혀 있는 이유입니다.

---

## 5. 맥에서 학습 돌려보기

### 스모크 테스트 — 이건 꼭 하세요

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 --gpu-ids None
```

**13초**면 끝납니다. `--gpu-ids None`이 CPU 모드이고, `WANDB_MODE=offline`이 없으면
wandb 로그인에서 막힙니다.

AGENTS.md는 긴 학습 전에 **항상** 이걸 먼저 돌리라고 합니다. 설정 오류의 95%가
여기서 잡히고, 맥에서는 공짜입니다. GPU가 있어도 이 습관은 유지하세요.

### 여러 코어 쓰기

맥에서 학습이 느린 이유는 **warp의 CPU 백엔드가 싱글스레드**여서입니다.
10코어 중 1개만 씁니다.

```bash
uv run scripts/train_cpu_dist.py --workers 8 Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 192 --agent.max_iterations 1000
```

총 env 1536 고정으로 잰 결과:

| 워커 | envs/워커 | iteration | speedup |
|---:|---:|---:|---:|
| 1 | 1536 | 45.7 s | 1.00× |
| 2 | 768 | 24.4 s | 1.87× |
| 4 | 384 | 14.9 s | 3.06× |
| 6 | 256 | 11.9 s | 3.83× |
| 8 | 192 | 10.8 s | 4.23× |

**주의: `--env.scene.num-envs`는 워커당 값입니다.** 실제 배치는 그 N배입니다.

자세한 배경은 [cpu-training.md](cpu-training.md)에 있습니다.

### 맥 GPU는 못 씁니다

플래그가 없는 게 아니라 **컴파일러 백엔드가 없습니다.** warp는 LLVM(CPU)과
NVRTC(CUDA) 두 타겟으로만 코드를 뽑고, Metal 백엔드가 아예 존재하지 않습니다.
torch는 MPS를 쓰지만 torch가 담당하는 건 전체의 2.2%뿐이라 의미가 없습니다.

---

## 6. 학습 결과 읽는 법

`max_iterations`는 5만으로 잡혀 있는데 **사실상 무한대**입니다. 자동 종료 기준도,
수렴 판정도 없습니다. 사람이 보고 판단합니다.

wandb에서 매 iteration 확인할 것:

- **mean reward가 오르는가**
- **episode length가 태스크에 맞게 움직이는가**
- **모든 페널티 항이 ≤ 0인가** — 부호 실수를 잡는 무오류 검사입니다. 자기부정형
  페널티에 음수 가중치를 주면 위반에 대한 *보상*으로 뒤집혀서 정책이 그걸 farming합니다
- **주 임무 항이 실제로 자라는가** — 전체 리워드는 regularizer만으로도 오릅니다.
  그래프는 예쁜데 정작 하려던 동작은 안 나오는 경우가 흔합니다

경험적 예산: **보행 4000~6000 iteration, 단순 episodic 트릭 ~1000 iteration**
(4096 envs 기준). 체크포인트는 250 iteration마다 저장됩니다.

그리고 마지막 기준은 숫자가 아닙니다. **영상을 보세요.** sim 지표가 통과해도
눈으로 보면 실패인 경우가 있습니다. 보고할 때도 "된다!"가 아니라
"구르긴 하는데 3번에 1번 얼굴로 착지한다" 식으로 적으세요.

---

## 7. 본 학습은 어디서

맥으로 4096 envs × 4000 iteration을 하면 8워커로도 **하루 이상** 걸리고 그동안
맥을 못 씁니다. GPU 한 장이면 1~2시간입니다.

이 레포는 GPU 없는 사람을 위해 Hugging Face Jobs 경로를 준비해뒀습니다:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs
```

**무료가 아닙니다.** 크레딧 잔액이 있어야 하고 초 단위 종량제입니다
(무료인 ZeroGPU는 Spaces 전용이라 Jobs에는 해당 없음).

| flavor | 시간당 |
|---|---:|
| `l4x1` (기본값) | $0.80 |
| `a10g-large` | $1.50 |
| `a100-large` | $2.50 |

정책 하나에 대략 **$2~10** 정도로 보면 됩니다. 정확한 시간은 실제로 재봐야 알 수
있으니, 긴 run을 던지기 전에 20 iteration짜리 짧은 job으로 먼저 측정하세요.
자세한 건 [scripts/hf/README.md](../scripts/hf/README.md).

---

## 8. 다음 단계

새 동작을 학습시키고 싶다면 순서를 권합니다.

**1단계 — 파이프라인 한 바퀴.** 새 태스크 말고, 기존 걷기의 리워드 가중치 하나만
바꿔서 돌려보세요 (`air_time` 3.0 → 6.0 같은). 목적은 결과가 아니라
**스모크 테스트 → 학습 → 영상 확인 → 비교** 사이클을 몸에 익히는 겁니다.

**2단계 — 쉬운 동작.** 제자리 리듬 동작처럼 짧은 episodic 트릭. 기존 standup /
roulade 템플릿 위에 얹으면 됩니다. 예산 ~1000 iteration.

**3단계 — 정해진 동작.** 특정 안무처럼 *정확한 궤적*이 필요하면 리워드 설계가
아니라 **모션 트래킹**이 맞습니다. mjlab에 이미 있습니다(`mjlab/tasks/tracking`).
참조 동작을 `.npz`로 주면 되고, microduck용 설정은 직접 써야 합니다
(예제는 Unitree G1 하나뿐). 참조 동작은 [scripts/crouch_pose_editor.py](../scripts/crouch_pose_editor.py)로
키포즈를 잡아 보간해서 만들 수 있습니다.

> 여기서 헷갈리기 쉬운 점: AGENTS.md는 "키프레임/웨이포인트 리워드"를 금지합니다
> — 정책이 웨이포인트에 눌러앉기 때문. 그런데 **모션 트래킹은 다릅니다.** 목표가
> 시계를 따라 계속 전진해서 눌러앉으면 보상이 끊깁니다. 같은 "정해진 동작
> 따라하기"인데 구조가 달라 착취가 안 됩니다.

새 환경을 만들기 전에 **AGENTS.md를 꼭 읽으세요.** 리워드 설계 함정들이 전부
누군가 실제로 당하고 적어둔 것들입니다.

---

## 부록 A — 오늘 측정한 수치

Apple M4 Mac mini (6P + 4E, 16 GB), `Mjlab-Velocity-Flat-MicroDuck`

| 항목 | 값 |
|---|---|
| 스모크 테스트 (64 envs × 5 iter, CPU) | 13 초 |
| 단일 프로세스 512 envs | 14.6 s/iter |
| 단일 프로세스 1536 envs | 45.7 s/iter |
| 8워커 1536 envs | 10.8 s/iter (4.23×) |
| CPU 점유 (단일 프로세스) | ~100% = 코어 1개 |
| Collection / Learning 비중 | 97.8% / 2.2% |
| 워커당 메모리 (512 envs) | 1.2~1.4 GB |
| 걷기 정책 실측 속도 | 명령 0.30 → 실제 0.11 m/s |
| 독립 프로세스 3개 동시 | 각 16.7 s/iter (단독 14.6) |

---

## 부록 B — 막혔던 지점 전부

다음 사람이 같은 데서 시간 쓰지 않도록.

| 증상 | 원인 | 해결 |
|---|---|---|
| 뷰어 창이 안 뜨는데 에러도 없음 | macOS는 뷰어에 메인 스레드 필요 | `.venv/bin/mjpython`으로 실행 |
| 정책은 로드됐는데 덕이 안 움직임 | 0.15 m/s가 게이트 문턱 아래 | 0.3 이상 주기 |
| 키보드가 안 먹음 | stdin이 TTY가 아님 | 터미널에서 직접 실행 (백그라운드 X) |
| 관측 차원 불일치 | `--new-cmd-obs` 누락 | 61D 정책에는 필수 |
| 학습이 wandb 로그인에서 멈춤 | wandb 기본이 온라인 | `WANDB_MODE=offline` |
| 학습이 GPU를 찾다 실패 | 맥에 CUDA 없음 | `--gpu-ids None` |
| 10코어 중 1개만 씀 | warp CPU 백엔드가 싱글스레드 | `scripts/train_cpu_dist.py --workers 8` |

### 예상이 틀렸던 것

**"효율코어가 동기 all-reduce의 발목을 잡을 테니 6워커가 최적일 것"** — 틀렸습니다.
8워커가 6워커보다 빨랐습니다(10.8초 vs 11.9초). 효율코어는 페이스를 떨어뜨리는
대신 그냥 일을 덜 받고도 보탰습니다.

측정하기 전에는 모릅니다. 이 레포의 AGENTS.md가 **"이론화하기 전에 측정하라"**를
가장 앞에 두는 이유입니다.
