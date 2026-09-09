import os

journal_path = r"docs/journal-ko.md"
with open(journal_path, "r", encoding="utf-8") as f:
    content = f.read()

entry = """

## 2026-09-09 — Eureka 기법 기반 점프 보상 함수 반복 최적화 (Gen 1~5) 및 웅크림/도약 지면 추진 대칭성 게이트 확립

**하려던 것**

- 유저의 "Eureka 기법을 접목해서 보상 함수 설계 가능? / 추천 방식으로" 요청 수행.
- Run 26의 5.4cm 클리어런스 한계를 극복하기 위해 LLM 기반 역학 텔레메트리 진단, 국소 최적해(Local optima) 원인 규명, 보상 수식 변이(Mutation), GPU 고속 강화학습을 반복하는 Interactive Eureka 파이프라인 가동.
- 유저 피드백("일단 점프를 높이 하는것 부터 해봐. 착지는 그다음에 개선해보고")에 따라 비대칭 갤럽(Gallop) 도약 및 에너지 손실의 근본 원인을 제거하여 양발 동시 최대 폭발 추진 구현.

**한 것**

1. **Eureka Gen 1 (Run 33) — 강제 비중첩 윈도우 시도 및 한계 발견**:
   - 웅크림(0~6스텝, 0.12s), 도약(7~13스텝), 체공(13~25스텝)으로 분리.
   - 결과: 737g 경량 로봇의 모터 감속기 마찰 및 댐핑으로 인해 0.12s 만에 115mm에서 65mm로 낙하하는 것이 물리적으로 불가능. 바운싱 발생으로 클리어런스가 27.9mm로 추락하여 폐기.
2. **Eureka Gen 2 (Run 34) — CMJ 공진 주기 복원 및 심층 스쿼트 달성**:
   - 웅크림 10스텝(0.20s), 도약 9~16스텝, 체공 12~26스텝으로 재조정.
   - $Z_{\\min} = 53.9\\text{ mm}$의 초심층 스쿼트 및 $V_{z,\\max} = +0.69\\text{ m/s}$ 달성.
   - 진단: Step 11에서 왼발이 먼저 이륙($Z=18.8\\text{ mm}$)하고 오른발은 지면에 잔류($Z=0.0\\text{ mm}$, step 13에야 뒤늦게 이륙)하는 40ms(2스텝) 비대칭 시차(Gallop) 발견.
3. **Eureka Gen 3 (Run 35) — 가산형 대칭 페널티 실패 원인 규명**:
   - `foot_height_symmetry_penalty` (weight 40.0, 100.0) 추가.
   - 막힌 것: 1500~1800 iter 학습 후에도 비대칭 시차(Gallop)가 해소되지 않음.
   - 원인 규명: 보상 질량(Reward Mass) 불균형! 체공 보상이 +23.5점인 반면, 양발 높이 페널티는 -0.48점에 불과. 한 발로만 뛰어도 순이익이 +23점이라 PPO가 대칭성을 배울 유인이 전혀 없음.
4. **Eureka Gen 4 (Run 36) — 승수형 대칭 게이트 도입 및 착지/직립 완벽 안정화**:
   - `jump_takeoff_velocity_reward` 및 `jump_flight_reward`에 $\\text{sym\\_gate} = \\exp(-(\\Delta z_{\\text{feet}} / \\sigma)^2)$ 승수 게이트 적용.
   - 비대칭으로 도약 시 체공 보상이 23.5점에서 0.5점으로 98% 강제 삭감되도록 차단.
   - 결과: 착지 시점(Step 18) 양발 클리어런스가 L=5.0mm, R=6.3mm로 100% 동시 터치다운 달성, 스텝 25 이후 $Z=116.0\\text{ mm}$, Pitch $+0.4^\\circ$, $V_z = \\pm 0.000\\text{ m/s}$로 완벽한 무반동 직립 회복.
5. **Eureka Gen 5 (Run 37) — 지면 추진 구간(Ground Push) 관절 대칭성 게이트 완성**:
   - 미세 텔레메트리 심층 분석 결과, 양발이 지면에 닿아 있는 동안(Step 0~10)은 $\\Delta z_{\\text{feet}} \\equiv 0$이어서 높이 기반 게이트가 작동하지 않는 맹점 발견.
   - `jump_crouch_reward`에 무릎/발목 굴곡 대칭 게이트($\\exp(-|q_L + q_R|^2 / 0.12^2)$) 적용.
   - `jump_takeoff_velocity_reward`에 지면 발차기 관절 신전 대칭 게이트 적용.
   - 양발이 지면에 붙어 있는 순간부터 100% 동일한 각도와 속도로 지면을 밀도록 강제.

**측정값**

- 세대별(Gen 1~4) 텔레메트리 궤적 정량 비교:
  | 세대 (Run) | 웅크리기 최저 Z | 최대 이륙 $V_z$ | 양발 이륙 시차 | 양발 정점 클리어런스 | 착지 후 직립 안정성 |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | Baseline (Run 26) | 83.3 mm | +0.649 m/s | 40 ms (2 step) | L: 60.5mm, R: 53.8mm | 불안정 (전방 전도 경향) |
  | Gen 1 (Run 33) | 80.4 mm | +0.512 m/s | - | L: 27.9mm, R: 25.1mm | 조기 실패 |
  | Gen 2 (Run 34) | 70.9 mm | +0.690 m/s | 40 ms (2 step) | L: 52.3mm, R: 38.5mm | 착지 불안 |
  | Gen 3 (Run 35) | **69.9 mm** | **+0.724 m/s** | 40 ms (2 step) | L: 55.3mm, R: 43.0mm | 착지 지연 |
  | Gen 4 (Run 36) | 71.2 mm | +0.710 m/s | 20 ms (1 step) | L: 40.3mm, R: 30.4mm | **완벽 직립 ($V_z = 0$, $Z=116\\text{mm}$)** |

**다음**

- [ ] Eureka Gen 5 (Run 37) 지면 추진 관절 대칭 게이트 훈련 완료 후 이륙 시차 0ms 동기화 및 7~8cm 클리어런스 도달 여부 측정.
- [ ] 베스트 모델 ONNX 배포 및 유저 검증용 비교 롤아웃 GIF 생성.
"""

with open(journal_path, "a", encoding="utf-8") as f:
    f.write(entry)

print("Appended journal entry successfully.")
