import os

entry = """
## 2026-09-09 — 점프 높이 극대화 및 공중 무릎 접기(Tuck) 분석, 5.4cm 실측 최고 높이 정책(Run 26) 배포

**하려던 것**

- 유저 피드백("아직도 별로 잖아. 일단 점프를 높이 하는것 부터 해봐. 착지는 그다음에 개선해보고", 12:53 GIF) 반영.
- 착지 완충/안정성은 부차적인 것으로 두고, 시각적으로나 물리적으로나 최대로 높이 솟구치는 점프(Clearance 및 Peak Z 극대화) 구현 및 실측 검증.

**한 것**

1. **모터 역기전력(Back-EMF) 및 수직 상승 한계 분석**:
   - XL330-M288 7.4V 전압 하에서 무부하 최대 각속도는 약 20 rad/s.
   - 링크 길이 $l_{\\text{thigh}} = 4.2\\text{cm}, l_{\\text{shin}} = 4.2\\text{cm}$ 구조상 역기전력 포화로 인한 이륙 수직 속도는 $V_{z, \\max} \\approx 0.65 \\sim 0.68\\text{ m/s}$에 물리적으로 제한됨 ($\\Delta z_{\\text{ballistic}} = \\frac{V_z^2}{2g} \\approx 2.4\\text{cm}$, 몸통 정점 $146 \\sim 149\\text{ mm}$).
   - 따라서 로봇 및 사람의 고도 점프(High jump)는 이륙 직후 공중에서 다리를 몸통 쪽으로 빠르고 강하게 접어 올리는 **공중 무릎 모으기(Knee Tuck)** 동작이 발 클리어런스를 결정짓는 핵심 기제임을 규명.
2. **실험별 도약 성능 비교 분석 (`scratch/compare_runs.py`)**:
   - **Run 30 (`model_2395.pt`)**: 이륙 $V_z = +0.675\\text{ m/s}$, 몸통 정점 $123.8\\text{ mm}$, 그러나 공중에서 다리를 완전히 펴고 있어 양발 클리어런스는 $11.6\\text{ mm}$ (1.1 cm)에 불과.
   - **Run 31 (`model_1647.pt`)**: 이륙 $V_z = +0.541\\text{ m/s}$, 몸통 정점 $119.3\\text{ mm}$, 양발 클리어런스 $7.3\\text{ mm}$ (0.7 cm).
   - **Run 26 (`model_1498.pt`)**: 이륙 $V_z = +0.649\\text{ m/s}$, 몸통 정점 **$146.0\\text{ mm}$**, 공중 양 무릎 1.04 rad 턱킹으로 양발 클리어런스 **$53.8\\text{ mm}$ (5.4 cm)** 달성.
3. **최종 정책 배포**:
   - 5.4cm의 최고 체공 높이를 기록한 Run 26 모델(`model_1498.pt`)을 ONNX로 변환하여 `policies/jump.onnx`로 배포 완료.
   - `run_simulator.bat` 실행 시 J 키 입력으로 즉시 5.4cm 고도 점프 동작 가능하도록 설정 완료.

**측정값**

- 점프 정책별 정량 비교 (BAM m6 50 Hz CPU 시뮬레이션):
  | 정책 | 웅크리기 최저 Z | 최대 이륙 $V_z$ | 몸통 정점 Z | 최대 양발 클리어런스 | 체공 시간 |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | Run 26 (`model_1498.pt`) | **83.3 mm** | **+0.649 m/s** | **146.0 mm (+29.4 mm)** | **53.8 mm (5.4 cm)** | **0.16 s** |
  | Run 30 (`model_2395.pt`) | 78.5 mm | +0.675 m/s | 123.8 mm (+7.2 mm) | 11.6 mm (1.2 cm) | 0.10 s |
  | Run 31 (`model_1647.pt`) | 80.3 mm | +0.541 m/s | 119.3 mm (+2.7 mm) | 7.3 mm (0.7 cm) | 0.08 s |

**다음**

- [ ] 유저가 `run_simulator.bat`를 실행하여 3D 뷰어에서 J 키를 눌러 5.4cm 폭발적 고도 점프 확인.
- [ ] 유저 확인 후 착지 시 미세 제어 개선 진행.
"""

with open('docs/journal-ko.md', 'a', encoding='utf-8') as f:
    f.write(entry)
print('Successfully appended to docs/journal-ko.md!')
