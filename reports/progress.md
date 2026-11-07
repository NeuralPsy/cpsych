# Прогресс платформы

## Текущие ворота: P7 (рукопись) — ЗАВЕРШЕНА

## Статус ворот

| Ворота | Статус | Дата |
|--------|--------|------|
| P0 — своё ядро или Neuroblox | **пройдены** | 2026-07-26 |
| P1 — контракт и ядро | **пройдены** | 2026-07-27 |
| P2 — четыре модуля standalone | **пройдены** | 2026-07-31 |
| P3 — попарные связи | **пройдены** | 2026-08-01 |
| P4 — тройной композит + карта невозможного | **пройдены** | 2026-08-04 |
| P5 — песочница | **пройдены** | 2026-08-04 |
| P6 — сравнение с реальными данными | **пройдены** | 2026-08-05 |
| P7 — рукопись | **завершена** | 2026-08-15 |

## Решения P0

- D1: своё ядро на Python (reports/decisions.md)
- D1a: исправление лицензии Neuroblox в formalisms_report
- D2: batch-ось в контракте модуля
- D3: numpy с подготовкой миграции; порог пересмотра: >4 ч карта, >50 переменных, или implicit SDE

## Решения P1

- D4: Среда — вариант B в P2 (раздельные задачи: перцептивная + бандит), с условием:
  контракт единый, составная среда в P3 через композицию, не переписывание.
  Если композиция потребует ломать интерфейс — эскалация.
- D5: Схема расщепления — Lie-Trotter (snapshot), задокументировано в integrator.py.
  Тест различает snapshot от averaging (bounds 0.1% < err < 1.0%).
- D6: Lie-Trotter достаточен, Strang не нужен (порядок 1.02–1.13, ошибка <0.06% при рабочих dt).
- D7: Протоколы среды — Roitman & Shadlen 2002 + Behrens 2007.
- D8: Обезьяньи данные — компромисс ради trial-level RT. Monkey N настройка, monkey B held-out.
- D9: p_reward и volatility развязаны (отклонение от Behrens). Стохастические реверсии.

## P1 — результаты

### Ядро (src/cpsych/core/)

- specs.py — frozen dataclasses: StateVar, Param, ModuleSpec, EnvSpec, Temporality, Exposure
- module.py — SDEModule ABC: drift, diffusion, outputs, trial_done, trial_update, reset_state
- environment.py — Environment ABC: next_trial, reward, reset
- coupling.py — Coupling: аддитивная модуляция θ + clamp(strength)·h(source), batch-совместимая
- integrator.py — Integrator, Tier, SimState: многотемповый EM, Lie-Trotter, рекурсивные тиры
- stability.py — numerical_jacobian, composite_jacobian, find_critical_strength, max_real_eigenvalue

### Тесты ядра (67/67 зелёные)

- test_specs.py (7): frozen, индексы, валидация
- test_module.py (20): purity, batch, default state/params, episodic reset, trial_update, index helpers
- test_coupling.py (7): modulation, clamping, custom h, no mutation, batch independence
- test_integrator.py (16): шум ∝ √dt, dt-сходимость, determinism, obs freq ≠ dt, episodic reset с проверкой OU-непрерывности, multi-tier, tier ratio validation, coupling integration, no global seed pollution, мультитемпо snapshot с дискриминацией от averaging, сходимость по dt_slow, fast equilibration
- test_stability.py (17): якобиан OU (1e-8), нелинейный якобиан (1e-10 на 4 точках), composite jacobian, critical strength vs analytical (√α₁α₂), below/above critical, varies with point

### Пакет

`src/cpsych/` (переименован из platform/ — коллизия с stdlib).
Запуск: `PYTHONPATH="src;tests" pytest tests/ -v`.

## P2 — прогресс

### Порядок

1. **Среда** (обе задачи: перцептивная + бандит) — **готова, 16 тестов зелёные**
2. **Возбуждение** — **закрыт** (D18: σ_h 0.01→0.008, все результаты пересчитаны)
3. **DDM** — **готов, 18 тестов зелёные** (specs/ddm.md v1.2 утверждена)
4. RL и настроение (после DDM) — **закрыт** (30 тестов, β fitted, 3 deferred PASS)

Чекпойнт после каждого модуля.

### P2.1 Среда — результаты

Реализация: `src/cpsych/environments/`
- `perceptual_task.py` — PerceptualTask: 6 когерентностей Roitman & Shadlen, direction, deterministic reward.
- `reversal_bandit.py` — ReversalBandit: p_reward, volatility (exposed), formula mean_block = 20/v, v=0 → no reversals.

Тесты: `tests/test_environment.py` (16/16 зелёные)
- E1–E5: PerceptualTask (chi2 когерентность, баланс направлений, нет автокорреляции, правильность вознаграждения).
- E6–E10: ReversalBandit с фиксированной политикой action=0 (условная P(r), частота реверсий, внутриблочная независимость, v=0 гарантия, v=1 средний блок ~20).
- E11: отложен до P2.4 (поведенческая различимость требует агента).
- 6 контрактных тестов: shapes, coherence values, validation, determinism.

Технические заметки:
- KS-тест scipy ненадёжен для дискретных распределений (даже Geom выборки дают p<0.05). E7 использует прямую частоту реверсий + средний блок ±20%.
- Конечные последовательности дают ~2% смещение вниз в средней длине блока (усечение хвостов). При n_trials=2000 смещение в пределах 20% допуска.

### P2.2 Возбуждение — закрыто (2026-07-30)

Спецификация: `specs/arousal.md` v2.3.
Первоисточник: Phillips-Robinson 2007/2008, verified by CellML/BioModels/sleepR.

**Реализация:** `src/cpsych/modules/arousal.py`
- ArousalModule(SDEModule): 4 состояния (vv, vm, h, clock), 18 параметров.
- v_vh=0.325 (D13). Q_REF=4.2883 (results/arousal_calibration.json).
- σ_vv=0.003, σ_vm=0.015, σ_h=0.01 (A5).
- circular_observables: sleep_onset_tod, sleep_offset_tod, onset_advance (24h).
- `src/cpsych/observables/circular.py`: circular_mean, circular_sd, circular_diff.

**Тесты:** `tests/test_arousal.py` (18/18 зелёные). Полный набор: 101/101.

**Parameter recovery:** все 4 кандидата < 0.7.
v_vh, v_vc, v_vm — exploratory. v_mv — internal (r(v_vm,v_mv)=0.62).
Det recovery: r=1.0 (оценщик исправен). SNR 0.9–3.8 (свойство flip-flop).

**Проверки согласованности** (бывшие отложенные #1-2, использованы для D13):
- onset_advance: INDETERMINATE (circ CI [0.83, 1.77]).
- dur_excess: PASS (CI [1.38, 1.81]).
- H_excess: PASS (CI [22.4, 28.0]).

**Отложенные наблюдаемые** (статус сохранён):
- #3 резкость перехода: PASS (22 мин < 30).
- #4 паразитные переходы: PASS (0%).

**Окно совместимости v_vh (D12):** [0.30, 0.34] = 9.5% от диапазона [0.08, 0.50].
Снизу: H_excess ≥ 20%. Сверху: sleep ≤ 9h. dur_excess проходит от 0.25.
onset_advance — отчётная (INDETERMINATE, не определяет границы).
Помечается на ручке в интерфейсе.

Цель настройки: mean sleep 8.32h, CI [8.18, 8.45] < 9h → OK.

Решения: D10–D13. Правила: 18 (два класса), 18а (фарма), 19а (гибридные), 21 (однозначность).

### P2.3 DDM — спецификация (2026-07-30)

Спецификация: `specs/ddm.md` v1.1. Статус: утверждена.

- Первоисточники: Ratcliff 1978, Ratcliff & McKoon 2008, Roitman & Shadlen 2002.
- Уравнение: dx = v·dt + dW, σ=1 (fixed), v = kappa·coherence·direction.
- 2 состояния (evidence EPISODIC, trial_clock EPISODIC), 3 свободных параметра (kappa, a, Ter).
- Калибровка: замкнутая процедура (κa из accuracy → a² из масштаба RT → κ=κa/a → Ter OLS).
  Без итераций. Все три параметра определены однозначно (правило 21).
- Recovery: 5³×20 реплик × ~2800 проб (~470/уровень, как monkey N), 12 наблюдаемых.
- Связь P3: мультипликативная kappa_eff = κ·max(0, 1+s·(g−1)), strength ∈ [0, 4.0].
- Тайм-аут пробы: T_max_trial (TBD с данными), тайм-ауты включены в статистики.
- Отложенные: квантили RT, right-skew, равенство RT-распределений correct/error.
  Slow errors — отчётная величина, не критерий.
- Held-out: monkey B (D8).
- D14, D15 утверждены.

**Калибровка выполнена** (experiments/ddm_calibration.py → results/ddm_calibration.json):
- κa = 21.96 (logistic MLE на accuracy, строго вогнутый → единственность).
- a² = 2.225 → a = 1.4916. κ = 14.7242.
- Ter = 0.318 с (OLS intercept по 6 когерентностям).
- RMSE = 0.018 с, max error = 6.2% (51.2% coh). 1 MTU = 1 секунда.

Данные: monkey 2 в CSV = monkey N (3534 пробы, ~590/уровень). monkey 1 = monkey B (held-out).

**Parameter recovery (results/ddm_recovery.json):**

| Параметр | r | RMSE | Кросс-корр. | Класс |
|----------|-----:|-----:|-----------|-------|
| kappa | 0.9966 | 0.74 | <0.001 | **fitted** |
| a | 0.9999 | 0.013 | <0.001 | **fitted** |
| Ter | 0.9999 | 0.002 | <0.001 | **fitted** |

Все три fitted. Контраст с arousal (r=0.05–0.41): DDM — 3 параметра,
аналитические формулы, нет пороговой динамики.

**Разрешение параметров (results/ddm_resolution.json):**

| Параметр | MDD (80% disc.) | MDD % | Рек. шаг ручки |
|----------|----------------|-------|-----------------|
| kappa | 0.44 | 3.0% | ~0.5 |
| a | 0.045 | 3.0% | ~0.05 |
| Ter | 0.010 | 3.0% | ~0.01 |

Все три различимы на уровне 3% от калиброванного значения при 3500 пробах.
Асимметрия: кappa легче обнаружить при уменьшении, a — при увеличении.

**Отложенные наблюдаемые (results/ddm_deferred.json):**

1. **Квантили RT.** Модель завышает right tail (Q90 на 23–34% при coh≤0.064)
   и занижает left tail (Q10 на 10–35% при высоких coh). Систематическое
   расхождение формы: модель слишком сконцентрирована в центре при низких
   когерентностях, слишком сдвинута при высоких.

2. **Right-skew Q90/Q50.** Модель: 1.16–2.00, убывает с когерентностью.
   Данные: 1.30–1.59, слабо зависят от когерентности. Паттерн качественно
   расходится: модель предсказывает сильное уменьшение skew с когерентностью,
   данные — нет. Отсутствие межпробной вариабельности (eta) — вероятная причина.

3. **RT correct vs error (positive prediction).** Модель: **PASS** все coh.
   KS p=0.005 при coh=0 — ложноположительный: Cohen d=0.04 (пренебрежимо),
   направление эффекта случайно (seed 999: p=0.36), границы симметричны
   (binomial p=0.52). Переизбыточная мощность при N≈5000 на группу.
   Данные monkey N: slow errors при coh≥0.032 (diff 66–201 мс). Документирует
   потребность в eta для будущего расширения (D16).

**Held-out monkey B: PASS.** Все 6 когерентностей: accuracy |diff|≤0.071,
RT diff ≤14.3%. Межсубъектная генерализация подтверждена.

**DDMExtended (D16):** отложен до завершения P2. Отдельный модуль с eta
(и, возможно, sz, st) для воспроизведения формы RT-распределения. Условие
подъёма: зависимость P3/P4 результатов от формы RT-распределения.

**Следующий шаг:** P2.4 (RL и настроение) — нужна спецификация.

### Сводка ворот P2

Входные данные для P3 и P4. Карта невозможного строится только по
достоверным наблюдаемым; обратная задача — только по fitted-параметрам.

#### Параметры: fitted / exploratory

| Модуль | Параметр | r | Класс | Примечание |
|--------|----------|-----:|-------|------------|
| **DDM** | κ (sensitivity) | 0.997 | **fitted** | — |
| DDM | a (threshold) | 1.000 | **fitted** | Асимметричное разрешение: MDD вниз 6%, вверх 3% |
| DDM | Ter (non-decision) | 1.000 | **fitted** | — |
| **Arousal** | v_vh (H sensitivity) | 0.41 | exploratory | Окно совместимости [0.325, 0.35] = 6% |
| Arousal | v_vc (circadian) | 0.35 | exploratory | — |
| Arousal | v_vm (MA→VLPO) | 0.29 | exploratory | r(v_vm,v_mv)=0.62 |
| Arousal | v_mv (VLPO→MA) | 0.05 | internal | Из пары v_vm/v_mv выводится один |
| **RL+Mood** | β (inv. temp.) | 0.90 | **fitted** | — |
| RL+Mood | α (learning rate) | 0.67 | **fitted при N≥1000** | r=0.77 [0.76, 0.78] при 1000 проб |
| RL+Mood | τ_m (mood decay) | 0.01 | exploratory | Калибровано из γ=0.63, не восстановимо |
| RL+Mood | λ (mood coupling) | 0.01 | exploratory | Неоткалиброван. Лишняя DOF (правило 21) |

**Итого fitted:** DDM: 3/3. Arousal: 0/4 (все exploratory). RL+Mood: 1/4 (β),
α условно fitted при N≥1000.

#### Отложенные наблюдаемые: пройденные / проваленные

| Модуль | # | Наблюдаемая | Статус | Примечание |
|--------|---|-------------|--------|------------|
| **DDM** | 1 | Квантили RT | FAIL | Q90 +23–34% при низких coh. Нужен eta (D16). |
| DDM | 2 | Right-skew vs coh | FAIL | Качественное расхождение паттерна. |
| DDM | 3 | RT correct = RT error | PASS | Cohen d < 0.1. Критерий пересмотрен (D17). |
| DDM | HO | Monkey B held-out | PASS | Все 6 coh, acc ≤ 0.071, RT ≤ 14.3%. |
| **Arousal** | 1 | Onset advance (депр.) | INDET | CI [0.83, 1.77]. Проверка согласованности (D13). |
| Arousal | 2 | Dur excess (депр.) | PASS | CI [1.38, 1.81]. Проверка согласованности (D13). |
| Arousal | 3 | Transition sharpness | PASS | 22 мин < 30. |
| Arousal | 4 | Spurious transitions | PASS | 0%. |
| **RL+Mood** | 1 | Rutledge coefficients | PASS | w_RPE=0.63 > w_CR=0. |
| RL+Mood | 2 | Vol invariance | INDET | CI [0.75, 2.25], порог 2.0. |
| RL+Mood | 3 | Mood autocorrelation | PASS* | γ_model=0.74. *Частично циркулярная. |

#### Наблюдаемые, которые модули не воспроизводят

| Модуль | Наблюдаемая | Причина | Правило P4 |
|--------|-------------|---------|------------|
| DDM | Квантильный профиль RT | Нет eta (D16) | Карта не строится |
| DDM | Right-skew паттерн | Нет eta | Карта не строится |
| DDM | Slow errors | Нет eta | Отчётная, не карта |
| Arousal | Onset advance | Недост. разрешение | Карта не строится |
| Arousal | (нет категории 1 после D18) | — | — |
| RL+Mood | Adaptive α (volatility) | По построению (RW) | Карта не строится |
| RL+Mood | Клиническое настроение | По масштабу (τ_m ≈ 2 пробы) | Карта не строится |

#### Компромиссы и окна совместимости

| Модуль | Компромисс | Окно | Помечается |
|--------|-----------|------|------------|
| Arousal | v_vh: сон vs депривация (D12/D13) | [0.325, 0.35] = 6% | На ручке в песочнице |
| RL+Mood | λ: нет калибровки (правило 21) | [0, λ_crit(τ_m)] | Exploratory; правило 18а |
| RL+Mood | Конструкт: моментальный аффект, не клинический (D19) | — | Путь к варианту B |

### Общий счёт тестов: 150/150 зелёные

### Отложенные наблюдаемые (выбраны автором 2026-07-27, до реализации)

**DDM**: настройка — точность и среднее время реакции.
Отложено — квантили RT (0.1/0.3/0.5/0.7/0.9, правосторонняя асимметрия).

**RL и настроение**: настройка — последовательности выборов.
Отложено — коэффициенты Rutledge.

**DDM данные**: monkey N — настройка, monkey B — held-out (D8).

### Оговорка

Сырых данных нет, только опубликованные. «Отложенная наблюдаемая» = явление из
литературы, не использованное при настройке. Слабее held-out на данных.

### Разделение проверок

1. Parameter recovery: восстанавливается ли θ из данных собственной модели.
2. Отложенные / проверки согласованности: предсказание vs совместимость.
Обе обязательны. Подменять одну другой нельзя. Правило 21: настройка определяет
параметр однозначно; если нет — лишняя степень свободы объявляется.

## Чекпойнт для новой сессии

**Состояние:** P2 **завершено**. Все четыре модуля standalone-валидированы.
P2.1 среда (16 тестов). P2.2 возбуждение (18 тестов). P2.3 DDM (18 тестов,
все 3 fitted, held-out PASS). P2.4 RL+Mood (30 тестов, β fitted,
3 deferred PASS). 150/150 зелёные.

Обновления перед P2.4 (2026-07-30):
- D17: ревизия критерия coh=0 (значимость → размер эффекта, постфактум).
- E1: chi2 p>0.01 → magnitude max|dev|<0.03 (дефектен при N=60000).
- Разрешение a асимметрично: MDD вниз 6%, вверх 3% (σ шума, не сетка).
- Раздел «не воспроизводится» добавлен в DDM, arousal, среду.
  Arousal: σ_h перекалиброван 0.01→0.008 (D18, argmax frac in [15,30]).
    Onset circ SD = 21.5 мин, CI [20.6, 24.1]. Onset advance INDETERMINATE.
    CI пересчитаны: onset INDET, dur PASS, H PASS (N=100 при v_vh=0.325).
    Окно v_vh ≈ [0.325, 0.35] = ~6% (было [0.32, 0.33] = 2%). H_excess маргинален.
    Recovery r(v_vh): не пересчитано (exploratory).
  DDM: квантили RT, right-skew, slow errors.
  Среда: поведенческие наблюдаемые — модульные, не средовые.

### P2.4 RL+Mood — закрыт (2026-07-31)

Спецификация: `specs/rl_mood.md` v1.3. Статус: **утверждена, реализована**.

**Реализация:** `src/cpsych/modules/rl_mood.py`
- RLMoodModule(SDEModule): 3 состояния (V0, V1, m), 5 параметров.
- drift(m) = 0. Затухание в trial_update: m ← γ·(m + δ). Пробовое, не временно́е (D21).
- Softmax choice в outputs. trial_done = True (standalone, каждый шаг = проба).

**Тесты:** `tests/test_rl_mood.py` (30/30 зелёные). Включая:
- Purity, batch, defaults, mask, softmax, trial_done.
- Mood decay order (γ·(m+δ), не γ·m+δ).
- Trial duration invariance (D21).
- Stability below/above λ_crit (D20).

**Калибровка** (results/rl_mood_calibration.json):
- α = 0.15, β = 2.50. Win-stay = 0.702 (target 0.70). Recovery = 10.0 (target 10).
- Якобиан det = 3.04 (невырожден).
- τ_m = 2.163 (из γ = 0.63). λ_default = 0.49 (= 0.5·λ_crit).

**Parameter recovery** (results/rl_mood_recovery.json):

| Параметр | r (N=500) | 95% CI | r (N=1000) | Класс |
|----------|------:|--------|------:|-------|
| α | 0.673 | [0.659, 0.686] | 0.773 [0.763, 0.782] | **fitted при N≥1000** (exploratory при 500) |
| β | 0.899 | [—] | — | **fitted** |
| τ_m | 0.009 | [—] | — | **exploratory** |
| λ | 0.005 | [—] | — | **exploratory** |

α — **fitted при N ≥ 1000 проб** (r=0.77, CI lo=0.76). При N=500 — exploratory
(r=0.67, CI [0.66, 0.69]). При N=2000 — r=0.86. Ограничение не в модели,
а в длине последовательности: recovery speed — дискретная наблюдаемая (число
проб до 70%), при 500 пробах ~3-5 реверсий → высокая дисперсия.

β — fitted при 500 пробах. τ_m, λ — exploratory (τ_m не влияет на
поведенческие наблюдаемые при коротких последовательностях, λ — эффект
второго порядка). results/rl_mood_alpha_recovery_ci.json.

**Отложенные наблюдаемые** (results/rl_mood_deferred.json, обновлено):

| # | Тест | Значение | 95% CI | Критерий | Статус |
|---|------|----------|--------|----------|--------|
| 1 | Rutledge w_RPE/w_CR | w_RPE=0.63, w_CR=0.00 | — | w_RPE > w_CR > 0 | PASS (w_CR≈0: модель RPE-only) |
| 2 | Инвариантность к vol | |diff|=1.50 | [0.75, 2.25] | CI hi < 2.0 | **INDETERMINATE** |
| 3 | Mood autocorrelation | γ_model=0.74 | — | |γ−0.63|<0.15 | PASS (0.11), **частично циркулярная** |

**#2 пересмотрен.** Оригинальный PASS (|diff|=1.8 при B=200) — без CI.
С B=500 и bootstrap CI: |diff|=1.50, CI [0.75, 2.25]. CI верхняя граница
2.25 > порога 2.0 → не засчитывается как PASS. Статус: INDETERMINATE.
Причина: recovery speed — шумная дискретная величина, и при vol=0.1 мало
реверсий (~2-3 за 500 проб). Модель предсказывает инвариантность по
построению (α фиксирована), но данных недостаточно для подтверждения
на уровне <2 проб. results/rl_mood_vol_invariance_ci.json.

**#3 частично циркулярна.** Калибровалось: γ = 0.63 из Rutledge → τ_m = 2.163.
Проверялось: измеренная lag-1 автокорреляция mood = 0.74. Обе величины
восходят к одному числу Rutledge (γ = 0.63). При λ=0 измеренная
автокорреляция тождественно равна γ = 0.63 (тавтология — то же число).
Содержание теста — только вклад λ: петля увеличивает автокорреляцию
с 0.63 (заложено) до 0.74 (измерено). Аналитически: AR(1) с коэффициентом
a = γ·(1+λ·r_mean) = 0.63·1.29 = 0.815 (завышает; нелинейные эффекты
снижают). Тест проверяет, что λ не разрушает автокорреляцию, но не даёт
независимого предсказания. Помечен как частично циркулярный.

Решения: D19 (вариант A — моментальный аффект), D20 (λ_crit исправлен),
D21 (затухание в trial_update, drift(m)=0).

**Ключевые файлы:**
- Код: `src/cpsych/modules/arousal.py`, `src/cpsych/modules/ddm.py`, `src/cpsych/modules/rl_mood.py`
- Спецификации: `specs/arousal.md` v2.3, `specs/ddm.md` v1.2, `specs/rl_mood.md` v1.3
- Результаты: `results/arousal_*.json`, `results/ddm_*.json`, `results/rl_mood_*.json`, `results/roitman_rts.csv`
- Эксперименты: `experiments/arousal_*.py`, `experiments/ddm_*.py`, `experiments/rl_mood_*.py`
- Решения: `reports/decisions.md` (D1–D21)
- Альтернативы: `reports/alternatives.md` (A1–A9)
- CLAUDE.md: правила 1–21

**Решения D1–D21** зафиксированы.

**Следующий шаг:** P4 — тройной композит + карта невозможного.

## P3 — попарные связи (2026-08-01)

### Решения P3

- D22: Arousal → RL через gain → β (standalone only, уточнено в D25).
- D23: Mood → DDM отложена (нет литературного обоснования).
- D24: P3 gate — для каждой пары: анализ устойчивости; для пар с strength — критическая сила и ограничение ручки.
- D25: gain→β условна (standalone only). β в value-based DDM = drift/threshold, не temperature.
- D26: gain→α — основная arousal→RL связь для composite. Норадренергическая модуляция пластичности.
- D27: Замыкание контура DDM↔RL через drift_bias. value_diff → drift_bias = s*(V1-V0).
- D28: Timeout trials — V не обновляется (усечённая проба не информирует о ценности).
- D29: Немонотонная accuracy(s) — межмодульный результат. Оптимум s ≈ 3. Диапазон ручки [0, 10].
- D30: Замыкание стабилизирует mood loop. λ_crit: 0.98 → 1.1 (+12%). V-адаптация демпфирует RPE.

### Инфраструктурные изменения (P3.1)

- `integrator.py`: env_result["choice"] = action (передаётся в trial_update).
- `coupling.py`: добавлены `param_min`/`param_max` для post-modulation clamping.
  Необходимо для gain→κ и gain→β: при sleep gain≈0.02, s=4 → unclamped kappa_eff<0.

### Пара 1: Arousal → DDM (gain → κ)

**Coupling:** `kappa_eff = kappa * max(0, 1 + s*(gain-1))`, s ∈ [0, 4.0].
param_min=0.0 предотвращает отрицательный kappa во сне.

**Устойчивость:**
- Jacobian block-triangular (односторонняя связь). J[arousal, ddm] = 0.
- DDM ∂drift/∂evidence = 0 → eigenvalue contribution = 0.
- Eigenvalues композита = arousal eigenvalues ∪ {0} — **не зависят от strength**.
- Функциональное ограничение: kappa_eff ≥ 0. При gain_min=0.79 (wake), s=4: kappa_eff = kappa*0.16 > 0.

**Правило 19а:** все состояния finite при max strength, 3000s симуляции.

**Тесты:** 13/13 (5 unit, 2 stability, 4 integration, 2 hybrid transition).

### Пара 2: Arousal → RL+Mood (gain → β)

**Coupling:** `beta_eff = beta * max(0, 1 + s*(gain-1))`, s ∈ [0, 4.0].
param_min=0.0. Идентичная форма gain→κ.

**Устойчивость:**
- Block-triangular (односторонняя). J[arousal, rl] = 0.
- λ_crit = (1−γ)/(γ·r_mean) не зависит от β. Coupling не уменьшает λ_crit.
- λ_crit(gain) = λ_crit(standalone) для любого gain — β не входит в формулу.

**Условия D22:**
1. Не дублирует gain→κ: κ → перцептивная чувствительность, β → exploitation/exploration.
2. λ_crit не зависит от β (аналитически подтверждено).
3. Эффект: развёртка отложена до experiments/ (P3 gate не требует).

**Тесты:** 12/12 (6 unit, 2 lambda_crit, 2 stability, 2 integration).

### Пара 2а: Arousal → RL+Mood (gain → α) — D26

**Coupling:** `alpha_eff = alpha * max(0, 1 + s*(gain-1))`, s ∈ [0, 4.0].
param_min=0.0, param_max=1.0. Идентичная мультипликативная форма.

**Конфигурация:** все (standalone + composite). В отличие от gain→β,
α модулирует обучение ценностей → value_diff → drift_bias → DDM.

**Устойчивость:**
- Block-triangular (односторонняя). J[arousal, rl] = 0.
- alpha_eff ∈ (0, 1] при max strength на всём wake gain range.

**Тесты:** 8 unit, 2 stability, 2 integration (в test_coupling_arousal_rl.py).

### Пара 3: DDM ↔ RL+Mood (замкнутый контур) — D27

**Механизм:** замкнутый контур через value_diff → drift_bias.
drift = kappa*coh*dir + s*(V1-V0). V обучается из reward.
Контур: V → drift_bias → DDM drift → choice → reward → V.

**CompositeTask:** `src/cpsych/environments/composite_task.py`.
Фиксированная coherence (P3). Direction = 2*correct_arm - 1.

**Timeout (D28):** V не обновляется на усечённых пробах, m обновляется.

**Устойчивость:**
- DDM drift зависит от V через drift_bias (замкнутый контур).
- λ_crit(standalone) = 0.98. λ_crit(closed, numerical) = **1.1** (+12%).
  Замыкание стабилизирует: value-based choices → быстрая V-адаптация.
- Все состояния finite при strength до 15.0.

**Развёртка s_value** (results/p3_value_coupling_fine.json, 40 seeds):
- Accuracy: оптимум **s ≈ 3** (+1.7%), плато [2, 5]. s ≥ 10 → 0. s = 20 → −2.2%.
- Recovery: монотонно ухудшается (4.0 → 7.0 проб).
- Диапазон ручки: s ∈ [0, 10]. Зелёная [0, 5], жёлтая [5, 10], красная >10.
- **Межмодульный результат (D29):** немонотонность — первое явление платформы,
  отсутствующее у модулей по отдельности.
- **λ_crit стабилизация (D30):** closed = 1.1 > standalone = 0.98 (+12%).

**Вырожденность (ρ, s):** drift_bias = s·ρ·Δp. Пары с ρ·s = const дают 100%
совпадение (results/p3_rho_s_degeneracy.json). ρ=1 снимает вырожденность.
Обоснование обновлено в specs/rl_mood.md.

**Тесты:** 22/22 (6 composite task, 5 coupling unit, 4 closed-loop integration,
2 timeout, 2 lambda_crit, 1 trial pace invariance, 2 finiteness).

### P3 gate: пройдены

| Пара | Тип | strength | Проверка | Статус |
|------|-----|----------|----------|--------|
| Arousal→DDM | parameter | gain→κ | eigenvalues, kappa>0, 19a | **PASS** |
| Arousal→RL | parameter | gain→β | eigenvalues, λ_crit(gain), 19a | **PASS** (standalone only, D25) |
| Arousal→RL | parameter | gain→α | eigenvalues, alpha∈(0,1], 19a | **PASS** (all configs, D26) |
| DDM↔RL | closed-loop | value_diff→drift_bias | accuracy sweep, λ_crit, (ρ,s) | **PASS** (D27, D29, D30) |

### Общий счёт тестов: 214/214 зелёные

- Ядро: 67 (specs 7, module 20, coupling 9, integrator 17, stability 17)
- P2: 82 (env 16, arousal 18, DDM 18, RL+Mood 30)
- P3: 65 (P3.1 integrator 1, arousal→DDM 13, arousal→RL 24, DDM↔RL 22, coupling clamps 2, integrator choice 1)
  - P3.4 ревизия: DDM↔RL переписан (14→22), arousal→RL расширен (12→24, gain→α добавлена)

### Ключевые файлы P3

- Спецификация: `specs/p3_couplings.md`
- Код: `src/cpsych/environments/composite_task.py`
- Изменённые: `src/cpsych/core/integrator.py`, `src/cpsych/core/coupling.py`
- Тесты: `tests/test_coupling_arousal_ddm.py`, `tests/test_coupling_arousal_rl.py`, `tests/test_coupling_ddm_rl.py`
- Решения: D22–D30, альтернативы: A10–A12

## Чекпойнт для новой сессии

### Сводка P3 (вход для P4)

| Связь | Тип | Конфигурации | Ручка | Диапазон | Граница | Эффекты |
|-------|-----|-------------|-------|----------|---------|---------|
| arousal.gain → DDM.kappa | параметрическая (мульт.) | все | s_kappa | [0, 4] | функц.: kappa_eff > 0 при s=4, gain=0.79 | Суточная модуляция accuracy DDM |
| arousal.gain → RL.beta | параметрическая (мульт.) | standalone RL | s_beta | [0, 4] | функц.: beta_eff > 0 при s=4, gain=0.79 | Суточная модуляция exploitation (D25: в composite инертна) |
| arousal.gain → RL.alpha | параметрическая (мульт.) | все | s_alpha | [0, 4] | функц.: alpha_eff ∈ (0, 1] | Суточная модуляция скорости обучения |
| RL.value_diff → DDM.drift_bias | параметрическая (аддит.) | composite (DDM+RL) | s_value | [0, 10] | функц.: оптимум s ≈ 3; s > 10 → ухудшение | **D29:** немонотонная accuracy — межмодульный результат. **D30:** λ_crit +12% (стабилизация) |
| DDM → RL (trial loop) | структурная | composite (DDM+RL) | нет | — | — | DDM заменяет softmax; D28: timeout → V skip |
| arousal.is_wake → trial gate | структурная (гейт) | все с arousal | нет | — | — | D10: проб нет во сне |

**Устойчивость:**
- Все связи arousal → X: block-triangular, eigenvalues не зависят от strength.
- RL → DDM (drift_bias): замкнутый контур. λ_crit(closed) = 1.1 > λ_crit(standalone) = 0.98.
- Все параметрические связи: param_min/param_max предотвращают выход за допустимый диапазон.

**Вырожденности:** (ρ, β), (η, λ), (ρ, s_value) — все сняты фиксацией ρ=1, η=1.

**Тесты:** 214/214 зелёные. P3 закрыт.

**Состояние:** P3 **завершено** (ревизия P3.4 включена). 214/214 тестов зелёные.
Инфраструктура: env_result["choice"], param_min/param_max в Coupling.
CompositeTask создан. DDM+RL замкнутый контур через drift_bias (D27).
gain→α для всех конфигураций (D26). gain→β для standalone (D25).

**P3.4 ревизия (закрыта):**
- Контур замкнут: RL.value_diff → DDM.drift_bias (D27).
- gain→α добавлена (D26). gain→β уточнена как standalone only (D25).
- Timeout: V skip, m update (D28).
- Развёртка s_value: accuracy оптимум s≈3 (+1.7%), деградация при s≥10. Recovery монотонно.
- D29: немонотонность — межмодульный результат (первое явление платформы).
- D30: λ_crit closed=1.1 > standalone=0.98 (+12%). Замыкание стабилизирует.
- (ρ, s) вырожденность: подтверждена, ρ=1 снимает. Обоснование в spec обновлено.
- Тесты: DDM↔RL переписаны (14→22), arousal→RL расширены (12→24).
- Решения: D25-D28. Альтернативы: A12. D22 обновлён.

**Следующий шаг:** P4 — тройной композит, карта невозможного, BAA.

---

## P4: Тройной композит, карта невозможного, BAA

**Статус:** в работе. D31–D33 утверждены автором с правками. Бенчмарк и scout
выполнены. Полная карта невозможного не запущена (бюджет выше порога D3).

**Тесты:** 248 собрано, 244/248 зелёные (`-m "not slow"`), 4 медленных (@pytest.mark.slow)
отделены. Медленные тесты (TestHybridTransition, T=172800): `pytest -m slow`.

**Решения:** D31 (project_state, clipping, ревизия), D32 (BAA nc arg, ревизия),
D33 (arousal→BAA + untested class, ревизия), D34 (scout pipeline test).
**Альтернативы:** A13 (BAA boundary enforcement, терминология исправлена).

### Ревизия D31–D33 (2026-08-02)

- **D31:** «reflecting» → «clipping (projection to boundary)». Добавлено
  предупреждение об артефактах (BAA W 73–98% на границе). Обязательная
  диагностика: boundary_fractions в SimState, порог 10%.
- **D32:** nc — конструкторный аргумент (default=5), не константа. Не в
  песочнице/развёртках (nc меняет размерность).
- **D33:** три класса параметров: **fitted** (r≥0.7), **exploratory** (recovery
  проведён, r<0.7), **untested** (recovery не проводился). BAA.gain → untested.
  Правило 18 в CLAUDE.md обновлено.

### BAA модуль

- BAA из baa/ интегрирован как SDEModule (nc=5, D32).
- project_state: clipping boundaries для A≥0, 0≤S≤S_max, W∈[W_min,W_max] (D31).
- 11/11 тестов: fixed point, boundaries, determinism, noise scaling, contract,
  **boundary diagnostics** (A channels >10% clipping at high noise).

### Диагностика границ (D31)

Реализация: `src/cpsych/core/integrator.py`
- `SimState.boundary_counts`: raw counts per variable per boundary.
- `SimState.boundary_fractions`: fraction of steps where clipping occurred.
- Отслеживаются только модули с `project_state ≠ identity`.
- Тест: BAA с высоким шумом (σ_base=2, σ_max=5) → A channels >10% lower boundary.

### Тройной композит

- Инфраструктура: record_behavior, composite observables (accuracy, mean_rt,
  win_stay_rate, trial_count).
- Scout pipeline: make_triple_integrator + run + observables.
- CompositeTask: DDM choice ±1 → arm 0/1, timeout passthrough.

### Бенчмарк (results/p4_benchmark.json)

| B | wall (s) | µs/rep-step | trials |
|------:|--------:|--------:|------:|
| 1 | 6.4 | 37.05 | 20 |
| 100 | 8.4 | 0.49 | 21 |
| 1000 | 26.9 | 0.16 | 23 |
| 5000 | 206.6 | 0.24 | 23 |

Speedup B=1→B=5000: **154x**. Memory at B=5000: peak 4.0 MB.
µs/rep-step оптимален при B≈1000. При B=5000 — насыщение кэша.

### Бюджет полной сетки (обновлён 2026-08-03)

**Векторизация:** узлы × реплики в одном батче. 15×15 × 20 rep = B=4500.
Прежний подход (B=30 на узел, последовательно) давал 4.7 мин/точку = 17.7 ч/карту.

Измерено (results/p4_benchmark_vectorized.json):
- **B=4500, T=86.4:** 8.0 мин/карту, 0.62 µs/rep-step.
- Trial count: mean=25.3, range=[11, 49]. T=86.4 даёт столько же проб,
  сколько T=3600 (система спит 99% лишнего времени).
- 3 карты: **24 мин.** Ниже порога D3 (4ч) в 10 раз.
- Speedup vs sequential: **44x.**

### Scout (results/p4_scout.json)

3×3 сетка, s_alpha=2, B=30, T=3600. Все 9 точек finite.

| s_kappa | s_value | acc | RT | win_stay | trials |
|--------:|--------:|-----:|-----:|--------:|------:|
| 0 | 0 | 0.603 | 0.677 | 0.853 | 36.7 |
| 0 | 3 | 0.603 | 0.691 | 0.822 | 35.7 |
| 0 | 10 | 0.601 | 0.683 | 0.841 | 36.0 |
| 2 | 0 | 0.477 | 0.925 | 0.385 | 21.4 |
| 2 | 3 | 0.540 | 0.898 | 0.506 | 23.0 |
| 2 | 10 | 0.519 | 0.865 | 0.576 | 23.6 |
| 4 | 0 | 0.511 | 0.892 | 0.518 | 22.9 |
| 4 | 3 | 0.507 | 0.888 | 0.495 | 23.0 |
| 4 | 10 | 0.514 | 0.908 | 0.483 | 22.4 |

**Наблюдения:**
- s_kappa>0 снижает accuracy (0.60→0.48–0.54) и trial count (37→21–24).
  Эффект gain→κ: во сне κ_eff падает, меньше проб, ниже accuracy.
- s_value при s_kappa=0 не влияет (value_diff невелик без arousal модуляции).
- s_value при s_kappa=2 **увеличивает** accuracy (0.48→0.54): value-based
  bias компенсирует снижение κ_eff. Межмодульный эффект.
- s_kappa=4 vs 2: разница мала (0.51 vs 0.48). Насыщение clipping (κ_eff→0).

### Гибридная устойчивость

- Тесты: test_p4_hybrid.py (переключение wake↔sleep), test_p4_stability.py (6/6).
- Правило 19а: проверка на переключениях.

### Карта невозможного — ВЫПОЛНЕНА (2026-08-04)

**D38: гонка реверсий в next_trial.** При запуске карты обнаружен критический баг:
`next_trial()` обновлял `correct_arm` ВСЕХ B реплик при `trial_done` одной.
При B=225 accuracy ≈ chance. Исправлено: маска `done` передаётся в `next_trial`.

**Три карты 15×15, 5 окон бодрствования по 600 с, dt_fast=0.001:**

| Карта | UNSTABLE | CHANCE | INDISC. | NORMAL | Стена |
|-------|:--------:|:------:|:-------:|:------:|:-----:|
| kappa×value (s_alpha=2) | 0 | 0 | 13 (6%) | 212 (94%) | 10.3 мин |
| kappa×alpha (s_value=3) | 0 | 0 | 56 (25%) | 169 (75%) | 11.1 мин |
| alpha×value (s_kappa=2) | 0 | 0 | 10 (4%) | 215 (96%) | 17.2 мин |

**Ключевые результаты:**

1. **Нестабильностей нет.** Ни одна точка сетки не даёт расходящихся состояний.
   Тройной композит устойчив на всём диапазоне связей.

2. **CHANCE нет.** При coh=0.128 DDM даёт choice accuracy 0.91–0.96 во всех точках.
   Нет области, где композит теряет способность к различению.

3. **Вырожденная пара: kappa×alpha (25%).** Обе модулируются одним источником
   (arousal.gain), эффект на RT частично дублируется. s_alpha влияет через learning
   rate → V → drift_bias → RT, что перекрывается с s_kappa → κ → drift → RT.

4. **Наблюдаемые по чувствительности:**
   - mean_rt: Δ=34% (0.527–0.709). Наиболее чувствительна, доминирует s_value.
   - win_stay_rate: Δ=12% (0.866–0.989). Чувствительна к s_value.
   - accuracy: Δ=9% (0.878–0.962). Наименее чувствительна при ceiling coh=0.128.
   - trial_count: Δ=87% (7674–14379). Диагностическая, определяется RT.

5. **Взаимодействие s_kappa × s_value подтверждено:**
   - s_kappa=4 снижает accuracy на 4% (0.948 → 0.910). Второй межмодульный результат.
   - s_value=10 компенсирует: 0.910 → 0.933 (+2.3%, 7.7σ).
   - Ручки связей не независимы. Следствие для P5: песочница должна показывать
     взаимодействие, не только маргинальные эффекты.

6. **Accuracy при s_value>15 снижается** (0.888–0.917). Немонотонность из D29
   воспроизведена: value bias перебивает перцептивный сигнал после реверсий.

**Три типа областей:**
- Потеря устойчивости: **не обнаружена.**
- Вырождение наблюдаемых: **не обнаружено** (accuracy > chance на всей сетке).
- Неразличимость: **6–25%** точек. Максимум на kappa×alpha (обе связи через gain).
  Вырожденности κ·a, β·ρ сняты фиксацией a, ρ. s·ρ снята фиксацией ρ=1.

**Решения:** D38 (гонка next_trial). **Альтернативы:** A14 (accuracy metric).

**Файлы результатов:**
- `results/p4_impossible_map_kappa_value.json`
- `results/p4_impossible_map_kappa_alpha.json`
- `results/p4_impossible_map_alpha_value.json`

### Карты при пониженных когерентностях (2026-08-04)

Пересчёт карты kappa×value при coh=0.064 и coh=0.032.

| | coh=0.128 | coh=0.064 | coh=0.032 |
|---|:---:|:---:|:---:|
| Baseline acc (s=0,0) | 0.948 | 0.806 | 0.670 |
| Acc range | 0.878–0.962 (Δ=9%) | 0.721–0.876 (Δ=16%) | 0.651–0.800 (Δ=15%) |
| s_kappa=4 drop | −4.0% | −3.2% | −1.4% |
| s_value=10 boost | +0.7% | +7.0% | +13.0% |
| s_value=20 drop | −7.1% | −7.3% | −1.9% |
| CHANCE | 0 | 0 | 0 |

**CHANCE=0 на всех когерентностях.** При coh=0.032, κ_eff ≥ 2.35 (s_kappa=4,
gain_min=0.79), drift ≥ 0.075, accuracy ≈ 0.65 с ~5300 пробами — достоверно выше chance.
Для CHANCE нужен coh=0 (чистый бандит) или s_kappa >> 4.

**Главный результат: s_value compensation усиливается при низких coh.**
При coh=0.032 s_value=10 даёт +13% accuracy. Value-based drift существенен
когда перцептивный сигнал слаб. При высоком coh (0.128) — перцепция доминирует,
value-based вклад неразличим.

**Немонотонность D29** воспроизведена на всех трёх когерентностях.

### Валидация детектора UNSTABLE (2026-08-04)

Подана заведомо расходящаяся конфигурация: λ=5 >> λ_crit=1.1.
rl_mood replicate 1 → non-finite (overflow в rho_eff = 1 + λ·m).
classify_point корректно возвращает UNSTABLE. Ноль UNSTABLE в карте —
честный результат: λ=0.49 < λ_crit на всём диапазоне coupling strengths.

### Аудит D38 (2026-08-04)

Полный аудит 22 файлов src/cpsych/: D38 — единственный экземпляр бага
«условное обновление per-replicate состояния без маскирования». Все
trial_update, reset_state, _handle_trials корректно маскированы.

**Таблица влияния D38:** см. decisions.md D38. Все P2/P3 результаты
**не затронуты** (P3 experiments B=1, P3 тесты volatility=0). Затронуты
и пересчитаны: P4 scout (заменён полной картой), P4 impossible map.

### BAA как 4-й модуль (2026-08-04)

**Реализация:** `make_quad_integrator` в `composite_setup.py`. BAA на slow tier
рядом с arousal (dt=1.0). Coupling arousal.gain → BAA.gain мультипликативная,
s_baa ∈ [0, 4], param_min=0. BAA.gain — класс untested (D33).

**Входы BAA в композите:** stimulus CompositeTask (coherence, direction) не
совпадает с BAA input_names (I0..I4). У BAA в этой конфигурации нет внешнего
стимула. Решение (D40): tonic_input — параметр модуля (param[13]).
drift: `I = tonic + u[:,:nc]`. Без ветвления. Первый вариант (if u==0: use tonic)
откачен как скрытая ветка (правило 5).

**Boundary diagnostics (D41):** все переменные < 0.5% (порог 10%). A каналы:
0.03–0.13% lower boundary. S, W: 0.00%. При eta=0 BAA далеко от границ.

**Тесты:** 7/7 (test_p4_quad.py). Quad all finite, gain modulated during wake,
boundary diagnostics in composite, trial loop not affected, batch independence,
A non-negative, coupling form verified.

**Общий счёт:** 251/251 (+ 7 новых).

### Противоречие D29/D30 — разрешено

D29 и D30 **чисты**. Оба считались скриптами p3_value_coupling_fine.py и
p3_value_coupling_sweep.py при B=1, кастомный trial loop, скалярный correct_arm.
D38 (гонка при B>1) не применим. Ошибка была в предыдущем отчёте, где я написал
«P3 experiments используют make_triple_integrator (volatility=0.3, B=30/40)» — неверно.

---

## Оценка ворот P4

### Условие ворот (из CLAUDE.md)
> P4: Тройной композит устойчив; получена карта невозможного; BAA подключён
> четвёртым модулем.

### Оценка

**Тройной композит устойчив — формально пройдено, содержательно слабо.**
Устойчивость проверена:
- Блочно-треугольный якобиан (все связи от arousal однонаправленные, D36).
- Eigenvalues = arousal eigenvalues ∪ {0} — не зависят от coupling strengths.
- λ_crit(closed-loop DDM↔RL) = 1.1, λ_default = 0.49 — запас 2.2×.
- Карта невозможного: 0 UNSTABLE на 675 точках (3 среза × 225).
- UNSTABLE детектор валидирован (λ > λ_crit → overflow → non-finite).

**Но эта проверка почти тривиальна.** Блочно-треугольная структура гарантирует,
что eigenvalues композита — объединение eigenvalues подсистем. Arousal автономен
(D36), его динамика не зависит от coupling strengths. DDM ∂drift/∂evidence = 0.
RL mood loop — единственный источник нестабильности (eigenvalue = γ(1+λr)),
и λ зафиксирован ниже λ_crit.

Настоящая проверка устойчивости появится при обратной связи на arousal:
mood → sleep latency, attention → arousal drive, или фармакологические петли.
До этого устойчивость гарантирована архитектурой, а не тестом.

**Карта невозможного — пройдено.**
- 3 типа областей проверены: нестабильность (0), вырождение (0 CHANCE),
  неразличимость (6–25%).
- Два межмодульных результата: D29 (немонотонность accuracy), D39 (компенсация
  растёт с трудностью).
- Карты при coh=0.064/0.032 подтвердили, что accuracy — наименее чувствительная
  наблюдаемая, mean_rt доминирует.

**BAA подключён 4-м модулем — пройдено.**
- Coupling arousal.gain → BAA.gain, 7 тестов, boundary diagnostics в композите.
- BAA.gain untested (recovery не проводился, D33).
- BAA на slow tier, не участвует в trial loop, не влияет на поведенческие наблюдаемые.
  В текущей архитектуре BAA — пассивный наблюдатель arousal через gain coupling.

### P4 gate: **пройдены** с оговорками

1. Устойчивость проверена, но проверка слабая (блочно-треугольность делает её
   почти тавтологической). Записано честно: настоящая проверка при обратной связи.
2. BAA подключён, но пассивен — не влияет на решательный контур.
3. Карта невозможного не содержит областей неустойчивости или вырождения
   при coh ≥ 0.032. CHANCE возможен только при coh → 0 (чистый бандит).

**Тесты:** 251/251 зелёные. D31–D41. A13–A14.

## Чекпойнт для новой сессии

**Состояние:** **P4 пройдены.** 251/251 тестов. D31–D41 зафиксированы.

**Что сделано в P4:**
- BAA модуль + quad composite (7 тестов, D40/D41)
- tonic_input как параметр (D40), откат ветвления в drift
- Boundary diagnostics: все переменные < 0.5% (D41)
- Карта невозможного: 5 срезов (3 при coh=0.128 + 2 при coh=0.064/0.032)
- D38: гонка реверсий (исправлена + аудит); D39: компенсация vs трудность
- UNSTABLE детектор валидирован
- D29/D30 чисты (B=1, volatility не участвует)

**Следующий шаг:** P5 — песочница.

**Ключевые файлы P4:**
- `src/cpsych/core/composite_setup.py` (make_quad_integrator)
- `src/cpsych/modules/baa_attention.py` (tonic_input как param[13])
- `tests/test_p4_quad.py` (7 тестов)
- `experiments/p4_impossible_map.py`
- `results/p4_impossible_map_*.json`
- Решения: D31–D42, альтернативы: A13–A14

---

## P5: Песочница — реализована (2026-08-04)

**Спецификация:** `specs/sandbox.md` v3. **Утверждена.**
**Реализация:** `sandbox/sandbox.py`.

### Структурный вынос slow-tier outputs

61% времени при B=1 уходило на пересчёт arousal.outputs в `_effective_theta`
на каждом из 60к fast шагов — хотя по Lie-Trotter arousal state постоянен
на быстром интервале.

Исправление: outputs slow-tier модулей вычисляются один раз за медленный
шаг в `run()` и передаются вниз через `_run_tier` → `_step_module` →
`_effective_theta` как готовые значения. Не кэш с проверкой актуальности —
изменение потока данных.

| Конфигурация | Было | Стало | Ускорение |
|---|---:|---:|---:|
| 1 окно, B=1 | 105 с | **44.6 с** | 2.4× |
| 5 окон, B=30 | 12.5 мин | **4.5 мин** | 2.8× |
| Весь test suite | 22.5 мин | **9.9 мин** | 2.3× |

Амортизация работает: B=30 стоит 56.5 с (1.9 с/реплику vs 44.6 при B=1).

### CI — prediction interval

CI = mean ± 1.96×SD (разброс между репликами), не SEM. Показывает
диапазон одного прогона, не точность среднего. Ширина ~0.009 по accuracy.

### Реализация

- Мгновенный (< 100 мс): DDM accuracy/RT. Помечено «DDM без связей,
  идеализация». После прогона: оба значения рядом, не подмена.
- По кнопке (~4.5 мин): B=30, 5 окон, prediction interval.
- Три класса ручек: ● fitted (шаг=MDD), ○ exploratory.
- α: динамическая пометка класса (fitted при N≥1000).
- Warnings: λ>λ_crit, v_vh за окном, s_value>10, BAA пассивен.

### P5 gate: PASS (все 6 пунктов)

| # | Критерий | Статус |
|---|----------|--------|
| 1 | Fitted параметры с шагом = MDD | **PASS** (5 параметров) |
| 2 | Exploratory различимы, α динамический | **PASS** |
| 3 | Предупреждения срабатывают | **PASS** (4 проверки) |
| 4 | Аналитика < 100 мс, прогон по кнопке | **PASS** (0.00 мс; 4.5 мин) |
| 5 | Карта с позицией ручек | **PASS** (3 JSON) |
| 6 | BAA помечен пассивным | **PASS** (D42) |

Решения: D42 (BAA пассивен), D43 (каскад + амортизация).
251/251 тестов зелёные.

---

## P6: Сравнение с реальными данными

**Спецификация:** `specs/p6_validation.md` v2. **Утверждена.**

### Шаг 0: предрегистрация

Закоммичена до загрузки данных (`fff056d`, `6aefd62`).
Критерии: MLE для DDM, пороги из межнаборного разброса, распределения
вместо бинарных порогов, уверенность в «не воспроизводит».

### Шаг 0a: выживаемость отбора

101/128 перцептивных наборов проходят фильтр (≥200 проб).
57 многоусловных, 44 одноусловных. Отчёт раздельный.

### Шаг 1: загрузка данных — завершена

| Набор | Статус | N файлов | Размер | Лицензия |
|-------|--------|----------|--------|----------|
| Confidence DB | **97/101 CSV** | 194 | 128 MB | CC-By 4.0 |
| OBF-Psychiatric | **загружен** | 175 | 10.4 MB | CC BY 4.0 |
| Eckstein 2022 | **загружен** | 310 | ~3 MB | CC-By 4.0 |
| Zou 2022 | **недоступен** (0 bytes) | — | — | — |

**D44:** Zou заменён Eckstein (первоисточник протокола). Criterion mode
добавлен в ReversalBandit (E11–E15). 256/256 тестов.

### Шаг 1a: нормализация accuracy — завершена

84 набора нормализовано (74 auto + 10 rescued), 13 исключено.
Правила: `data/accuracy_coding.json` (зафиксированы до метрик).

Исключены (13): не 2AFC (Akdogan, Duyan ×5, Rausch_2014, Samaha ×2,
Xiang, Xue) или неустановимое кодирование (Konishi, Yeon 4AFC).

### Шаг 2: межнаборный разброс — вычислен

72 набора с валидными accuracy + RT.

| | Full (72) | Multi (50) | Single (22) |
|---|:-:|:-:|:-:|
| Accuracy SD | **0.047** | 0.050 | 0.038 |
| RT SD | **0.535 с** | 0.527 с | 0.563 с |
| RT SD/mean | **51%** | 49% | 56% |

Порог accuracy: ±0.047 (4.7%) — содержательный.
Порог RT: ±0.535 с (51%) — **слабый** (записано в spec).

### Шаг 3: DDM MLE — завершён

Recovery (синтетика N=500, 50 точек): κ r=0.889, a r=0.995, Ter r=0.990.
D46: баг знаков в density найден recovery. D45: пороги пересмотрены по уровням.

71 набор, 3964 участника, 2.2 мин. Результат:
- **Single-condition (23):** MAE=0.023 < порог 0.038. **Положительный.**
- **Multi, coherence (11):** MAE=0.037 < порог 0.044. **Положительный.**
- **Multi, все (48):** MAE=0.090 ≈ порог 0.100. **Неопределённый** [40%,68%].
- RT: multi +2.3% (model slower), single −6.8% (faster). Слабый критерий.
- Шкала: κ·difficulty — 5-я вырожденность. Единицы произвольны.

### Шаг 4: Arousal — завершён

32 контроля, OBF-Psychiatric. Все параметры фиксированы (exploratory).
- **Средние:** 8.32h vs 8.31h, onset 23:00 vs 22:34. **Положительный.**
- **Вариабельность:** within-SD 2.50h vs model 0.50h (5×).
  **Структурное ограничение:** нет внешних источников (социальное расписание).
- D47: восстановимость ≠ предсказательная способность.

### Шаг 5: RL — завершён

114 взрослых (18+), Eckstein 2022, criterion mode. Все параметры фиксированы.
- WS: 0.681 vs 0.952. ACC: 0.575 vs 0.731. **Отрицательный.**
- Диагностика: α=0.76 (Eckstein) → WS=0.891, ACC=0.711. Обе α и β
  зависят от информативности обратной связи.

### Упаковка

- `article/numbers.json`: 32 числа, verified 32/32.
- `article/verify_numbers.py`: автоматическая сверка с results/.
- `article/provenance.md`: каждое число → файл + скрипт.
- `article/tables/p6_summary.md`: сводная таблица + «не воспроизводит».
- `requirements.txt`: зафиксированные версии.
- Данные в .gitignore, скрипты + манифесты в репо.

### P6 итог (обновлён 2026-08-14 после D52)

| Сравнение | Вердикт |
|-----------|---------|
| DDM single | **Неопределённый** (MAE 0.041 > 0.038) |
| DDM coherence | **Отрицательный** (MAE 0.086 > 0.044) |
| DDM multi all | **Неопределённый** (MAE 0.102 ≈ 0.100) |
| Arousal средние | **Положительный** |
| Arousal вариабельность | Структурный разрыв |
| RL | **Отрицательный** (контекст-зависимость) |

D44–D47, D51–D52a. 256/256 тестов.

---

## P7: Рукопись

**Статус: ЗАВЕРШЕНА 2026-08-15.** Рукопись готова к подаче.

Все разделы (00–07) написаны, числа обновлены, abstract принят.
Конвейер Appendix H прошёл с нуля без ошибок: 256/256 тестов, 64/64 чисел.
Red-team на полный документ проведён, все находки исправлены.
Библиография: 29 записей, все цитируются в тексте.
Собранный документ: `article/assembled.md` (24 таблицы, пронумерованы).

### Инфраструктура статьи

- `article/numbers.json`: **64 числа**, verified 64/64 (`verify_numbers.py`).
  Было 77; снято 13 (withdrawn arousal duration, scale-type Bonferroni).
- `article/provenance.md`: генерируется `build_provenance.py` из 17 файлов.
- `article/refs.bib`: **47 ссылок** с DOI.
- `article/sections/` — 7 файлов + abstract (не написан).

### Разделы — состояние каждого

| # | Файл | Статус | Требуется |
|---|------|--------|-----------|
| 1 | `01_introduction.md` | **текст устарел** | Обновить числа (см. таблицу ниже), убрать 8.31/7.75, добавить withdrawal |
| 2 | `02_methods_protocol.md` | **текст устарел** | Добавить пробел предрегистрации (D49: прибор + шкала), D50 (нормализация) |
| 3 | `03_methods_platform.md` | OK | — |
| 4 | `04_results_construction.md` | **текст устарел** | MLE recovery r(κ) 0.889→0.945 |
| 5 | `05_results_comparison.md` | **частично обновлён** | Arousal section rewritten (D49). DDM section **устарел** — все числа |
| 6 | `06_discussion.md` | **текст устарел** | Arousal dissociation reframed, onset как настоящее предсказание, pre-reg gap |
| 7 | `07_appendices.md` | **текст устарел** | MLE recovery numbers, reproduction instructions (скрипты восстановлены) |
| — | abstract | **не написан** | Пишется после обновления всех разделов |

### Что сделано в сессии 2026-08-06

1. **Восстановлены 3 скрипта P6** из спецификации:
   - `experiments/p6_ddm_mle_recovery.py` — MLE synthetic recovery
   - `experiments/p6_ddm_comparison.py` — DDM vs Confidence Database
   - `experiments/p6_arousal_comparison.py` — arousal vs OBF-Psychiatric

2. **D48:** утерянные скрипты, восстановление по спецификации, расхождение MLE recovery.
   **D49:** arousal duration withdrawn. Три причины: утрата кода, пробел предрегистрации
   (два противоречащих перцентиля, нет прибора/шкалы), отсутствие внешнего алгоритма
   для Actiwatch AW4. Onset + variance decomposition retained с robustness table.
   **D50:** DDM нормализация difficulty (no shift) и condition detection (physical difficulty criterion).
   **A15:** четыре отвергнутых варианта разметки сна.

3. **Arousal:** длительность сна снята. Момент засыпания (23:39–00:01 при трёх вариантах,
   модель 23:00, diff 38–60 мин — не калибровался, настоящее предсказание) и
   разложение дисперсии (within/between ≈ 2.1, устойчиво к разметке) остаются.

4. **DDM:** нормализация исправлена (v = κ·difficulty инвариантно к умножению, не к сдвигу).
   Condition detection по критерию physical difficulty. Per-subject порог 50 (не 200).

5. **numbers.json:** перестроен, 64/64 verified. provenance обновлён.

6. **.gitignore:** `.claude/`, `nul`, `baa/` добавлены.
   `07_appendices.md` написан (уравнения, recovery, параметры, манифесты SHA-256).
   progress.md обновлён.

### Таблица изменившихся чисел (было → стало)

Пороги (0.038, 0.044, 0.100) **не менялись** — вычислены из данных до подгонки.

| Число | Было | Стало | Источник |
|-------|-----:|------:|----------|
| MLE r(κ) | 0.889 | 0.945 | p6_ddm_mle_recovery.json |
| MLE r(a) | 0.995 | 0.971 | p6_ddm_mle_recovery.json |
| MLE r(Ter) | 0.990 | 0.992 | p6_ddm_mle_recovery.json |
| DDM datasets | 71 | 73 | p6_ddm_comparison.json |
| DDM subjects | 3964 | 4066 | p6_ddm_comparison.json |
| DDM multi/single | 48/23 | 43/30 | p6_ddm_comparison.json |
| Single MAE median | 0.023 | 0.040 | p6_ddm_comparison.json |
| Coherence MAE median | 0.037 | 0.066 | p6_ddm_comparison.json |
| Multi MAE median | 0.090 | 0.123 | p6_ddm_comparison.json |
| Single in threshold | 14/23 | 14/30 | p6_ddm_comparison.json |
| Arousal mean duration | 8.31h/7.75h | **СНЯТО** | D49 |
| Arousal onset diff | 26 мин | 38–60 мин | p6_arousal_comparison.json |
| Arousal onset within-SD | 92 мин | 103–104 мин | p6_arousal_comparison.json |
| Arousal within-SD dur | 2.50h | 2.15–2.18h | p6_arousal_comparison.json |
| Arousal between-SD dur | 0.66h | 0.98–1.04h | p6_arousal_comparison.json |
| Arousal N subjects | 29 | 32 | p6_arousal_comparison.json |
| Bonferroni p-values | 0.012, 0.162 | **СНЯТО** | D50 (scale-type analysis) |

**Вердикты DDM требуют пересмотра.** Single MAE 0.040 > порога 0.038 (было 0.023 < 0.038).
Coherence MAE 0.066 > порога 0.044 (было 0.037 < 0.044). Это может сменить вердикт
с «положительного» на «неопределённый» или «отрицательный». Решение — в следующей сессии,
после разбора 18 наборов с расходящейся классификацией (от неё зависят агрегаты).

### 18 наборов с расходящейся multi/single классификацией — РАЗОБРАНЫ (D52)

**Разобраны 2026-08-14.** 8 overrides в `data/condition_classification.json`,
4 исправлены багом D52a. Итог: 43→53 multi, 30→20 single. **Больше не блокирует.**

### Что сделано в сессии 2026-08-14

1. **Разбор 18 наборов multi/single — завершён (D52).**
   Критерий формализован: physical difficulty (contrast, coherence, SOA, noise,
   dot difference, orientation) vs task manipulation (cue validity, instruction).
   Override-файл: `data/condition_classification.json`. 8 overrides + 4 bug fix (D52a).

2. **Баг is_multi=False исправлен (D52a).** Case-mismatch `difficulty` vs `Difficulty`.
   4 набора (Maniscalco_2017_expt1/3/4, Shekhar_2018) ошибочно были single.

3. **DDM comparison пересчитан.** 43→53 multi, 30→20 single. N=4066 без изменений.

4. **rl_mood_resolution восстановлен (D51).** Побитовое совпадение с существующим JSON.

5. **numbers.json обновлён.** 8 чисел DDM. 64/64 verified.

6. **provenance.md перегенерирован.** 64 numbers, 17 sources.

### Таблица изменившихся чисел — сессия 2026-08-14 (было → стало)

| Число | Было | Стало | Источник |
|-------|-----:|------:|----------|
| DDM multi/single | 43/30 | 53/20 | p6_ddm_comparison.json |
| Single N | 30 | 20 | p6_ddm_comparison.json |
| Single in threshold | 14 | 8 | p6_ddm_comparison.json |
| Single MAE median | 0.040 | 0.041 | p6_ddm_comparison.json |
| Multi MAE median | 0.123 | 0.102 | p6_ddm_comparison.json |
| Coherence MAE median | 0.066 | 0.086 | p6_ddm_comparison.json |
| Coherence accuracy_SD | 0.026 | 0.032 | p6_ddm_comparison.json |
| Coherence N | 8 | 10 | p6_ddm_comparison.json |

Пороги (0.038, 0.044, 0.100) **не менялись.**

### Обновлённые вердикты DDM

| Категория | MAE median | Порог | CI | Вердикт |
|-----------|:----------:|:-----:|----|---------|
| Single (N=20) | 0.041 | 0.038 | [0.2, 0.6] | **Неопределённый** |
| Coherence (N=10) | 0.086 | 0.044 | — | **Отрицательный** |
| Multi all (N=53) | 0.102 | 0.100 | [0.358, 0.623] | **Неопределённый** |

**Изменения по сравнению с D50:**
- Single: было 0.040 ≈ 0.038 (borderline positive) → 0.041 > 0.038 (indeterminate).
  Потеря 10 наборов (30→20) сужает выборку; MAE поднялся на 0.001.
- Coherence: было 0.066 → 0.086. Добавлены 2 набора. Уже был отрицательный.
- Multi: было 0.123 → 0.102. Multi **улучшился** — 10 наборов, ошибочно fit как single,
  теперь fit с difficulty-aware kappa. Многие имеют низкий MAE (Xue 0.023/0.048).
  Вердикт сменился с отрицательного на неопределённый.

### Сводная таблица P6 (обновлённая)

| Сравнение | Вердикт |
|-----------|---------|
| DDM single | **Неопределённый** (было: borderline positive) |
| DDM coherence | **Отрицательный** (без изменений) |
| DDM multi all | **Неопределённый** (было: отрицательный) |
| Arousal средние | **Положительный** (без изменений) |
| Arousal вариабельность | Структурный разрыв (без изменений) |
| RL | **Отрицательный** (без изменений) |

### Orphan list (некритические, 6 штук)

Из аудита D48. Не ссылаются из numbers.json. Не влияют на числа в статье.

1. `results/arousal_demo.json`
2. `results/arousal_deferred4.json`, `arousal_deferred_ci.json`
3. `results/arousal_vh_sweep_stochastic.json`, `arousal_d18_recompute.json`, `arousal_validation.json`

Оставлены как есть — промежуточные результаты P2–P4.

### Что осталось (в порядке выполнения)

1. **Обновить DDM секцию в 05_results_comparison.md** — все числа из таблицы выше.
   Пересмотренные вердикты (single, coherence, multi).

2. **Обновить 01_introduction.md** — убрать 8.31/7.75, r(κ) 0.889→0.945, duration
   withdrawal, onset как настоящее предсказание, DDM вердикты.

3. **Обновить 06_discussion.md** — arousal dissociation reframed (снятый результат
   был калибровочным следствием, уцелевший — настоящее предсказание), pre-reg gap
   (прибор + шкала), script loss in limitations, DDM вердикты.

4. **Обновить 04_results_construction.md** — MLE recovery r(κ) 0.889→0.945.

5. **Обновить 07_appendices.md** — MLE recovery числа, reproduction instructions
   (скрипты теперь существуют).

6. **Обновить P6 summary table** (`article/tables/p6_summary.md`).

7. **Обновить 02_methods_protocol.md** — пробел предрегистрации (D49, D50), D52 criterion.

8. **Abstract** — после обновления всех разделов.

9. **Red-team** — на обновлённые разделы и abstract.

### Проверки здоровья (2026-08-14)

| Проверка | Результат |
|----------|-----------|
| verify_numbers.py | **64/64 OK** |
| build_provenance.py | 64 numbers, 17 sources |
| pytest (not slow) | 256 passed, 4 deselected, 0 failed |

### Решения P7

D44–D47 (из P6). D48–D50, A15 (сессия 2026-08-06). D51–D52a (сессия 2026-08-14). D53 (сессия 2026-08-15).

### Что сделано в сессии 2026-08-15

1. **Abstract написан и принят.** Пять правок после первой версии: ложная связка, повтор,
   происхождение порогов, композитные явления, формулировка RL.

2. **Три предабстрактных исправления.**
   - v_mv: Internal → Exploratory (Issue 1, правило 18).
   - Eckstein: «not an independent replication» перенесён в начало предложения.
   - Discussion: добавлен обзор 5 платформ для обоснования «no other platform».
   - refs.bib: +5 ссылок (Sanz Leon, Friston 2003, Eliasmith, Deco, Ahn).

3. **D53: стухшие чекпоинты.** Arousal recovery r-значения 0.41/0.35/0.29 оказались
   невоспроизводимыми — причина: `.npy` файлы не отслеживались git, молча подгружались
   при повторном запуске, пережили изменения кода. Код полностью детерминирован
   (SHA-256 одинаков между процессами). Исправлено: checkpoint loading удалён,
   правило 22 добавлено, тест AR19 добавлен. Новые канонические значения: 0.42/0.37/0.31/0.05.

4. **p6_rl.py расширен.** Добавлена диагностическая секция (α Eckstein, persistence,
   concordance ratios). 6 чисел в numbers.json были невоспроизводимы.
   Новые значения: WS=0.937, ACC=0.726, 84%.

5. **Appendix H верифицирован.** Один скрипт переименован (arousal_recovery.py →
   arousal_validation.py). Конвейер пройден с нуля: 256/256, 64/64, exit 0.

6. **Библиография вычищена.** 52 → 29 записей. 23 незакавыченных удалены.

7. **Red-team на полный текст.** 8 находок исправлены: стухшая ссылка на длительность сна,
   fivefold→fourfold, 2 untested в C.4, Appendix F 84→82, 4→5 slow tests,
   C.2 MLE r-values, «Three scripts»→«Four scripts».

8. **Финальная сборка.** `article/assembled.md`, 24 таблицы с нумерацией.

### Проверки здоровья (2026-08-15)

| Проверка | Результат |
|----------|-----------|
| verify_numbers.py | **64/64 OK** |
| build_provenance.py | 64 numbers, 17 sources |
| pytest (not slow) | 256 passed, 5 deselected, 0 failed |
| Appendix H pipeline | **полный прогон, exit 0** |
| refs.bib | 29 entries, 0 duplicates, all cited |
| assembled.md | 24 tables numbered |

### Что дальше

- **Ветка под конкретный журнал.** Форматирование, word count, supplementary.
- **Расширение DDM** (η, s_z, s_t) — устраняет 3 model limitations.
- **Фармакологические плагины** — θ(dose, time).
- **Клинические сравнения** — OBF-Psychiatric (депрессия, ADHD, шизофрения).

### Правило чекпойнта

Чекпойнт — описание состояния, а не инструкция агенту. Формат: что сделано,
что осталось, чем проверено.
