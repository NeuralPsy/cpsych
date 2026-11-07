# P3: Попарные связи — спецификация v1.0

Статус: зафиксирована до реализации (правило 15).

## Три пары

### Пара 1: Arousal → DDM (gain → kappa)

**Механизм:** мультипликативная модуляция drift sensitivity.
```
kappa_eff = kappa_base * max(0, 1 + s*(gain - 1))
```
При gain=1 (baseline): kappa_eff = kappa_base. При gain>1: усиление.
При gain<1 (засыпание/сон): ослабление.

**h-функция:** `h(gain, kappa_base) = kappa_base * (gain - 1)`
**Semantics coupling:** `theta[kappa] += s * h(gain, kappa_base)`
  → `kappa_eff = kappa_base + s * kappa_base * (gain - 1) = kappa_base * (1 + s*(gain-1))`

**Strength:** s ∈ [0, 4.0].
Теоретический порог переворота знака: s = 1/(1 - gain_min).
gain_min бодрствования ≈ 0.79 → s_crit = 4.76. max_strength = 4.0 (запас 16%).

**Односторонняя связь.** Arousal не зависит от DDM. Composite Jacobian —
block-triangular. DDM drift ∂/∂evidence = 0 (evidence не входит в drift
кроме как через kappa*coh*dir). Eigenvalues композита = arousal eigenvalues
∪ {0}. Устойчивость не зависит от strength — но kappa_eff > 0 обязательно.

**Правило 19а:** прогон wake→sleep→wake, проверка нет расхождений,
нет залипания, нет паразитных осцилляций вблизи порога.

### Пара 2: Arousal → RL+Mood (gain → beta)

**Механизм:** мультипликативная модуляция inverse temperature (D22).
```
beta_eff = beta_base * max(0, 1 + s*(gain - 1))
```
Форма идентична gain→kappa. Различие: kappa масштабирует чувствительность
к перцептивному стимулу (coherence), beta масштабирует чувствительность
к разности выученных ценностей (V1-V0).

**h-функция:** `h(gain, beta_base) = beta_base * (gain - 1)`
**Strength:** s ∈ [0, 4.0]. Аналогичный запас.

**Условия D22:**
1. Не дублирует gain→kappa. kappa → accuracy (задача-зависимо),
   beta → win-stay и exploitation. Проверка пересечения — P4.
2. lambda_crit(gain): развёртка strength × lambda при gain из [0.79, 1.05].
   Высокий beta → более детерминированный выбор → bias → RPE pattern → mood.
   lambda_crit зависит от beta_eff через r_mean (если r_mean меняется).
3. Показать эффект: развёртка strength, наблюдаемые RL на 24-часовом цикле.

**Правило 19а:** аналогично паре 1.

### Пара 2а: Arousal → RL+Mood (gain → alpha) — D26

**Механизм:** мультипликативная модуляция learning rate.
```
alpha_eff = alpha_base * max(0, 1 + s*(gain - 1))
```
Форма идентична gain→kappa и gain→beta. Норадренергическая модуляция
пластичности: высокий arousal → быстрое обучение, низкий → медленное.

**h-функция:** `h(gain, alpha_base) = alpha_base * (gain - 1)`
**Strength:** s ∈ [0, 4.0]. Аналогичный запас.
**param_min=0.0, param_max=1.0:** α > 1 → V перескакивает цель.

**Конфигурация:** все (standalone + composite). В отличие от gain→β,
gain→α действует и при наличии DDM: α модулирует скорость обучения
ценностей, которые через value_diff → drift_bias влияют на DDM.

### Пара 3: DDM ↔ RL+Mood (замкнутый контур) — D27

**Механизм:** замкнутый контур через value_diff → drift_bias.
RL.value_diff = V1 − V0 модулирует DDM.drift_bias:
```
drift = kappa * coherence * direction + s * (V1 - V0)
```
V обучается из reward. Контур: V → drift_bias → DDM drift → choice → reward → V.

**CompositeTask:** среда-композит, объединяющая перцептивное решение
(coherence, direction) с бандитом (correct_arm, reward).
- ReversalBandit определяет correct_arm. Direction = correct_arm mapped ±1.
- DDM решает direction по coherence + value bias. Choice ±1 → arm 0/1.
- RL обновляет V на reward от бандита.

**Mapping:** DDM choice=+1 → arm 1, DDM choice=-1 → arm 0.

**Замыкание контура (D27).** DDM drift зависит от V через drift_bias.
Стандартная value-based DDM: drift содержит и перцептивный, и ценностный
компоненты. При coh=0, drift = s*(V1-V0): выбор определяется обученными
ценностями.

**Timeout (D28).** Усечённые пробы: V не обновляется (нет информации о
ценности), m обновляется (агент переживает RPE от исхода).

**Trial pace.** DDM RT зависит от coherence. RL mood decay — per-trial (D21).
Per-trial m trajectory инвариантна к coherence. Per-time — нет (больше решений
при высокой coherence). Это feature, не bug (D21).

## Конфигурации связей

| Связь | Конфигурации |
|-------|-------------|
| arousal.gain → DDM.kappa | все |
| arousal.gain → RL.beta | standalone RL (softmax choice) |
| arousal.gain → RL.alpha | все |
| RL.value_diff → DDM.drift_bias | composite (DDM+RL) |

## Критерии успеха

### Пара 1: Arousal → DDM

| # | Критерий | Порог | Тип |
|---|----------|-------|-----|
| C1 | kappa_eff > 0 при s=max, gain=0.79 | > 0 | pass/fail |
| C2 | Accuracy корреляция с gain | r > 0 (монотонность) | pass/fail |
| C3 | No divergence at wake/sleep transition | max |evidence| < 100 | pass/fail |
| C4 | Trial completion during wake | trial_count > 0 | pass/fail |
| C5 | No trial completion during sleep | trial_count = 0 при gate | pass/fail |

### Пара 2: Arousal → RL+Mood

| # | Критерий | Порог | Тип |
|---|----------|-------|-----|
| C6 | beta_eff > 0 при s=max, gain=0.79 | > 0 | pass/fail |
| C7 | lambda_crit(beta_eff) >= lambda_crit(standalone) | >= | pass/fail |
| C8 | No divergence at transition | max |m| < 100 | pass/fail |
| C9 | Effect size of strength on win-stay | отчётная | report |

### Пара 2а: Arousal → RL+Mood (gain → alpha)

| # | Критерий | Порог | Тип |
|---|----------|-------|-----|
| C6a | alpha_eff > 0 при s=max, gain=0.79 | > 0 | pass/fail |
| C6b | alpha_eff ≤ 1.0 при s=max, gain=1.05 | ≤ 1.0 | pass/fail |
| C8a | No divergence at max strength | max |V|, |m| < 100 | pass/fail |

### Пара 3: DDM ↔ RL+Mood (замкнутый контур)

| # | Критерий | Порог | Тип |
|---|----------|-------|-----|
| C15 | DDM biased by V (drift_bias ≠ 0) | V differentiation > 0.02 | pass/fail |
| C16 | V_correct > V_incorrect after learning | movement from init | pass/fail |
| C17 | Accuracy non-monotonic with s | peak, then decline | report |
| C18 | λ_crit(closed) > λ_crit(standalone) | numerical > 0.98 | report |
| C13 | Per-trial m trajectory invariant to trial duration | identical | pass/fail |
| C14 | Timeout: V not updated, m updated | unit test | pass/fail |

**Развёртка s_value** (results/p3_value_coupling_fine.json, 40 seeds, 500 проб):

| s | acc_gain | SE | recovery (проб) |
|---|---------|------|-----------------|
| 0.0 | 0.000 | 0.003 | 4.0 |
| 1.0 | +0.007 | 0.002 | 4.0 |
| 2.0 | +0.013 | 0.003 | 4.0 |
| **3.0** | **+0.017** | **0.002** | **4.1** |
| 4.0 | +0.013 | 0.003 | 4.1 |
| 5.0 | +0.013 | 0.003 | 4.3 |
| 8.0 | +0.009 | 0.003 | 4.8 |
| 10.0 | +0.001 | 0.004 | 5.0 |
| 15.0 | +0.000 | 0.005 | 5.8 |
| 20.0 | −0.022 | 0.008 | 7.0 |

**Оптимум: s ≈ 3** (+1.7% accuracy gain). Широкое плато s ∈ [2, 5]
(все в пределах 0.4% от пика). При s≥10 прирост исчезает. При s=20
точность ниже baseline. Recovery монотонно ухудшается от s≈3.5.

**Диапазон ручки:** s ∈ [0, 10]. Оптимум ≈ 3. Ухудшение accuracy
помечается в песочнице: зелёная зона [0, 5], жёлтая [5, 10],
красная >10. max_strength=20 по finiteness.

**Межмодульный результат (D29).** Немонотонная зависимость accuracy(s) —
первое явление платформы, отсутствующее у модулей по отдельности.
Слабый перенос выученных ценностей в решение улучшает точность;
сильный перебивает перцептивное свидетельство и замедляет восстановление
после реверсии. Ни DDM (нет обучения), ни RL (нет перцептивного сигнала)
не порождают этот эффект standalone.

**λ_crit(closed) = 1.1** (при s=5, 50% divergence). Standalone = 0.98.
Замкнутый контур стабилизирует петлю настроения на **+12%** (D30).
Механизм: value-based choices делают выбор более детерминированным →
V быстрее адаптируется к correct arm → RPE самокорректируется →
mood loop демпфируется.

## Отложенные наблюдаемые P3

Помечены held-out до проверки в конце P3. Не используются при настройке.

1. **Суточная модуляция accuracy DDM** (пара 1): accuracy выше при gain peak
   vs gain trough бодрствования. Ожидание: монотонная зависимость.
   Порог: |diff accuracy| > 0.01 (при s=1, средняя когерентность).

2. **Суточная модуляция win-stay RL** (пара 2): win-stay различается при
   gain peak vs gain trough. Порог: |diff win-stay| > 0.01 (при s=1).

3. **RL learning speed в композите vs standalone** (пара 3): число проб
   до 70% win-stay. Ожидание: в композите медленнее (DDM noise).
   Порог: отчётная (без pass/fail, так как нет целевого значения).

## Вырожденность (ρ, s)

drift_bias = s·(V₁−V₀). V сходится к ρ·p. Произведение s·ρ входит в drift
DDM неразличимо. Пары (ρ, s) с ρ·s = const дают 100% совпадение выборов
при V_init ∝ ρ (results/p3_rho_s_degeneracy.json). Соглашение ρ = 1 снимает
вырожденность одновременно с парой (ρ, β) — обоснование обновлено в
specs/rl_mood.md.

## Провал

Провал P3 — если любой pass/fail критерий не пройден и не может быть
починен без изменения модулей (а модули в P3 не меняются).

Если lambda_crit зависит от coupling strength неожиданным образом — записать
как ограничение, не как провал.
