# Источники: модуль возбуждения (arousal)

## Первоисточники

| Ref | Цитата | DOI | Роль |
|-----|--------|-----|------|
| PR07 | Phillips AJK, Robinson PA (2007). A quantitative model of sleep-wake dynamics based on the physiology of the brainstem ascending arousal system. *J Biol Rhythms* 22(2):167-179. | 10.1177/0748730406297512 | **Первоисточник уравнений** |
| PR08 | Phillips AJK, Robinson PA (2008). Sleep deprivation in a quantitative physiologically based model. *J Theor Biol* 255(4):413-423. | 10.1016/j.jtbi.2008.08.022 | Расширение: депривация, фазовый сдвиг |
| SS90 | Servan-Schreiber D, Printz H, Cohen JD (1990). A network model of catecholamine effects. *Science* 249:892-895. | 10.1126/science.2392679 | Функция усиления g → сигмоида |
| AJC05 | Aston-Jones G, Cohen JD (2005). An integrative theory of locus coeruleus-norepinephrine function. *Annu Rev Neurosci* 28:403-450. | 10.1146/annurev.neuro.28.061604.135709 | Концепция: LC-NE → глобальный gain; без уравнений |
| Saper05 | Saper CB, Scammell TE, Lu J (2005). Hypothalamic regulation of sleep and circadian rhythms. *Nature* 437:1257-1263. | 10.1038/nature04284 | Концепция flip-flop; без уравнений |
| Skeldon14 | Skeldon AC, Dijk DJ, Derks G (2014). Mathematical models for sleep-wake dynamics. *PLOS ONE* 9(8):e103877. | 10.1371/journal.pone.0103877 | Обзор: сравнение PR и two-process; обоснование редукции |

## Верификация

Оригинальные статьи PR07 и PR08 за paywall. Уравнения и параметры верифицированы по
**трём независимым депонированным реализациям** оригинала:
- CellML implementation (models.cellml.org, deposited by авторской группой)
- BioModels BIOMD0000000917
- sleepR R package (Rodriguez-Sanchez, derived from PR07)

Все три дают идентичные уравнения и параметры. Помечено как **"verified from deposited
implementation of original"** — не из самого журнала, но из авторизованных репозиториев.

## Полная система PR07 (3 переменных состояния)

### Переменные

| Переменная | Смысл | Единицы |
|-----------|-------|---------|
| V_v | Средний мембранный потенциал VLPO (сон) | mV |
| V_m | Средний мембранный потенциал MA (бодрствование) | mV |
| H | Гомеостатическое давление сна (сомноген, аденозин) | nM |

### ОДУ

```
dV_v/dt = [v_vm * Q(V_m) + D(t) - V_v] / tau_v
dV_m/dt = [A + v_mv * Q(V_v) - V_m] / tau_m
dH/dt   = [mu * Q(V_m) - H] / chi
```

Сигмоида:
```
Q(V) = Q_max / (1 + exp(-(V - theta) / sigma_q))
```

Привод VLPO:
```
D(t) = v_vc * C(t) + v_vh * H
C(t) = c0 + cos(omega * (t - alpha))
```

Постоянный холинергический вход:
```
A = v_ma * Q(V_a0) ≈ 1.0 mV    (V_a0 = 1.0, без ОДУ)
```

### Параметры PR07

| Параметр | Символ | Значение | Единицы | Источник |
|----------|--------|----------|---------|----------|
| Постоянная VLPO | tau_v | 10 | s | PR07 CellML |
| Постоянная MA | tau_m | 10 | s | PR07 CellML |
| Ингибиция MA→VLPO | v_vm | -1.9 | mV·s | PR07 CellML |
| Ингибиция VLPO→MA | v_mv | -1.9 | mV·s | PR07 CellML |
| Тонический вход MA | A | 1.0 | mV | PR07 CellML |
| Циркадная связь | v_vc | -6.3 (PR07), -3.2 (PR08) | mV | PR07/PR08 CellML |
| Гомеостатическая связь | v_vh | 0.19 | mV/nM | PR07 CellML |
| Max firing rate | Q_max | 100 | s^-1 | PR07 CellML |
| Порог сигмоиды | theta | 10 | mV | PR07 CellML |
| Наклон сигмоиды | sigma_q | 3 | mV | PR07 CellML |
| Постоянная сомногена | chi | 10.8 (PR07), 11.0 (PR08) | h | PR07/PR08 CellML |
| Скорость производства | mu | 3.6 | nM·s | PR07 CellML |
| Циркадный сдвиг | c0 | 1.0 | безразмерный | PR07 CellML |
| Циркадная частота | omega | 2*pi/24 | h^-1 | PR07 CellML |
| Фаза цикла | alpha | 0 (PR07), 16 (PR08) | h | PR07/PR08 CellML |

### Различия PR07 → PR08

- v_vc: -6.3 → -3.2 mV
- chi: 10.8 → 11.0 h
- alpha: 0 → 16 h (фазовый сдвиг)

## Редукция до 2 переменных (Skeldon 2014) — ОТВЕРГНУТА

tau_v = tau_m = 10s, но V_v реагирует на изменения V_m квази-мгновенно на масштабе H.
Устанавливаем dV_v/dt = 0:

```
V_v* = v_vm * Q(V_m) + D(t)
```

Подставляем в уравнение V_m:

```
dV_m/dt = [A + v_mv * Q(v_vm * Q(V_m) + D(t)) - V_m] / tau_m
dH/dt   = [mu * Q(V_m) - H] / chi
```

**Результат при параметрах PR08:** бистабильность уничтожена. Система плавно
следит за единственным равновесием, flip-flop переключений нет. Причина: адиабатическое
исключение требует разделения масштабов, но τ_v = τ_m = 10 с — разделения нет.
Подробности: reports/alternatives.md, A1.

**Решение:** используется полная 3-переменная система PR07.

## Gain output

Связь с DDM через относительную частоту MA:

```
g = Q(V_m) / Q_M_WAKE_MEAN_BASELINE
```

Q_M_WAKE_MEAN_BASELINE = 4.3318 s^{-1} — средняя частота MA во время бодрствования
(trimmed ±30 мин) при **baseline** параметрах. Вычисляется однократно калибровочным
прогоном (48 ч, детерминированный). Фиксированная константа, не пересчитывается
при изменении ручек. Gain ≈ 1.0 при baseline, смещается с параметрами: g_mean от
0.67 (v_vm=-0.50) до 1.21 (A=1.5). Во сне: gain ≈ 0.01-0.03.

Обоснование нормировки:
- SS90: катехоламины модулируют наклон сигмоиды (gain) — относительное, не абсолютное
- AJC05: LC-NE задаёт глобальный gain — тоническая модуляция, baseline-relative
- PR07: Q(V_m) кодирует моноаминергическое возбуждение, включая LC

## Масштабы времени

| Масштаб | Значение | Что управляет |
|---------|----------|---------------|
| Нейрональный (tau_m) | 10 s | Скорость flip-flop перехода |
| Гомеостатический (chi) | ~11 h | Накопление/спад давления сна |
| Циркадный (T) | 24 h | Внешний ритм |

Жёсткость: tau_m / chi ≈ 4000. Требует multi-tempo интегрирования.

## Шум (не в оригинале)

PR07 — детерминированная модель. Для СДУ-фреймворка необходимо добавить шум.
Естественное место: аддитивный на V_m (флуктуации возбуждения) и/или на H
(флуктуации гомеостатического давления). Амплитуда — свободный параметр,
калибруемый по вариабельности сна между ночами.
