# Синтетический бенчмарк задержки — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Команда `python bench.py`, которая без torch и моделей программно меряет сквозную задержку аудио-пайплайна (p50/p95/max, underruns, clamp-drops) и позволяет сравнивать «стало лучше/хуже» между изменениями кода.

**Architecture:** Микромир реального движка в реальном времени: настоящие `RingBuffer`, настоящий `Pipeline` (инференс подменён через существующий hook `Pipeline._chain` на identity-функцию со `time.sleep` заданной длительности), настоящий `Engine._processor_loop`. Поток-фидер пишет во входное кольцо пронумерованные сэмплы в темпе 48 кГц, поток-консьюмер читает выходное кольцо в том же темпе и по номеру сэмпла вычисляет его сквозную задержку. Сценарии задают длительность «инференса» на окно, включая разовые затыки.

**Tech Stack:** Python stdlib + numpy. Никаких новых зависимостей.

**Spec:** Диалог сессии VOICEBOX-OPTIMIZATIONS от 2026-11-06. Суть: (1) пользователь видел отставание звука до ~1 с; корень — «храповик» бэклога, пофикшен clamp'ом в `engine.py`; (2) нужен синтетический, программный показатель задержки, чтобы итеративно улучшать пайплайн (окна, prefill, clamp) без установки torch/моделей; (3) реальный режим (с моделями, CPU/RAM) — отдельная будущая работа, не в этом плане.

## Global Constraints

- Запуск без torch, scipy, sounddevice, моделей: только numpy (+ stdlib).
- Использовать реальный код: `audio_io.RingBuffer`, `pipeline.Pipeline`, `engine.Engine._processor_loop`, `engine.drop_stale_input`. Дублировать логику цикла в бенчмарке нельзя — иначе бенч меряет не то, что работает в проде.
- Prefill выходного кольца — как в `Engine.start()`: одно окно тишины.
- Сэмплы-маркеры: значения `BASE + i` (BASE = 1_000_000.0), float32 точно хранит целые до 2^24 ≈ 16.7 M → при 48 кГц хватает на ~5.5 мин симуляции; прогоны ≤ 60 с.
- Никаких комментариев в коде, кроме неочевидных инвариантов (правило репо).
- Тесты в `tests/`, запуск той же командой, что и остальные: `python -m pytest tests/ -q`.

## Review Focus

- Консьюмер стартует одновременно с фидером, но первые блоки — prefill-тишина (нули < BASE): они не должны попадать в метрики, иначе p50 искажён. → тест в Task 1.
- Underrun (из кольца пришло `None`): консьюмер обязан посчитать и продолжить, не упасть. → тест в Task 1 (сценарий overload).
- Затык длиннее ёмкости входного кольца (1 с): кольцо дропает новые сэмплы (overrun), нумерация в потоке рвётся — расчёт задержки по id обязан остаться корректным (id не из будущего). → spike-тест в Task 3.
- Джиттер `time.sleep` фидера/консьюмера (~1–3 мс на macOS) не должен валить регрессионные пороги: все пороги с запасом ≥ 50 мс. → пороговые значения в Task 3.
- Повторный запуск `run_scenario` в одном процессе не должен наследовать состояние (новые кольца, новый Pipeline). → тест Task 2 (матрица гоняет несколько прогонов подряд).

---

### Task 1: Ядро симуляции — `bench.py: run_scenario()`

**Files:**
- Create: `bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Produces:
  - `@dataclass ScenarioResult: p50_ms: float, p95_ms: float, max_ms: float, final_ms: float, underrun_samples: int, measured_samples: int`
  - `run_scenario(window_ms: int, crossfade_ms: int, infer_ms: float, duration_s: float, spike: tuple[float, float] | None = None, sr: int = 48000, blocksize: int = 480, warmup_s: float = 2.0) -> ScenarioResult`
    - `infer_ms` — длительность фейкового инференса на окно (реальный `time.sleep`).
    - `spike=(at_s, stall_ms)` — на первом окне после момента `at_s` инференс длится `stall_ms`.
    - `final_ms` — медиана задержки за последнюю секунду прогона (показатель «куда устаканилось»).
    - Метрики считаются только по сэмплам, съеденным после `warmup_s`.

- [ ] **Step 1: Написать падающие тесты ядра**

```python
import numpy as np
from bench import run_scenario

def test_steady_scenario_reports_sane_latency():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                     duration_s=3.0, warmup_s=1.0)
    assert r.measured_samples > 48000
    assert r.underrun_samples == 0
    assert 48 <= r.p50_ms <= 400
    assert r.p50_ms <= r.p95_ms <= r.max_ms

def test_overloaded_scenario_counts_underruns_without_crash():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=120,
                     duration_s=3.0, warmup_s=1.0)
    assert r.underrun_samples > 0
```

Нижняя граница `48 <= p50` — минимум структурной задержки (hop 48 мс при окне 64/16).

- [ ] **Step 2: Прогнать тесты, убедиться, что падают с `ImportError: bench`**

Run: `<testenv>/bin/python -m pytest tests/test_bench.py -q`
Expected: FAIL/ERROR — модуля `bench` нет.

- [ ] **Step 3: Реализовать `run_scenario` в `bench.py`**

Схема (решения, которые нельзя выбрать иначе):
- Собрать `Engine(EngineConfig(...))` вручную, НЕ вызывая `prepare()/start()`: подставить `engine._pipeline = Pipeline(denoiser=None, rvc=None, sr=sr, window_ms=..., crossfade_ms=..., denoise=False, bypass=False)` и `pipeline._chain = fake_chain`, где `fake_chain(x)` спит `infer_ms` (или `stall_ms` на spike-окне) и возвращает `(x, sr)`.
- `engine._audio_io` = минимальный объект с полями `input_ring`/`output_ring` (настоящие `RingBuffer` с ёмкостями как в `AudioIO`: 48000 / 96000) — например `types.SimpleNamespace`.
- Prefill: записать в `output_ring` окно нулей (как `Engine.start()`).
- Три потока: `engine._processor_loop` (настоящий метод), фидер, консьюмер. Фидер и консьюмер тикают по дедлайнам от общего `t0 = time.perf_counter()` (`target = t0 + k * blocksize / sr`, `sleep(max(0, target - now))`), а не накапливающимся `sleep(dt)`.
- Фидер пишет `np.arange(BASE + k*blocksize, BASE + (k+1)*blocksize, dtype=np.float32)`.
- Консьюмер: `chunk = output_ring.read(blocksize)`; `None` → `underruns += blocksize`; иначе для первого сэмпла чанка с `v >= BASE`: `lat_ms = (now - (t0 + (v - BASE) / sr)) * 1000`; после warmup копить в список. Сэмплы `< BASE` (prefill-тишина) пропускать.
- По завершении `duration_s`: `engine._stop_event.set()`, join всех потоков, собрать `ScenarioResult` (перцентили — `np.percentile`).

- [ ] **Step 4: Прогнать тесты ядра**

Run: `<testenv>/bin/python -m pytest tests/test_bench.py -q`
Expected: PASS (2 passed), ~6–8 с.

### Task 2: Матрица сценариев + CLI + сравнение с бейзлайном

**Files:**
- Modify: `bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `run_scenario`, `ScenarioResult` из Task 1.
- Produces:
  - `DEFAULT_MATRIX: list[dict]` — сценарии: для конфигов (384, 128) и (256, 64): `steady_fast` (infer 60% hop), `steady_slow` (infer 90% hop), `spike` (infer 60% hop + stall 800 мс на 4-й секунде); duration 12 с, warmup 2 с.
  - `results_to_dict(results: dict[str, ScenarioResult]) -> dict` — JSON-совместимый: `{"git_rev": ..., "timestamp": ..., "scenarios": {name: {метрики}}}`.
  - CLI: `python bench.py [--quick] [--save PATH] [--baseline PATH]`.
    - `--quick`: duration 5 с, warmup 1 с.
    - `--save`: записать JSON.
    - `--baseline`: напечатать дельты p50/p95/final против сохранённого JSON со знаком (`+хуже/−лучше`).

- [ ] **Step 1: Падающий тест сериализации и дельт**

```python
def test_results_round_trip_and_delta(tmp_path):
    from bench import results_to_dict, load_results, format_delta
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                     duration_s=2.0, warmup_s=0.5)
    d = results_to_dict({"steady": r})
    p = tmp_path / "base.json"
    p.write_text(json.dumps(d))
    base = load_results(p)
    out = format_delta({"steady": r}, base)
    assert "steady" in out and "+0.0" in out or "-0.0" in out or "0.0" in out
```

(Точную форму ассерта дельты имплементер уточняет по своей реализации `format_delta`; суть — дельта самого с собой ≈ 0.)

- [ ] **Step 2: Прогнать — FAIL (нет функций)**

- [ ] **Step 3: Реализовать матрицу, `results_to_dict`, `load_results`, `format_delta`, `main()` с argparse и таблицей**

Таблица: строка на сценарий, колонки p50/p95/max/final/underruns. `git_rev` — `git rev-parse --short HEAD` через subprocess, при ошибке `"unknown"`.

- [ ] **Step 4: Прогнать тесты — PASS. Запустить `bench.py --quick` руками, глазами проверить таблицу**

### Task 3: Регрессионные гейты — spike-восстановление и потолок задержки

**Files:**
- Modify: `tests/test_bench.py`

**Interfaces:**
- Consumes: `run_scenario` (spike-параметр).

- [ ] **Step 1: Написать тесты-гейты (сразу ожидаемо зелёные на текущем коде с clamp, но красные на коде до clamp — проверить это откатом `drop_stale_input` локально, см. Step 2)**

```python
def test_latency_recovers_after_stall():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                     duration_s=6.0, warmup_s=1.0,
                     spike=(2.0, 600.0))
    steady = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                          duration_s=3.0, warmup_s=1.0)
    assert r.final_ms <= steady.p95_ms + 150

def test_latency_never_exceeds_clamp_ceiling():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                     duration_s=6.0, warmup_s=1.0,
                     spike=(2.0, 600.0))
    assert r.max_ms <= 1200
```

- [ ] **Step 2: Red-green проверка гейта**

Временно закомментировать вызов `drop_stale_input` в `engine._processor_loop` → `test_latency_recovers_after_stall` обязан упасть (final_ms застревает выше). Вернуть как было → PASS. Это доказывает, что гейт ловит именно храповик.

- [ ] **Step 3: Полный прогон всего набора**

Run: `<testenv>/bin/python -m pytest tests/ -q`
Expected: все зелёные, суммарно ≤ ~40 с.

---

## Чего в этом плане нет (сознательно)

- Реальный режим с torch/моделями, замер CPU%/RAM инференса — следующая итерация, когда появится окружение с моделями.
- Отдельная папка результатов/история прогонов — `--save/--baseline` с явными путями достаточно.
