#!/usr/bin/env python3
"""Benchmark completo di CLM-8B (Qwen3-8B Metal last-token pooling + CLM_v0.1-8B.pt) su Mac M4."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import statistics
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "examples/t_rex")
sys.path.insert(0, "examples")

from clm.embedder import Embedder, l2
from clm.engine import Engine
from trex.planner import Planner, snapshot, ACTIONS
from trex.engine import Game, FPS, FRAME_MS
from trex.pilot import RESTART_DELAY_MS
from trex.backends import build_question

LLAMA_SERVER = "/Applications/Ollama.app/Contents/Resources/llama-server"
GGUF_PATH = "/Users/ingdifilippo/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
PORT = 8090


class LocalMetalEmbedder(Embedder):
    """Compatibile con llama-server --embeddings --pooling last su porta 8090."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session.trust_env = False

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        body = {"model": self.model, "input": texts, "encoding_format": "float"}
        r = self.session.post(self.url, json=body, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Embedder error {r.status_code}: {r.text[:300]}")
        j = r.json()
        out = [None] * len(texts)
        for d in j["data"]:
            v = np.asarray(d["embedding"], dtype=np.float32)
            out[d["index"]] = l2(v)
        tokens = int((j.get("usage") or {}).get("prompt_tokens", 0) or 0)
        return out, tokens


def start_metal_encoder():
    emb = LocalMetalEmbedder(url=f"http://127.0.0.1:{PORT}/v1/embeddings", model="qwen3-8b")
    try:
        r = emb.session.get(f"http://127.0.0.1:{PORT}/health", timeout=1.0)
        if r.status_code == 200:
            print(f"[encoder] Server Metal gia attivo sulla porta {PORT}")
            return None, emb
    except Exception:
        pass

    print(f"[encoder] Avvio llama-server (Apple M4 Metal GPU, --embeddings --pooling last) su porta {PORT}...")
    cmd = [
        LLAMA_SERVER,
        "-m", GGUF_PATH,
        "--embeddings",
        "--pooling", "last",
        "--port", str(PORT),
        "-c", "2048",
        "-ngl", "99",
        "--log-disable",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            r = emb.session.get(f"http://127.0.0.1:{PORT}/health", timeout=1.0)
            if r.status_code == 200:
                return proc, emb
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("Timeout nell'avvio di llama-server su porta 8090")


def main():
    print("=" * 72)
    print("1. AVVIO ENCODER QWEN3-8B (Apple M4 Metal, Last-Token Pooling) + CLM HEAD")
    print("=" * 72)

    t_boot0 = time.perf_counter()
    proc, emb = start_metal_encoder()
    try:
        vecs, tok = emb.embed(["Warmup state on Apple M4 Metal"])
        t_boot = (time.perf_counter() - t_boot0) * 1000
        print(f"Encoder pronto in {t_boot:.0f} ms | Dimensione vettore: {vecs.shape} (atteso: (1, 4096))")

        # Misura latenza pura di un singolo embedding su M4 Metal (senza cache)
        raw_times = []
        for i in range(15):
            t0 = time.perf_counter()
            emb._fetch([f"Dino runner game. 2 large cacti ahead, {80 + i} px away.\n\nChoose the best safe action for the dinosaur."])
            raw_times.append((time.perf_counter() - t0) * 1000)
        print(f"Latenza pura Encoder Qwen3-8B su M4 Metal (1 stato, ~22 token, cache miss):")
        print(f"  -> Mediana (p50): {statistics.median(raw_times):.2f} ms | Min: {min(raw_times):.2f} ms | p95: {np.percentile(raw_times, 95):.2f} ms")

        engine = Engine(
            embedder=emb,
            checkpoint="checkpoints/CLM_v0.1-8B.pt",
            device="cpu",
            action_cache="64MiB",
        )

        print("\n" + "=" * 72)
        print("2. TEST QUALITATIVO DI CLM-8B (Zero-Shot Decision Making)")
        print("=" * 72)

        ranked = engine.rank(
            "What causes tides on Earth?",
            [
                "The Moon's gravitational pull.",
                "Photosynthesis in plants.",
                "Because the Earth is round.",
            ],
        )
        print("Domanda: 'What causes tides on Earth?'")
        for item in ranked:
            print(f"  #{item['rank']}: {item['candidate']:<35} prob = {item['prob']:.4f}")

        r = engine.answer(
            state="Customer: my invoice was charged twice and nobody answers the phone!",
            questions={
                "urgency": {"type": "noul", "instructions": "Is this urgent?"},
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Charges, invoices, refunds",
                        "technical": "Bugs and outages",
                    },
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"],
                },
            },
        )
        print("\nStato: 'Customer: my invoice was charged twice and nobody answers the phone!'")
        print(f"  - Urgency (Noul):       {r['answers']['urgency']['noul']:.4f}")
        print(f"  - Department (Choice):  {r['answers']['department']['choice']} -> {r['answers']['department']['probabilities']}")
        print(f"  - Frustration (Score):  {r['answers']['frustration']['score']:.3f} / 2.000")

        print("\n" + "=" * 72)
        print("3. TEST SUL GIOCO T-REX (60 secondi = 3.600 frame @ 60 FPS, SENZA SCUDO)")
        print("=" * 72)

        planner = Planner()
        game = Game(seed=0)
        game.press_jump()

        total_frames = 60 * FPS  # 3600 frame (60s)
        decision_every = 4
        decisions = 0
        agreements = 0
        call_latencies_ms = []
        miss_latencies_ms = []
        hit_latencies_ms = []
        held = "run"

        t_game_start = time.perf_counter()
        for frame in range(total_frames):
            if game.crashed:
                game.step()
                if game.crash_ms >= RESTART_DELAY_MS:
                    game.restart()
                    held = "run"
                continue

            if game.playing and frame % decision_every == 0:
                snap = snapshot(game, held)
                plan = planner.plan(snap, (1, 1), (decision_every, decision_every))
                state, questions = build_question(plan, "labeled")

                misses_before = engine.arena.stats()["pools"]["512"]["misses"]
                t0 = time.perf_counter()
                ans = engine.answer(state, questions)
                dt_ms = (time.perf_counter() - t0) * 1000
                misses_after = engine.arena.stats()["pools"]["512"]["misses"]

                call_latencies_ms.append(dt_ms)
                if misses_after > misses_before:
                    miss_latencies_ms.append(dt_ms)
                else:
                    hit_latencies_ms.append(dt_ms)

                probs = ans["answers"]["action"]["probabilities"]
                proposed = max(ACTIONS, key=lambda a: (probs[a], -ACTIONS.index(a)))
                agreed = (proposed == plan.best) or (plan.airborne and {proposed, plan.best} <= {"jump", "run"})
                decisions += 1
                if agreed:
                    agreements += 1

                if proposed == "jump":
                    if not game.trex.jumping:
                        if game.trex.ducking:
                            game.release_duck()
                        game.press_jump()
                    held = "run"
                else:
                    held = proposed

            if game.playing and not game.crashed:
                if held == "duck":
                    if game.trex.jumping:
                        if not game.trex.speed_drop:
                            game.press_duck()
                    elif not game.trex.ducking:
                        game.press_duck()
                elif game.trex.speed_drop or (not game.trex.jumping and game.trex.ducking):
                    game.release_duck()

            game.step()

        wall_s = time.perf_counter() - t_game_start
        cache_stats = engine.arena.stats()["pools"]["512"]

        print(f"Partita da 60s (3.600 frame) completata in {wall_s:.2f} secondi reali!")
        print(f"  - Sopravvissuto 60s senza mai morire: {'SI (0 morti!)' if game.deaths == 0 else f'NO ({game.deaths} morti)'}")
        print(f"  - Punteggio finale raggiunto:         {game.score} punti")
        print(f"  - Decisioni prese da CLM-8B:          {decisions}")
        print(f"  - Accordo col Planner Ottimale:       {agreements}/{decisions} ({100.0 * agreements / max(1, decisions):.2f}%)")
        print(f"  - Latenza CLM complessiva (p50):      {statistics.median(call_latencies_ms):.3f} ms (media: {statistics.fmean(call_latencies_ms):.2f} ms)")
        if hit_latencies_ms:
            print(f"  - Latenza su Cache HIT ({len(hit_latencies_ms)} chiamate):  p50 = {statistics.median(hit_latencies_ms):.3f} ms")
        if miss_latencies_ms:
            print(f"  - Latenza su Cache MISS ({len(miss_latencies_ms)} chiamate): p50 = {statistics.median(miss_latencies_ms):.2f} ms (include Qwen3-8B su M4 Metal)")
        print(f"  - Statistiche VectorArena (512-d):    Hit Rate = {cache_stats['hit_rate']*100:.2f}% (Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']})")
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()
