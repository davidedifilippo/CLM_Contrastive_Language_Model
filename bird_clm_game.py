#!/usr/bin/env python3
"""
Gioco dell'Uccellino a Scorrimento (60 FPS) controllato in tempo reale da CLM (Contrastive-LM).

Caratteristiche:
- Usa la vera architettura del repository Contrastive-LM/CLM (`Engine`, `HeadPair`, `Choice`, `VectorArena`).
- All'avvio esegue in ~0.8s su Apple M4 un rapido addestramento contrastivo (InfoNCE loss) di una
  coppia `state_head` + `action_head` specifica per la fisica del gioco, salvandola in `checkpoints/bird_clm_head.pt`.
- Permette di cambiare modalita in tempo reale dalla finestra (tasti 1, 2, 3):
    [1] CLM Fast Head (InfoNCE + VectorArena, ~0.02 - 0.5 ms/frame @ 60 FPS)
    [2] CLM-8B Zero-Shot (Qwen3-8B Metal + CLM_v0.1-8B.pt in thread asincrono)
    [3] Giocatore Umano (Frecce Tastiera SU / GIU)
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from clm.embedder import Embedder, l2
from clm.engine import Engine
from clm.heads import HIDDEN, PROJ_DIM, make_head

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (
     QColor,
    QFont,
    QKeyEvent,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
)
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Costanti di gioco (60 FPS)
FPS = 60
GAME_W = 640
GAME_H = 480
GROUND_Y = 430
BIRD_X = 130
BIRD_R = 15
PIPE_W = 62
PIPE_GAP = 128
PIPE_SPEED = 3.6
PIPE_SPAWN_DIST = 260

ACTIONS = ("up", "down")
ACTION_CRITERIA = {
    "up": "Press UP key: flap wings and climb higher to gain altitude.",
    "down": "Press DOWN key: dive downwards to lose altitude quickly.",
}


def featurize_text_4096(text: str) -> np.ndarray:
    """
    Encoder deterministico ultra-rapido a 4096 dimensioni per gli stati del gioco.
    Estrae sia feature numeriche strutturate (se presenti nello stato) sia n-grammi lessicali,
    producendo un vettore L2-normalizzato compatibile con l'input a 4096-d delle HeadPair di CLM.
    """
    vec = np.zeros(HIDDEN, dtype=np.float32)
    t = text.lower()

    # Se e' una descrizione di azione ("up" vs "down")
    if "press up key" in t or "climb higher" in t:
        vec[0:64] = 1.0
        vec[64:128] = 0.5
    elif "press down key" in t or "dive downwards" in t:
        vec[0:64] = -1.0
        vec[64:128] = -0.5

    # Estrazione feature fisiche dallo stato testuale del frame
    # Formato stato: "Bird Y: <by> | Gap Y: <gy> | Delta Y: <dy> | Vel Y: <vy> | Dist X: <dx>"
    if "delta y:" in t:
        try:
            parts = {
                kv.split(":")[0].strip(): float(kv.split(":")[1].strip().split()[0])
                for kv in t.split("\n")[0].split("|")
                if ":" in kv
            }
            dy = parts.get("delta y", 0.0) / 100.0      # > 0 significa uccellino SOTTO il centro del varco -> serve UP?
            # Attenzione alle coordinate schermo: Y=0 e' in alto, Y=430 e' in basso!
            # Quindi se bird_y > gap_y (dy = bird_y - gap_y > 0), l'uccellino e' PIU IN BASSO del varco -> deve andare SU (UP, ridurre Y)!
            vy = parts.get("vel y", 0.0) / 6.0
            dx = parts.get("dist x", 200.0) / 300.0
            by = (parts.get("bird y", 215.0) - 215.0) / 215.0

            # Segnale di controllo fisico proiettato sui primi blocchi del vettore 4096-d
            need_up = np.tanh((dy * 1.8) + (vy * 0.45))
            vec[0:64] = need_up
            vec[64:128] = np.tanh(dy)
            vec[128:192] = np.tanh(vy)
            vec[192:256] = np.clip(dx, -1.0, 1.0)
            vec[256:320] = np.clip(by, -1.0, 1.0)
        except Exception:
            pass

    # Hash n-grammi sul resto delle dimensioni (320..4096) per preservare la struttura testuale
    for n in (3, 4):
        for i in range(0, max(0, len(t) - n + 1), 2):
            h = int.from_bytes(hashlib.blake2b(t[i : i + n].encode(), digest_size=4).digest(), "big")
            idx = 320 + (h % (HIDDEN - 320))
            vec[idx] += 0.15 if (h & 1) else -0.15

    return l2(vec)


class FastGameEmbedder(Embedder):
    """Embedder in-process a 4096-d per la modalita CLM Fast Head (latenza < 0.2 ms)."""
    def __init__(self):
        super().__init__(url="inproc://fast-4096", model="bird-4096")

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        out = [featurize_text_4096(t) for t in texts]
        tokens = sum(max(1, len(t) // 4) for t in texts)
        return out, tokens

    def healthy(self) -> bool:
        return True


def ensure_bird_clm_checkpoint(ckpt_path: str = "checkpoints/bird_clm_head.pt") -> str:
    """
    Addestra (in ~0.6s) una coppia state_head + action_head di CLM usando la loss InfoNCE
    sugli stati del gioco dell'uccellino, salvando un vero checkpoint compatibile con `clm.Engine`.
    """
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    if os.path.exists(ckpt_path):
        return ckpt_path

    print("[CLM-Train] Addestramento contrastivo (InfoNCE) della projection head per il gioco dell'uccellino...")
    t0 = time.perf_counter()
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    width, depth, proj = 512, 2, PROJ_DIM
    state_head = make_head(width=width, depth=depth, proj=proj, activation="gelu", layernorm=True, hidden=HIDDEN)
    action_head = make_head(width=width, depth=depth, proj=proj, activation="gelu", layernorm=True, hidden=HIDDEN)
    logit_scale = torch.tensor(math.log(25.0), requires_grad=True)

    # Costruiamo gli embedding delle 2 azioni ("up", "down")
    act_texts = [ACTION_CRITERIA["up"], ACTION_CRITERIA["down"]]
    act_embs = torch.from_numpy(np.stack([featurize_text_4096(t) for t in act_texts]))

    # Generiamo 400 stati sintetici rappresentativi di varie posizioni (bird_y, gap_y, vel_y, dist_x)
    states_list = []
    labels_list = []  # 0 = "up", 1 = "down"
    instr = "Choose UP or DOWN to fly through the center of the obstacle gap."
    for _ in range(400):
        gap_y = int(rng.integers(100, GROUND_Y - 100))
        bird_y = int(rng.integers(30, GROUND_Y - 20))
        vel_y = round(float(rng.uniform(-5.0, 5.0)), 1)
        dist_x = int(rng.integers(10, 280))
        dy = bird_y - gap_y
        s_txt = (
            f"Bird Y: {bird_y} | Gap Y: {gap_y} | Delta Y: {dy:+d} | Vel Y: {vel_y:+.1f} | Dist X: {dist_x}\n\n{instr}"
        )
        states_list.append(featurize_text_4096(s_txt))
        # Regola fisica ottima: se bird_y + vel_y*3 > gap_y (uccellino piu' basso del target su schermo), serve UP (0), altrimenti DOWN (1)
        target_action = 0 if (dy + vel_y * 2.5) > 0 else 1
        labels_list.append(target_action)

    X_states = torch.from_numpy(np.stack(states_list))
    y_labels = torch.tensor(labels_list, dtype=torch.long)

    opt = torch.optim.AdamW(
        list(state_head.parameters()) + list(action_head.parameters()) + [logit_scale],
        lr=3e-3,
        weight_decay=1e-4,
    )

    for step in range(80):
        opt.zero_grad()
        zs = F.normalize(state_head(X_states), dim=-1)
        za = F.normalize(action_head(act_embs), dim=-1)
        scale = logit_scale.exp().clamp(max=100.0)
        logits = scale * (zs @ za.T)  # [N, 2]
        loss = F.cross_entropy(logits, y_labels)
        loss.backward()
        opt.step()

    with torch.no_grad():
        zs = F.normalize(state_head(X_states), dim=-1)
        za = F.normalize(action_head(act_embs), dim=-1)
        preds = (zs @ za.T).argmax(dim=-1)
        acc = (preds == y_labels).float().mean().item() * 100.0

    ckpt = {
        "state_head": state_head.state_dict(),
        "action_head": action_head.state_dict(),
        "logit_scale": logit_scale.detach().cpu(),
        "projection_dim": proj,
        "cfg": {
            "model": "bird-4096",
            "hidden_size": HIDDEN,
            "projection_dim": proj,
            "width": width,
            "depth": depth,
            "activation": "gelu",
            "layernorm": True,
            "residual": False,
        },
    }
    torch.save(ckpt, ckpt_path)
    print(f"[CLM-Train] Head salvata in {ckpt_path} in {(time.perf_counter()-t0)*1000:.0f} ms (Accuracy: {acc:.1f}%)")
    return ckpt_path


class QwenMetalEmbedder(Embedder):
    """Connessione a llama-server (Qwen3-8B su Metal GPU, porta 8090) per la modalita Zero-Shot 8B."""
    def __init__(self, port: int = 8090):
        super().__init__(url=f"http://127.0.0.1:{port}/v1/embeddings", model="qwen3-8b", timeout=5.0)
        self.session.trust_env = False

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        body = {"model": self.model, "input": texts, "encoding_format": "float"}
        r = self.session.post(self.url, json=body, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        j = r.json()
        out = [None] * len(texts)
        for d in j["data"]:
            out[d["index"]] = l2(np.asarray(d["embedding"], dtype=np.float32))
        tokens = int((j.get("usage") or {}).get("prompt_tokens", 0) or 0)
        return out, tokens


class BirdGameWidget(QWidget):
    """Canvas grafico a 60 FPS che disegna l'uccellino, gli ostacoli a scorrimento e la traiettoria."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(GAME_W, GAME_H)
        self.reset_game()

    def reset_game(self):
        self.bird_y = float(GROUND_Y // 2)
        self.bird_vy = 0.0
        self.pipes: list[dict] = []
        self.clouds = [
            {"x": float(i * 170 + 40), "y": float(random.randint(35, 130)), "w": random.randint(60, 95)}
            for i in range(4)
        ]
        self.ground_offset = 0.0
        self.trail: deque[tuple[float, float]] = deque(maxlen=35)
        self.score = 0
        self.best_score = getattr(self, "best_score", 0)
        self.deaths = getattr(self, "deaths", 0)
        self.frame_count = 0
        self.current_action = "up"
        self.flash_frames = 0
        self._spawn_pipe(x_pos=380.0)
        self._spawn_pipe(x_pos=380.0 + PIPE_SPAWN_DIST)

    def _spawn_pipe(self, x_pos: float | None = None):
        if x_pos is None:
            last_x = max((p["x"] for p in self.pipes), default=float(GAME_W))
            x_pos = max(float(GAME_W + 20), last_x + PIPE_SPAWN_DIST)
        # Centro del varco tra 105 e GROUND_Y - 105
        gap_y = float(random.randint(105, GROUND_Y - 105))
        self.pipes.append({"x": x_pos, "gap_y": gap_y, "passed": False})

    def next_pipe(self) -> dict:
        for p in self.pipes:
            if p["x"] + PIPE_W >= BIRD_X - BIRD_R:
                return p
        return self.pipes[0]

    def step_physics(self, action: str):
        self.frame_count += 1
        self.current_action = action
        if self.flash_frames > 0:
            self.flash_frames -= 1

        # Controllo UP / DOWN fluido con inerzia
        target_vy = -4.2 if action == "up" else 4.2
        self.bird_vy = 0.72 * self.bird_vy + 0.28 * target_vy
        self.bird_y += self.bird_vy

        # Scorrimento nuvole e terreno
        self.ground_offset = (self.ground_offset - PIPE_SPEED) % 24
        for c in self.clouds:
            c["x"] -= 0.7
            if c["x"] + c["w"] < 0:
                c["x"] = float(GAME_W + 30)
                c["y"] = float(random.randint(35, 130))

        # Scorrimento ostacoli (tubi)
        for p in self.pipes:
            p["x"] -= PIPE_SPEED
            if not p["passed"] and p["x"] + PIPE_W < BIRD_X - BIRD_R:
                p["passed"] = True
                self.score += 1
                if self.score > self.best_score:
                    self.best_score = self.score

        if self.pipes and self.pipes[0]["x"] + PIPE_W < -10:
            self.pipes.pop(0)
            self._spawn_pipe()

        # Aggiorna scia dell'uccellino
        for i in range(len(self.trail)):
            tx, ty = self.trail[i]
            self.trail[i] = (tx - PIPE_SPEED, ty)
        self.trail.append(( float(BIRD_X), self.bird_y ))

        # Controllo collisioni (soffitto, suolo, tubi)
        crashed = False
        if self.bird_y - BIRD_R < 8 or self.bird_y + BIRD_R > GROUND_Y:
            crashed = True
        for p in self.pipes:
            if (BIRD_X + BIRD_R - 3 > p["x"]) and (BIRD_X - BIRD_R + 3 < p["x"] + PIPE_W):
                top_edge = p["gap_y"] - PIPE_GAP / 2
                bot_edge = p["gap_y"] + PIPE_GAP / 2
                if (self.bird_y - BIRD_R + 3 < top_edge) or (self.bird_y + BIRD_R - 3 > bot_edge):
                    crashed = True
                    break

        if crashed:
            self.deaths += 1
            self.score = 0
            self.flash_frames = 10
            np_pipe = self.next_pipe()
            self.bird_y = np_pipe["gap_y"]
            self.bird_vy = 0.0
            self.trail.clear()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Cielo sfumato
        sky = QLinearGradient(0, 0, 0, GROUND_Y)
        if self.flash_frames > 0:
            sky.setColorAt(0.0, QColor(255, 140, 140))
            sky.setColorAt(1.0, QColor(255, 200, 200))
        else:
            sky.setColorAt(0.0, QColor(30, 144, 210))
            sky.setColorAt(1.0, QColor(175, 232, 255))
        painter.fillRect(0, 0, GAME_W, GROUND_Y, sky)

        # 2. Nuvole
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 195))
        for c in self.clouds:
            painter.drawRoundedRect(QRectF(c["x"], c["y"], c["w"], 26), 13, 13)
            painter.drawEllipse(QRectF(c["x"] + 12, c["y"] - 10, c["w"] * 0.55, 24))

        # 3. Linea guida verso il centro del prossimo varco (target visivo del modello)
        np_pipe = self.next_pipe()
        target_x = np_pipe["x"] + PIPE_W / 2
        target_y = np_pipe["gap_y"]
        dash_pen = QPen(QColor(255, 255, 255, 130), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(dash_pen)
        painter.drawLine(QPointF(BIRD_X, self.bird_y), QPointF(target_x, target_y))
        painter.setBrush(QColor(255, 235, 59, 200))
        painter.setPen(QPen(QColor(40, 40, 40, 160), 1.2))
        painter.drawEllipse(QPointF(target_x, target_y), 5, 5)

        # 4. Ostacoli (Tubi verdi con bordo e riflesso)
        for p in self.pipes:
            px = p["x"]
            top_h = p["gap_y"] - PIPE_GAP / 2
            bot_y = p["gap_y"] + PIPE_GAP / 2
            bot_h = GROUND_Y - bot_y

            pipe_grad = QLinearGradient(px, 0, px + PIPE_W, 0)
            pipe_grad.setColorAt(0.0, QColor(46, 160, 67))
            pipe_grad.setColorAt(0.35, QColor(102, 217, 118))
            pipe_grad.setColorAt(1.0, QColor(34, 120, 50))

            painter.setPen(QPen(QColor(20, 75, 30), 2))
            painter.setBrush(pipe_grad)
            # Tubo superiore + bocchetta
            if top_h > 0:
                painter.drawRect(QRectF(px, 0, PIPE_W, top_h))
                painter.drawRoundedRect(QRectF(px - 4, max(0.0, top_h - 18), PIPE_W + 8, 18), 4, 4)
            # Tubo inferiore + bocchetta
            if bot_h > 0:
                painter.drawRect(QRectF(px, bot_y, PIPE_W, bot_h))
                painter.drawRoundedRect(QRectF(px - 4, bot_y, PIPE_W + 8, 18), 4, 4)

        # 5. Scia della traiettoria dell'uccellino
        if len(self.trail) > 2:
            for i in range(1, len(self.trail)):
                alpha = int(170 * (i / len(self.trail)))
                painter.setPen(QPen(QColor(255, 230, 80, alpha), 2.5))
                painter.drawLine(QPointF(*self.trail[i - 1]), QPointF(*self.trail[i]))

        # 6. Uccellino
        painter.save()
        painter.translate(BIRD_X, self.bird_y)
        angle = max(-28.0, min(28.0, self.bird_vy * 4.5))
        painter.rotate(angle)

        # Corpo
        painter.setPen(QPen(QColor(60, 40, 0), 2))
        painter.setBrush(QColor(255, 208, 38))
        painter.drawEllipse(QPointF(0, 0), BIRD_R + 2, BIRD_R - 1)

        # Ala animata in base a UP / DOWN
        wing_y = -5 if (self.current_action == "up" and (self.frame_count // 3) % 2 == 0) else 3
        painter.setBrush(QColor(255, 242, 160))
        painter.drawEllipse(QPointF(-3, wing_y), 8, 5)

        # Occhio
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPointF(6, -4), 5, 5)
        painter.setBrush(QColor(20, 20, 20))
        painter.drawEllipse(QPointF(7.5, -4), 2.3, 2.3)

        # Becco
        beak = QPainterPath()
        beak.moveTo(BIRD_R, -3)
        beak.lineTo(BIRD_R + 10, 1)
        beak.lineTo(BIRD_R, 5)
        beak.closeSubpath()
        painter.setBrush(QColor(245, 95, 34))
        painter.drawPath(beak)
        painter.restore()

        # 7. Terreno a scorrimento
        painter.setPen(QPen(QColor(40, 90, 30), 2))
        painter.setBrush(QColor(115, 195, 75))
        painter.drawRect(QRectF(0, GROUND_Y, GAME_W, 14))
        painter.setBrush(QColor(218, 198, 138))
        painter.drawRect(QRectF(0, GROUND_Y + 14, GAME_W, GAME_H - GROUND_Y - 14))
        painter.setPen(QPen(QColor(85, 155, 50), 2))
        for x in range(int(self.ground_offset) - 24, GAME_W + 24, 24):
            painter.drawLine(x, GROUND_Y + 2, x + 10, GROUND_Y + 12)

        # 8. Overlay HUD compatto in alto a sinistra sul canvas
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 175))
        painter.drawRoundedRect(QRectF(12, 12, 235, 54), 8, 8)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
        painter.drawText(24, 34, f"SCORE: {self.score}   BEST: {self.best_score}")
        painter.setFont(QFont("Menlo", 10))
        painter.setPen(QColor(180, 220, 255))
        painter.drawText(24, 54, f"Morti: {self.deaths}   Frame: {self.frame_count}")


class MainWindow(QMainWindow):
    def __init__(self, fast_engine: Engine, qwen_engine: Engine | None, llama_proc):
        super().__init__()
        self.fast_engine = fast_engine
        self.qwen_engine = qwen_engine
        self.llama_proc = llama_proc

        # Modalita: 1 = CLM Fast Head (60FPS), 2 = CLM-8B Qwen3 Metal, 3 = Manuale
        self.mode = 1
        self.manual_action = "down"

        # Statistiche live CLM
        self.last_probs = {"up": 0.5, "down": 0.5}
        self.last_choice = "up"
        self.last_latency_ms = 0.0
        self.last_state_str = ""
        self.qwen_busy = False

        self.setWindowTitle("CLM (Contrastive Language Model) — Real-Time Bird Player [Apple M4]")
        self.setStyleSheet("background-color: #0f172a; color: #f8fafc;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(16)

        # Sinistra: Canvas del gioco
        self.canvas = BirdGameWidget()
        layout.addWidget(self.canvas)

        # Destra: Pannello di telemetria CLM in tempo reale
        panel = QVBoxLayout()
        panel.setSpacing(10)
        layout.addLayout(panel)

        title = QLabel("TELEMETRIA CLM LIVE")
        title.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #38bdf8;")
        panel.addWidget(title)

        # Selettore modalita
        self.btn_m1 = QPushButton("[1] CLM Fast Head (0.03 ms @ 60FPS)")
        self.btn_m2 = QPushButton("[2] CLM-8B Zero-Shot (Qwen3-8B Metal)")
        self.btn_m3 = QPushButton("[3] Giocatore Umano (Frecce SU/GIU)")
        for idx, btn in enumerate((self.btn_m1, self.btn_m2, self.btn_m3), start=1):
            btn.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda checked, m=idx: self.set_mode(m))
            panel.addWidget(btn)

        # Indicatore visivo dei tasti UP / DOWN premuti da CLM
        keys_box = QHBoxLayout()
        self.lbl_key_up = QLabel("▲ TASTO UP")
        self.lbl_key_down = QLabel("▼ TASTO DOWN")
        for lbl in (self.lbl_key_up, self.lbl_key_down):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
            lbl.setFixedHeight(44)
            keys_box.addWidget(lbl)
        panel.addLayout(keys_box)

        # Box Stato ed Embedding
        self.lbl_state = QLabel()
        self.lbl_state.setFont(QFont("Menlo", 10))
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setStyleSheet(
            "background-color: #1e293b; padding: 10px; border-radius: 6px; color: #cbd5e1;"
        )
        panel.addWidget(self.lbl_state)

        # Box Probabilita e Latenza
        self.lbl_metrics = QLabel()
        self.lbl_metrics.setFont(QFont("Menlo", 11))
        self.lbl_metrics.setStyleSheet(
            "background-color: #1e293b; padding: 10px; border-radius: 6px; color: #e2e8f0;"
        )
        panel.addWidget(self.lbl_metrics)

        panel.addStretch(1)

        footer = QLabel("Premi 1, 2, 3 per cambiare giocatore\nIn modalita [3] usa Freccia SU / GIU")
        footer.setFont(QFont("Menlo", 10))
        footer.setStyleSheet("color: #94a3b8;")
        panel.addWidget(footer)

        self.set_mode(1)

        # Timer 60 FPS (16 ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / FPS))

    def set_mode(self, mode: int):
        self.mode = mode
        active_css = "background-color: #0284c7; color: white; padding: 8px; border-radius: 6px;"
        idle_css = "background-color: #334155; color: #cbd5e1; padding: 8px; border-radius: 6px;"
        self.btn_m1.setStyleSheet(active_css if mode == 1 else idle_css)
        self.btn_m2.setStyleSheet(active_css if mode == 2 else idle_css)
        self.btn_m3.setStyleSheet(active_css if mode == 3 else idle_css)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_1:
            self.set_mode(1)
        elif event.key() == Qt.Key.Key_2:
            self.set_mode(2)
        elif event.key() == Qt.Key.Key_3:
            self.set_mode(3)
        elif event.key() in (Qt.Key.Key_Up, Qt.Key.Key_W, Qt.Key.Key_Space):
            self.manual_action = "up"
        elif event.key() in (Qt.Key.Key_Down, Qt.Key.Key_S):
            self.manual_action = "down"

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_W, Qt.Key.Key_Space):
            self.manual_action = "down"

    def build_frame_state_and_question(self):
        p = self.canvas.next_pipe()
        # Quantizziamo alle decine/unita per massimizzare il riutilizzo nella VectorArena cache di CLM
        by = int(round(self.canvas.bird_y / 4.0) * 4)
        gy = int(round(p["gap_y"] / 4.0) * 4)
        dy = by - gy
        vy = round(self.canvas.bird_vy, 0)
        dx = int(round(max(0.0, p["x"] - BIRD_X) / 8.0) * 8)

        state_str = f"Bird Y: {by} | Gap Y: {gy} | Delta Y: {dy:+d} | Vel Y: {vy:+.1f} | Dist X: {dx}"
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Choose UP or DOWN to fly through the center of the obstacle gap.",
                "criteria": ACTION_CRITERIA,
            }
        }
        return state_str, questions

    def _ask_qwen_async(self, state_str: str, questions: dict):
        try:
            t0 = time.perf_counter()
            out = self.qwen_engine.answer(state_str, questions)
            dt = (time.perf_counter() - t0) * 1000
            ans = out["answers"]["move"]
            self.last_probs = ans["probabilities"]
            self.last_choice = ans["choice"]
            self.last_latency_ms = dt
        except Exception:
            pass
        finally:
            self.qwen_busy = False

    def tick(self):
        state_str, questions = self.build_frame_state_and_question()
        self.last_state_str = state_str

        if self.mode == 1:
            t0 = time.perf_counter()
            out = self.fast_engine.answer(state_str, questions)
            self.last_latency_ms = (time.perf_counter() - t0) * 1000
            ans = out["answers"]["move"]
            self.last_probs = ans["probabilities"]
            self.last_choice = ans["choice"]
            action = self.last_choice
        elif self.mode == 2:
            if self.qwen_engine is not None and not self.qwen_busy:
                self.qwen_busy = True
                threading.Thread(
                    target=self._ask_qwen_async, args=(state_str, questions), daemon=True
                ).start()
            action = self.last_choice
        else:
            action = self.manual_action
            self.last_choice = action
            self.last_probs = {"up": 1.0 if action == "up" else 0.0, "down": 1.0 if action == "down" else 0.0}
            self.last_latency_ms = 0.0

        # Avanza fisica del gioco
        self.canvas.step_physics(action)
        self.canvas.update()
        self.update_telemetry_ui()

    def update_telemetry_ui(self):
        # Evidenzia il pulsante UP o DOWN attivo
        if self.last_choice == "up":
            self.lbl_key_up.setStyleSheet(
                "background-color: #22c55e; color: #052e16; border-radius: 6px; padding: 6px;"
            )
            self.lbl_key_down.setStyleSheet(
                "background-color: #1e293b; color: #64748b; border-radius: 6px; padding: 6px;"
            )
        else:
            self.lbl_key_up.setStyleSheet(
                "background-color: #1e293b; color: #64748b; border-radius: 6px; padding: 6px;"
            )
            self.lbl_key_down.setStyleSheet(
                "background-color: #f97316; color: #431407; border-radius: 6px; padding: 6px;"
            )

        self.lbl_state.setText(
            f"STATO DEL FRAME (Input a CLM):\n{self.last_state_str}\n\n"
            f"AZIONI CANDIDATE (Cached in Arena):\n"
            f"• up:   climb higher\n"
            f"• down: dive downwards"
        )

        p_up = self.last_probs.get("up", 0.0)
        p_dn = self.last_probs.get("down", 0.0)
        bar_up = "█" * int(round(p_up * 18)) + "░" * (18 - int(round(p_up * 18)))
        bar_dn = "█" * int(round(p_dn * 18)) + "░" * (18 - int(round(p_dn * 18)))

        active_eng = self.fast_engine if self.mode != 2 else (self.qwen_engine or self.fast_engine)
        pool_stats = active_eng.arena.stats()["pools"]["512"]
        hit_rate = (pool_stats["hit_rate"] or 0.0) * 100.0

        self.lbl_metrics.setText(
            f"PROBABILITA CLM (Dot Product + Softmax):\n"
            f"P(UP)   [{bar_up}] {p_up*100:5.1f}%\n"
            f"P(DOWN) [{bar_dn}] {p_dn*100:5.1f}%\n\n"
            f"Decisione:      {self.last_choice.upper()}\n"
            f"Latenza CLM:    {self.last_latency_ms:.3f} ms\n"
            f"Arena Hit Rate: {hit_rate:.1f}% ({pool_stats['hits']} hits / {pool_stats['misses']} miss)"
        )

    def closeEvent(self, event):
        if self.llama_proc is not None:
            try:
                self.llama_proc.terminate()
            except Exception:
                pass
        super().closeEvent(event)


def start_optional_qwen_server() -> subprocess.Popen | None:
    llama_bin = "/Applications/Ollama.app/Contents/Resources/llama-server"
    gguf = "/Users/ingdifilippo/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
    if not (os.path.exists(llama_bin) and os.path.exists(gguf)):
        return None
    try:
        cmd = [
            llama_bin,
            "-m", gguf,
            "--embeddings",
            "--pooling", "last",
            "--port", "8090",
            "-c", "2048",
            "-ngl", "99",
            "--log-disable",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None


def main():
    ckpt_bird = ensure_bird_clm_checkpoint("checkpoints/bird_clm_head.pt")
    fast_engine = Engine(
        embedder=FastGameEmbedder(),
        checkpoint=ckpt_bird,
        device="cpu",
        action_cache="32MiB",
    )

    llama_proc = start_optional_qwen_server()
    qwen_engine = None
    if os.path.exists("checkpoints/CLM_v0.1-8B.pt"):
        qwen_engine = Engine(
            embedder=QwenMetalEmbedder(port=8090),
            checkpoint="checkpoints/CLM_v0.1-8B.pt",
            device="cpu",
            action_cache="32MiB",
        )

    app = QApplication(sys.argv)
    win = MainWindow(fast_engine, qwen_engine, llama_proc)
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
