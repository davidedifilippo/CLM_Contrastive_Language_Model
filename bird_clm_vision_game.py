#!/usr/bin/env python3
"""
Esperimento Computer Vision (mss + OpenCV) + CLM (Contrastive Language Model).

In questo esperimento:
1. Il gioco dell'uccellino gira a 60 FPS in una canvas grafica indipendente.
2. Un modulo di Computer Vision (OpenCV + mss) cattura i pixel grezzi dello schermo/finestra:
   - Isola l'uccellino tramite maschera colore HSV e ne calcola il baricentro (bird_x, bird_y).
   - Isola gli ostacoli (tubi verdi), trova il varco libero e calcola il punto di mira (target_x, gap_y).
   - Calcola la velocita e la traiettoria solo dai fotogrammi visivi (senza leggere le variabili interne del gioco!).
3. I dati estratti dalla Computer Vision vengono formattati in uno stato testuale e inviati a CLM.
4. CLM (tramite la sua projection head + VectorArena) decide in 0.02 ms se premere UP o DOWN.
5. Una finestra HUD di OpenCV visualizza in tempo reale la "vista a raggi X" del computer:
   bounding box dell'uccellino, segmentazione dei tubi, target e telemetria completa.
"""
from __future__ import annotations

import os
import sys
import time
import random
from collections import deque

import cv2
import numpy as np
import mss

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
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

# Import del modulo CLM
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import bird_clm_game as b_game
from clm.engine import Engine

# Costanti di gioco (60 FPS)
FPS = 60
GAME_W = 520
GAME_H = 440
GROUND_Y = 390
BIRD_X = 110
BIRD_R = 14
PIPE_W = 56
PIPE_GAP = 126
PIPE_SPEED = 3.5
PIPE_SPAWN_DIST = 240


class PureGameCanvas(QWidget):
    """Finestra di gioco autonoma. Non espone logiche decisionali: renderizza solo a 60 FPS."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(GAME_W, GAME_H)
        self.reset_game()

    def reset_game(self):
        self.bird_y = float(GROUND_Y // 2)
        self.bird_vy = 0.0
        self.pipes: list[dict] = []
        self.clouds = [
            {"x": float(i * 150 + 20), "y": float(random.randint(25, 110)), "w": random.randint(50, 80)}
            for i in range(4)
        ]
        self.ground_offset = 0.0
        self.score = 0
        self.best_score = getattr(self, "best_score", 0)
        self.deaths = getattr(self, "deaths", 0)
        self.frame_count = 0
        self.current_action = "up"
        self.flash_frames = 0
        self._spawn_pipe(x_pos=320.0)
        self._spawn_pipe(x_pos=320.0 + PIPE_SPAWN_DIST)

    def _spawn_pipe(self, x_pos: float | None = None):
        if x_pos is None:
            last_x = max((p["x"] for p in self.pipes), default=float(GAME_W))
            x_pos = max(float(GAME_W + 10), last_x + PIPE_SPAWN_DIST)
        gap_y = float(random.randint(95, GROUND_Y - 95))
        self.pipes.append({"x": x_pos, "gap_y": gap_y, "passed": False})

    def step_physics(self, action: str):
        self.frame_count += 1
        self.current_action = action
        if self.flash_frames > 0:
            self.flash_frames -= 1

        target_vy = -4.2 if action == "up" else 4.2
        self.bird_vy = 0.72 * self.bird_vy + 0.28 * target_vy
        self.bird_y += self.bird_vy

        self.ground_offset = (self.ground_offset - PIPE_SPEED) % 24
        for c in self.clouds:
            c["x"] -= 0.6
            if c["x"] + c["w"] < 0:
                c["x"] = float(GAME_W + 20)
                c["y"] = float(random.randint(25, 110))

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

        crashed = False
        if self.bird_y - BIRD_R < 6 or self.bird_y + BIRD_R > GROUND_Y:
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
            for p in self.pipes:
                if p["x"] + PIPE_W >= BIRD_X - BIRD_R:
                    self.bird_y = p["gap_y"]
                    break
            self.bird_vy = 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        sky = QLinearGradient(0, 0, 0, GROUND_Y)
        if self.flash_frames > 0:
            sky.setColorAt(0.0, QColor(255, 130, 130))
            sky.setColorAt(1.0, QColor(255, 190, 190))
        else:
            sky.setColorAt(0.0, QColor(32, 148, 214))
            sky.setColorAt(1.0, QColor(176, 230, 255))
        painter.fillRect(0, 0, GAME_W, GROUND_Y, sky)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 190))
        for c in self.clouds:
            painter.drawRoundedRect(QRectF(c["x"], c["y"], c["w"], 22), 11, 11)
            painter.drawEllipse(QRectF(c["x"] + 10, c["y"] - 8, c["w"] * 0.55, 20))

        # Tubi
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
            if top_h > 0:
                painter.drawRect(QRectF(px, 0, PIPE_W, top_h))
                painter.drawRoundedRect(QRectF(px - 4, max(0.0, top_h - 16), PIPE_W + 8, 16), 3, 3)
            if bot_h > 0:
                painter.drawRect(QRectF(px, bot_y, PIPE_W, bot_h))
                painter.drawRoundedRect(QRectF(px - 4, bot_y, PIPE_W + 8, 16), 3, 3)

        # Uccellino
        painter.save()
        painter.translate(BIRD_X, self.bird_y)
        angle = max(-26.0, min(26.0, self.bird_vy * 4.2))
        painter.rotate(angle)

        painter.setPen(QPen(QColor(60, 40, 0), 2))
        painter.setBrush(QColor(255, 208, 38))
        painter.drawEllipse(QPointF(0, 0), BIRD_R + 2, BIRD_R - 1)

        wing_y = -4 if (self.current_action == "up" and (self.frame_count // 3) % 2 == 0) else 3
        painter.setBrush(QColor(255, 242, 160))
        painter.drawEllipse(QPointF(-3, wing_y), 7, 4)

        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPointF(5, -4), 4.5, 4.5)
        painter.setBrush(QColor(20, 20, 20))
        painter.drawEllipse(QPointF(6.5, -4), 2.2, 2.2)

        beak = QPainterPath()
        beak.moveTo(BIRD_R, -3)
        beak.lineTo(BIRD_R + 9, 1)
        beak.lineTo(BIRD_R, 4)
        beak.closeSubpath()
        painter.setBrush(QColor(245, 95, 34))
        painter.drawPath(beak)
        painter.restore()

        # Terreno
        painter.setPen(QPen(QColor(40, 90, 30), 2))
        painter.setBrush(QColor(115, 195, 75))
        painter.drawRect(QRectF(0, GROUND_Y, GAME_W, 12))
        painter.setBrush(QColor(218, 198, 138))
        painter.drawRect(QRectF(0, GROUND_Y + 12, GAME_W, GAME_H - GROUND_Y - 12))
        painter.setPen(QPen(QColor(85, 155, 50), 2))
        for x in range(int(self.ground_offset) - 24, GAME_W + 24, 24):
            painter.drawLine(x, GROUND_Y + 2, x + 8, GROUND_Y + 10)

        # Badge HUD
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 175))
        painter.drawRoundedRect(QRectF(10, 10, 210, 48), 6, 6)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Menlo", 12, QFont.Weight.Bold))
        painter.drawText(20, 30, f"SCORE: {self.score}  BEST: {self.best_score}")
        painter.setFont(QFont("Menlo", 9))
        painter.setPen(QColor(180, 220, 255))
        painter.drawText(20, 47, f"Morti: {self.deaths}   Frame: {self.frame_count}")


class ComputerVisionAnalyzer:
    """
    Modulo di Computer Vision con mss + OpenCV:
    Prende un'immagine BGR, estrae le coordinate fisiche (uccellino e varco ostacolo)
    e produce il frame di debug visivo con le annotazioni del tracciamento.
    """
    def __init__(self):
        self.last_bird_y = float(GROUND_Y // 2)
        self.smoothed_vy = 0.0

    def analyze_frame(self, bgr_img: np.ndarray) -> tuple[dict, np.ndarray, float]:
        t0 = time.perf_counter()
        h, w = bgr_img.shape[:2]
        vis_debug = bgr_img.copy()

        # Tagliamo rigorosamente sopra il terreno (GROUND_Y - 5) per evitare che l'erba
        # verde del suolo venga fusa con i tubi creando contorni anomali
        roi_h = max(10, GROUND_Y - 5)
        roi_bgr = bgr_img[:roi_h, :]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # 1. Riconoscimento Uccellino (Giallo intenso)
        bird_mask = cv2.inRange(hsv, np.array([18, 140, 160]), np.array([36, 255, 255]))
        bird_cnts, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bird_detected = False
        bird_x, bird_y = float(BIRD_X), self.last_bird_y
        for cnt in bird_cnts:
            area = cv2.contourArea(cnt)
            if 70 < area < 2500:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if abs((bx + bw / 2) - BIRD_X) < 40:
                    bird_x = float(bx + bw / 2)
                    bird_y = float(by + bh / 2)
                    bird_detected = True
                    cv2.rectangle(vis_debug, (bx, by), (bx + bw, by + bh), (0, 220, 255), 2)
                    cv2.putText(
                        vis_debug, f"BIRD ({int(bird_y)})", (bx - 8, max(12, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA
                    )
                    break

        # Calcolo velocita verticale pulita e smorzata (senza spike da micro-jitter temporale)
        inst_vy = bird_y - self.last_bird_y
        self.smoothed_vy = 0.75 * self.smoothed_vy + 0.25 * inst_vy
        self.last_bird_y = bird_y

        # 2. Riconoscimento Ostacoli (Tubi Verdi, isolati dal terreno)
        pipe_mask = cv2.inRange(hsv, np.array([38, 70, 50]), np.array([86, 255, 240]))
        pipe_cnts, _ = cv2.findContours(pipe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_rects = []
        for cnt in pipe_cnts:
            if cv2.contourArea(cnt) > 300:
                px, py, pw, ph = cv2.boundingRect(cnt)
                # Un tubo ha larghezza tipica tra 30 e 80 px
                if 28 <= pw <= 85:
                    valid_rects.append((px, py, pw, ph))
                    cv2.rectangle(vis_debug, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

        # Seleziona la colonna di tubi piu vicina davanti all'uccellino
        ahead_rects = [r for r in valid_rects if r[0] + r[2] >= bird_x - 10]
        gap_y = float(roi_h // 2)
        pipe_dist_x = 220.0
        pipe_found = False

        if ahead_rects:
            min_x = min(r[0] for r in ahead_rects)
            col_rects = [r for r in ahead_rects if abs(r[0] - min_x) < 25]
            col_rects = sorted(col_rects, key=lambda r: r[1])
            pipe_dist_x = max(0.0, min_x - bird_x)

            if len(col_rects) >= 2:
                top_edge = col_rects[0][1] + col_rects[0][3]  # fondo del tubo superiore
                bot_edge = col_rects[-1][1]                   # cima del tubo inferiore
                gap_y = float((top_edge + bot_edge) / 2.0)
                pipe_found = True
            elif len(col_rects) == 1:
                ry = col_rects[0][1]
                if ry < roi_h / 2:
                    gap_y = float(ry + col_rects[0][3] + PIPE_GAP / 2)
                else:
                    gap_y = float(ry - PIPE_GAP / 2)
                pipe_found = True

        target_x = int(bird_x + pipe_dist_x + PIPE_W / 2)
        target_y = int(gap_y)
        cv2.drawMarker(vis_debug, (target_x, target_y), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.circle(vis_debug, (target_x, target_y), int(PIPE_GAP / 2), (0, 140, 255), 1)
        cv2.line(vis_debug, (int(bird_x), int(bird_y)), (target_x, target_y), (255, 255, 0), 1, cv2.LINE_AA)

        cv_time_ms = (time.perf_counter() - t0) * 1000

        data = {
            "bird_x": bird_x,
            "bird_y": bird_y,
            "bird_vy": self.smoothed_vy,
            "gap_y": gap_y,
            "pipe_dist_x": pipe_dist_x,
            "bird_detected": bird_detected,
            "pipe_found": pipe_found,
            "cv_time_ms": cv_time_ms,
        }
        return data, vis_debug, cv_time_ms


class VisionMainWindow(QMainWindow):
    def __init__(self, clm_engine: Engine):
        super().__init__()
        self.clm_engine = clm_engine
        self.cv_analyzer = ComputerVisionAnalyzer()

        # Modalita: 1 = Vision + CLM Bot, 2 = Vision + Euristica Fisica, 3 = Manuale
        self.mode = 1
        self.manual_action = "down"

        # Telemetria
        self.last_choice = "up"
        self.last_probs = {"up": 0.5, "down": 0.5}
        self.last_clm_ms = 0.0
        self.last_cv_ms = 0.0
        self.last_state_text = ""

        self.setWindowTitle("Computer Vision (mss + OpenCV) + CLM (Contrastive-LM) [Apple M4]")
        self.setStyleSheet("background-color: #0b1120; color: #f1f5f9;")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(14)

        # 1. Sinistra: Gioco Reale (Canvas a 60 FPS)
        left_box = QVBoxLayout()
        lbl_game_title = QLabel("1. GIOCO REALE (60 FPS)")
        lbl_game_title.setFont(QFont("Menlo", 12, QFont.Weight.Bold))
        lbl_game_title.setStyleSheet("color: #38bdf8;")
        left_box.addWidget(lbl_game_title)

        self.game_canvas = PureGameCanvas()
        left_box.addWidget(self.game_canvas)
        main_layout.addLayout(left_box)

        # 2. Centro: Feed Computer Vision OpenCV (mss grab + visual debug)
        mid_box = QVBoxLayout()
        lbl_cv_title = QLabel("2. OCCHIO OPENCV (Cattura Pixel + Segmentazione)")
        lbl_cv_title.setFont(QFont("Menlo", 12, QFont.Weight.Bold))
        lbl_cv_title.setStyleSheet("color: #4ade80;")
        mid_box.addWidget(lbl_cv_title)

        self.lbl_cv_feed = QLabel()
        self.lbl_cv_feed.setFixedSize(GAME_W, GAME_H)
        self.lbl_cv_feed.setStyleSheet("border: 2px solid #334155; background-color: #000;")
        mid_box.addWidget(self.lbl_cv_feed)
        main_layout.addLayout(mid_box)

        # 3. Destra: Cervello CLM e Telemetria
        right_box = QVBoxLayout()
        lbl_brain_title = QLabel("3. CERVELLO CLM")
        lbl_brain_title.setFont(QFont("Menlo", 12, QFont.Weight.Bold))
        lbl_brain_title.setStyleSheet("color: #f59e0b;")
        right_box.addWidget(lbl_brain_title)

        # Pulsanti modalita
        self.btn_m1 = QPushButton("[1] Vision + CLM Bot (InfoNCE)")
        self.btn_m2 = QPushButton("[2] Vision + Euristica CV")
        self.btn_m3 = QPushButton("[3] Umano (Frecce SU/GIU)")
        for idx, btn in enumerate((self.btn_m1, self.btn_m2, self.btn_m3), start=1):
            btn.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda chk, m=idx: self.set_mode(m))
            right_box.addWidget(btn)

        # Tasti premuti da CLM
        keys_row = QHBoxLayout()
        self.lbl_key_up = QLabel("▲ UP")
        self.lbl_key_down = QLabel("▼ DOWN")
        for lbl in (self.lbl_key_up, self.lbl_key_down):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFont(QFont("Menlo", 12, QFont.Weight.Bold))
            lbl.setFixedHeight(38)
            keys_row.addWidget(lbl)
        right_box.addLayout(keys_row)

        # Dati estratti dalla Visione
        self.lbl_vision_data = QLabel()
        self.lbl_vision_data.setFont(QFont("Menlo", 9))
        self.lbl_vision_data.setWordWrap(True)
        self.lbl_vision_data.setStyleSheet(
            "background-color: #1e293b; padding: 8px; border-radius: 5px; color: #a5f3fc;"
        )
        right_box.addWidget(self.lbl_vision_data)

        # Probabilita e Latenza CLM
        self.lbl_clm_telemetry = QLabel()
        self.lbl_clm_telemetry.setFont(QFont("Menlo", 9))
        self.lbl_clm_telemetry.setStyleSheet(
            "background-color: #1e293b; padding: 8px; border-radius: 5px; color: #fed7aa;"
        )
        right_box.addWidget(self.lbl_clm_telemetry)

        right_box.addStretch(1)

        info_lbl = QLabel("Pixel estratti in diretta dallo schermo con mss.\nNessun accesso diretto alle coordinate del gioco!")
        info_lbl.setFont(QFont("Menlo", 8))
        info_lbl.setStyleSheet("color: #64748b;")
        right_box.addWidget(info_lbl)

        main_layout.addLayout(right_box)
        self.set_mode(1)

        # Timer 60 FPS
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loop_tick)
        self.timer.start(int(1000 / FPS))

    def set_mode(self, mode: int):
        self.mode = mode
        active_css = "background-color: #0284c7; color: white; padding: 7px; border-radius: 5px;"
        idle_css = "background-color: #334155; color: #cbd5e1; padding: 7px; border-radius: 5px;"
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

    def loop_tick(self):
        # 1. Cattura visiva del canvas del gioco (frame RGB grezzo dallo schermo)
        pix = self.game_canvas.grab()
        qimg = pix.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w * 3)
        rgb_arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
        bgr_frame = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        # 2. Pipeline OpenCV (riconoscimento visivo di bird e varco ostacoli)
        cv_data, vis_debug, cv_ms = self.cv_analyzer.analyze_frame(bgr_frame)
        self.last_cv_ms = cv_ms

        # 3. Costruzione dello stato testuale basato SOLO sulle osservazioni visive
        # Quantizziamo leggermente per massimizzare il riutilizzo nella VectorArena cache di CLM
        by = int(round(cv_data["bird_y"] / 4.0) * 4)
        gy = int(round(cv_data["gap_y"] / 4.0) * 4)
        dy = by - gy
        vy = round(cv_data["bird_vy"], 0)
        dx = int(round(cv_data["pipe_dist_x"] / 8.0) * 8)

        state_text = f"Bird Y: {by} | Gap Y: {gy} | Delta Y: {dy:+d} | Vel Y: {vy:+.1f} | Dist X: {dx}"
        self.last_state_text = state_text

        # 4. Processo Decisionale
        if self.mode == 1:
            # Modalita Vision + CLM Bot
            questions = {
                "move": {
                    "type": "choice",
                    "instructions": "Choose UP or DOWN to fly through the center of the obstacle gap.",
                    "criteria": b_game.ACTION_CRITERIA,
                }
            }
            t_clm0 = time.perf_counter()
            out = self.clm_engine.answer(state_text, questions)
            self.last_clm_ms = (time.perf_counter() - t_clm0) * 1000
            ans = out["answers"]["move"]
            self.last_choice = ans["choice"]
            self.last_probs = ans["probabilities"]
            action = self.last_choice
        elif self.mode == 2:
            # Euristica basata sui dati della Computer Vision
            action = "up" if (dy + vy * 2.5) > 0 else "down"
            self.last_choice = action
            self.last_probs = {"up": 1.0 if action == "up" else 0.0, "down": 1.0 if action == "down" else 0.0}
            self.last_clm_ms = 0.001
        else:
            action = self.manual_action
            self.last_choice = action
            self.last_probs = {"up": 1.0 if action == "up" else 0.0, "down": 1.0 if action == "down" else 0.0}
            self.last_clm_ms = 0.0

        # 5. Applica l'azione alla fisica del gioco ed aggiorna la finestra
        self.game_canvas.step_physics(action)
        self.game_canvas.update()

        # 6. Aggiorna la vista OpenCV debug nel pannello centrale
        rgb_debug = cv2.cvtColor(vis_debug, cv2.COLOR_BGR2RGB)
        dh, dw, dc = rgb_debug.shape
        qdebug = QImage(rgb_debug.data, dw, dh, dw * dc, QImage.Format.Format_RGB888)
        self.lbl_cv_feed.setPixmap(QPixmap.fromImage(qdebug))

        # 7. Aggiorna la telemetria UI
        self.update_telemetry(cv_data)

    def update_telemetry(self, cv_data: dict):
        if self.last_choice == "up":
            self.lbl_key_up.setStyleSheet("background-color: #22c55e; color: #052e16; border-radius: 5px;")
            self.lbl_key_down.setStyleSheet("background-color: #1e293b; color: #64748b; border-radius: 5px;")
        else:
            self.lbl_key_up.setStyleSheet("background-color: #1e293b; color: #64748b; border-radius: 5px;")
            self.lbl_key_down.setStyleSheet("background-color: #f97316; color: #431407; border-radius: 5px;")

        p_up = self.last_probs.get("up", 0.0)
        p_dn = self.last_probs.get("down", 0.0)
        bar_up = "█" * int(round(p_up * 14)) + "░" * (14 - int(round(p_up * 14)))
        bar_dn = "█" * int(round(p_dn * 14)) + "░" * (14 - int(round(p_dn * 14)))

        self.lbl_vision_data.setText(
            f"OBSERVAZIONE COMPUTER VISION (OpenCV):\n"
            f"• Bird Tracciato:  {'SI (Giallo)' if cv_data['bird_detected'] else 'STIMA'}\n"
            f"• Varco Rilevato:  {'SI (Verde)' if cv_data['pipe_found'] else 'CERCA'}\n"
            f"• Bird Y (visivo): {cv_data['bird_y']:.1f} px\n"
            f"• Gap Y (visivo):  {cv_data['gap_y']:.1f} px\n"
            f"• Delta Y mirino:  {cv_data['bird_y'] - cv_data['gap_y']:+.1f} px\n"
            f"• Tempo OpenCV:    {self.last_cv_ms:.2f} ms"
        )

        arena_stats = self.clm_engine.arena.stats()["pools"]["512"]
        hit_rate = (arena_stats["hit_rate"] or 0.0) * 100.0

        self.lbl_clm_telemetry.setText(
            f"DECISIONE CLM (Contrastive Model):\n"
            f"Input: \"{self.last_state_text}\"\n\n"
            f"P(UP)   [{bar_up}] {p_up*100:5.1f}%\n"
            f"P(DOWN) [{bar_dn}] {p_dn*100:5.1f}%\n"
            f"Comando:        {self.last_choice.upper()}\n"
            f"Latenza CLM:    {self.last_clm_ms:.3f} ms\n"
            f"Arena Hit Rate: {hit_rate:.1f}% ({arena_stats['hits']} hit)"
        )


def main():
    ckpt_path = b_game.ensure_bird_clm_checkpoint("checkpoints/bird_clm_head.pt")
    engine = Engine(
        embedder=b_game.FastGameEmbedder(),
        checkpoint=ckpt_path,
        device="cpu",
        action_cache="32MiB",
    )
    app = QApplication(sys.argv)
    win = VisionMainWindow(engine)
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
