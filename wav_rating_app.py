import csv
import math
import random
import re
import socket
import time
from pathlib import Path

import pygame

# --- config ---
WAV_DIRS = [
    ("Maj", Path(r"extraits\Maj"), 1),
    ("Min", Path(r"extraits\Min"), 2),
    ("Post-tonal", Path(r"extraits\Post-tonal"), 3),
]
COUNTDOWN_SECONDS = 3
TUTORIAL_SECONDS = 20
SILENCE_SECONDS = 5
OUTPUT_CSV = Path("ratings.csv")
WINDOW_SIZE = (1200, 800)
GRID_MARGIN = 90
FPS = 60
SEND_MARKERS = True
MARKER_HOST = "127.0.0.1"
MARKER_PORT = 15361
MARKER_LOG_PATH = Path("experiment_markers.txt")

# Colors (RGB)
GRID = (60, 64, 80)
AXIS = (210, 210, 210)
TEXT = (235, 235, 235)
TEXT_DIM = (170, 170, 180)
HILITE = (255, 180, 70)

BG_READY_TOP = (24, 26, 36)
BG_READY_BOTTOM = (14, 16, 24)
BG_COUNT_TOP = (26, 20, 38)
BG_COUNT_BOTTOM = (12, 10, 18)
BG_PLAY_TOP = (16, 28, 36)
BG_PLAY_BOTTOM = (10, 16, 24)
BG_RATE_TOP = (18, 24, 30)
BG_RATE_BOTTOM = (12, 14, 20)
BG_DONE_TOP = (22, 26, 20)
BG_DONE_BOTTOM = (12, 16, 10)

ACCENT_READY = (120, 170, 255)
ACCENT_COUNT = (255, 120, 180)
ACCENT_PLAY = (120, 220, 200)
ACCENT_RATE = (255, 200, 120)
ACCENT_DONE = (160, 240, 140)

TUTORIAL_LINE_DELAY = 1.5
TUTORIAL_FADE_SECONDS = 0.5

enter_pressed = False

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def pos_to_rating(pos, grid_rect):
    cx, cy = grid_rect.center
    x = (pos[0] - cx) / (grid_rect.width / 2) * 3.0
    y = (cy - pos[1]) / (grid_rect.height / 2) * 3.0
    x = clamp(x, -3.0, 3.0)
    y = clamp(y, -3.0, 3.0)
    return x, y  # arousal, valence


def rating_to_pos(arousal, valence, grid_rect):
    cx, cy = grid_rect.center
    x = cx + (arousal / 3.0) * (grid_rect.width / 2)
    y = cy - (valence / 3.0) * (grid_rect.height / 2)
    return int(x), int(y)


def append_rating(csv_path, wav_path, valence, arousal, selected, elapsed_s):
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["wav_file", "valence(energique)", "arousal(sentiment positif)", "selected", "elapsed_s", "rated_at"]
             )
        writer.writerow(
            [
                wav_path.name,
                "" if valence is None else f"{valence:.2f}",
                "" if arousal is None else f"{arousal:.2f}",
                "yes" if selected else "no",
                f"{elapsed_s:.3f}",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ]
         )


def append_marker_log(log_path, wav_path, start_code, end_code):
    new_file = not log_path.exists()
    with log_path.open("a", encoding="ascii") as f:
        if new_file:
            f.write("song_name,start_id,end_id\n")
        start_text = "" if start_code is None else str(start_code)
        end_text = "" if end_code is None else str(end_code)
        f.write(f"{wav_path.name},{start_text},{end_text}\n")


def make_gradient_surface(size, top_color, bottom_color):
    w, h = size
    surf = pygame.Surface((w, h))
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
        pygame.draw.line(surf, (r, g, b), (0, y), (w, y))
    return surf


def windows_natural_key(path):
    parts = re.split(r"(\d+)", path.name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((1, int(part)))
        else:
            key.append((0, part.lower()))
    return key


class MarkerClient:
    def __init__(self, host, port, enabled=True):
        self.host = host
        self.port = port
        self.enabled = enabled
        self.sock = None

    def connect(self):
        if not self.enabled:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.host, self.port))
            print(f"[markers] connected {self.host}:{self.port}")
        except OSError as exc:
            print(f"[markers] disabled ({exc})")
            self.enabled = False
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def send(self, tag):
        if not self.enabled or not self.sock:
            return
        try:
            payload = f"{tag}\n".encode("ascii", errors="ignore")
            self.sock.sendall(payload)
            print(f"[markers] {tag}")
        except OSError:
            self.enabled = False

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


def blit_centered_lines(surface, line_surfaces, center, gap=8):
    if not line_surfaces:
        return
    total_h = sum(s.get_height() for s in line_surfaces) + gap * (len(line_surfaces) - 1)
    y = center[1] - total_h // 2
    for s in line_surfaces:
        x = center[0] - s.get_width() // 2
        surface.blit(s, (x, y))
        y += s.get_height() + gap


def main():
    enter_pressed = False
    missing_dirs = [d for _, d, _ in WAV_DIRS if not d.exists()]
    if missing_dirs:
        missing_text = ", ".join(str(d) for d in missing_dirs)
        raise SystemExit(f"WAV_DIRS not found: {missing_text}")

    wav_entries = []
    empty_dirs = []
    for dir_name, wav_dir, dir_code in WAV_DIRS:
        files = sorted(wav_dir.glob("*.wav"), key=windows_natural_key)
        if not files:
            empty_dirs.append(wav_dir)
            continue
        if len(files) > 99:
            raise SystemExit(
                f"Too many .wav files in {dir_name} ({len(files)}). "
                "Ranking must fit XX (1..99)."
            )
        for rank, wav_path in enumerate(files, start=1):
            start_code = 34000 + rank * 10 + dir_code
            end_code = 35000 + rank * 10 + dir_code
            wav_entries.append(
                {
                    "path": wav_path,
                    "dir_code": dir_code,
                    "rank": rank,
                    "start_code": start_code,
                    "end_code": end_code,
                }
            )

    if empty_dirs:
        empty_text = ", ".join(str(d) for d in empty_dirs)
        raise SystemExit(f"No .wav files found in: {empty_text}")

    if not wav_entries:
        raise SystemExit("No .wav files found across WAV_DIRS")

    random.shuffle(wav_entries)

    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Valence-Arousal Rating")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 24)
    font_big = pygame.font.SysFont(None, 32)
    font_title = pygame.font.SysFont(None, 48)
    font_count = pygame.font.SysFont(None, 160)
    font_tutorial_title = pygame.font.SysFont(None, 56)
    font_tutorial = pygame.font.SysFont(None, 36)

    grid_rect = pygame.Rect(
        GRID_MARGIN,
        GRID_MARGIN,
        WINDOW_SIZE[0] - GRID_MARGIN * 2,
        WINDOW_SIZE[1] - GRID_MARGIN * 2,
    )

    STATE_READY = "ready"
    STATE_TUTORIAL = "tutorial"
    STATE_COUNTDOWN = "countdown"
    STATE_PLAYING = "playing"
    STATE_RATING = "rating"
    STATE_SILENCE = "silence"
    STATE_DONE = "done"

    current_idx = 0
    state = STATE_TUTORIAL
    tutorial_start = time.monotonic()
    countdown_start = None
    rating_start = None
    silence_start = None
    selection = None  # (arousal, valence)

    def start_countdown():
        nonlocal state, countdown_start
        countdown_start = time.monotonic()
        state = STATE_COUNTDOWN

    def start_playback():
        nonlocal state
        entry = wav_entries[current_idx]
        pygame.mixer.music.load(str(entry["path"]))
        pygame.mixer.music.play()
        marker_client.send(str(entry["start_code"]))
        state = STATE_PLAYING

    def start_rating():
        nonlocal state, rating_start, selection
        selection = None
        rating_start = time.monotonic()
        state = STATE_RATING

    def finish_rating():
        nonlocal current_idx, state, silence_start
        entry = wav_entries[current_idx]
        elapsed = time.monotonic() - rating_start
        selected = selection is not None
        arousal = selection[0] if selection else None
        valence = selection[1] if selection else None
        append_rating(OUTPUT_CSV, entry["path"], valence, arousal, selected, elapsed)
        append_marker_log(
            MARKER_LOG_PATH,
            entry["path"],
            entry["start_code"],
            entry["end_code"],
        )
        current_idx += 1
        if current_idx >= len(wav_entries):
            state = STATE_DONE
        else:
            silence_start = time.monotonic()
            state = STATE_SILENCE

    bg_ready = make_gradient_surface(WINDOW_SIZE, BG_READY_TOP, BG_READY_BOTTOM)
    bg_count = make_gradient_surface(WINDOW_SIZE, BG_COUNT_TOP, BG_COUNT_BOTTOM)
    bg_play = make_gradient_surface(WINDOW_SIZE, BG_PLAY_TOP, BG_PLAY_BOTTOM)
    bg_rate = make_gradient_surface(WINDOW_SIZE, BG_RATE_TOP, BG_RATE_BOTTOM)
    bg_done = make_gradient_surface(WINDOW_SIZE, BG_DONE_TOP, BG_DONE_BOTTOM)

    marker_client = MarkerClient(MARKER_HOST, MARKER_PORT, enabled=SEND_MARKERS)
    marker_client.connect()

    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN:
                if state == STATE_READY and event.key == pygame.K_SPACE:
                    start_countdown()
                elif state == STATE_TUTORIAL and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    enter_pressed = True
                elif state == STATE_RATING and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if selection is not None:
                        finish_rating()
                elif state == STATE_DONE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if state == STATE_RATING and grid_rect.collidepoint(event.pos):
                    arousal, valence = pos_to_rating(event.pos, grid_rect)
                    selection = (arousal, valence)
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                if state == STATE_RATING and grid_rect.collidepoint(event.pos):
                    arousal, valence = pos_to_rating(event.pos, grid_rect)
                    selection = (arousal, valence)

        if state == STATE_COUNTDOWN:
            elapsed = time.monotonic() - countdown_start
            if elapsed >= COUNTDOWN_SECONDS:
                start_playback()
        elif state == STATE_TUTORIAL:
            elapsed = time.monotonic() - tutorial_start
            if elapsed >= TUTORIAL_SECONDS or enter_pressed == True :
                state = STATE_READY
                enter_pressed = False
        elif state == STATE_PLAYING:
            if not pygame.mixer.music.get_busy():
                entry = wav_entries[current_idx]
                marker_client.send(str(entry["end_code"]))
                start_rating()
        elif state == STATE_RATING:
            pass
        elif state == STATE_SILENCE:
            elapsed = time.monotonic() - silence_start
            if elapsed >= SILENCE_SECONDS:
                start_playback()

        # --- draw ---
        if state == STATE_READY:
            screen.blit(bg_ready, (0, 0))
        elif state == STATE_TUTORIAL:
            screen.blit(bg_ready, (0, 0))
        elif state == STATE_COUNTDOWN:
            screen.blit(bg_count, (0, 0))
        elif state == STATE_PLAYING:
            screen.blit(bg_play, (0, 0))
        elif state == STATE_RATING:
            screen.blit(bg_rate, (0, 0))
        elif state == STATE_SILENCE:
            screen.blit(bg_ready, (0, 0))
        elif state == STATE_DONE:
            screen.blit(bg_done, (0, 0))

        cx, cy = grid_rect.center
        if state in (STATE_TUTORIAL, STATE_RATING):
            # grid
            pygame.draw.rect(screen, GRID, grid_rect, 1)
            pygame.draw.line(screen, AXIS, (grid_rect.left, cy), (grid_rect.right, cy), 2)
            pygame.draw.line(screen, AXIS, (cx, grid_rect.top), (cx, grid_rect.bottom), 2)

            # ticks
            for i in range(-3, 4):
                x = int(cx + (i / 3.0) * (grid_rect.width / 2))
                y = int(cy - (i / 3.0) * (grid_rect.height / 2))
                pygame.draw.line(screen, GRID, (x, grid_rect.top), (x, grid_rect.bottom), 1)
                pygame.draw.line(screen, GRID, (grid_rect.left, y), (grid_rect.right, y), 1)

            # axis labels
            v_pos = font.render("Valence +3 (sentiment positif)", True, TEXT)
            v_neg = font.render("Valence -3 (sentiment negatif)", True, TEXT)
            a_pos_main = font.render("Arousal +3", True, TEXT)
            a_pos_sub = font.render("(energique)", True, TEXT_DIM)
            a_neg_main = font.render("Arousal -3", True, TEXT)
            a_neg_sub = font.render("(calme)", True, TEXT_DIM)
            screen.blit(
                v_pos,
                (cx - v_pos.get_width() // 2, grid_rect.top - 20),
            )
            screen.blit(
                v_neg,
                (cx - v_neg.get_width() // 2, grid_rect.bottom - v_neg.get_height() + 20),
            )
            line_gap = 2
            neg_lines = [a_neg_main, a_neg_sub]
            pos_lines = [a_pos_main, a_pos_sub]
            neg_total_h = sum(s.get_height() for s in neg_lines) + line_gap * (len(neg_lines) - 1)
            pos_total_h = sum(s.get_height() for s in pos_lines) + line_gap * (len(pos_lines) - 1)
            neg_y = cy - neg_total_h // 2
            pos_y = cy - pos_total_h // 2
            neg_x = grid_rect.left - 90
            pos_right = grid_rect.right + 90
            for s in neg_lines:
                screen.blit(s, (neg_x, neg_y))
                neg_y += s.get_height() + line_gap
            for s in pos_lines:
                screen.blit(s, (pos_right - s.get_width(), pos_y))
                pos_y += s.get_height() + line_gap

            # selection marker / tutorial example
            if state == STATE_RATING and selection is not None:
                pos = rating_to_pos(selection[0], selection[1], grid_rect)
                pygame.draw.circle(screen, HILITE, pos, 8, 0)
                pygame.draw.circle(screen, (0, 0, 0), pos, 8, 2)
            elif state == STATE_TUTORIAL:
                demo_pos = rating_to_pos(1.5, 1.0, grid_rect)
                pygame.draw.circle(screen, HILITE, demo_pos, 7, 0)
                pygame.draw.circle(screen, (0, 0, 0), demo_pos, 7, 2)
                demo_label = font.render("Example", True, TEXT_DIM)
                screen.blit(demo_label, (demo_pos[0] + 10, demo_pos[1] - 10))

        # UI text
        center = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
        if state == STATE_READY:
            lines = [
                font_title.render("Ready?", True, ACCENT_READY),
                font_big.render("Press SPACE to start", True, TEXT),
            ]
            # if current_idx < len(wav_entries):
            #     lines.append(
            #         font.render(f"Next: {wav_entries[current_idx]['path'].name}", True, TEXT_DIM)
            #     )
            blit_centered_lines(screen, lines, center)
        elif state == STATE_TUTORIAL:
            elapsed = time.monotonic() - tutorial_start
            title = font_tutorial_title.render("How to Rate", True, ACCENT_READY)
            screen.blit(title, (40, 20))

            phrases = [
                "1) Listen to each sound clip",
                "2) Click or drag on the grid",
                "Left/Right = Valence",
                "Down/Up = Arousal",
                "Press ENTER when you are done",
            ]
            y = 20 + title.get_height() + 10
            for i, text in enumerate(phrases):
                start_t = (i + 1) * TUTORIAL_LINE_DELAY
                if elapsed < start_t:
                    continue
                alpha = min(1.0, (elapsed - start_t) / TUTORIAL_FADE_SECONDS)
                surf = font_tutorial.render(text, True, TEXT)
                surf.set_alpha(int(255 * alpha))
                screen.blit(surf, (40, y))
                y += surf.get_height() + 6
        elif state == STATE_COUNTDOWN:
            count = max(1, int(math.ceil(COUNTDOWN_SECONDS - (time.monotonic() - countdown_start))))
            label = font_count.render(str(count), True, ACCENT_COUNT)
            screen.blit(
                label,
                (cx - label.get_width() // 2, cy - label.get_height() // 2),
            )
            tip = font_big.render("Get ready", True, TEXT_DIM)
            screen.blit(
                tip,
                (cx - tip.get_width() // 2, cy + label.get_height() // 2 - 10),
            )
        elif state == STATE_PLAYING:
            lines = [
                font_title.render("Playing", True, ACCENT_PLAY),
                # font_big.render(wav_entries[current_idx]["path"].name, True, TEXT),
                font.render("Listen carefully...", True, TEXT_DIM),
            ]
            blit_centered_lines(screen, lines, center)
        elif state == STATE_RATING:
            title = font_title.render("Rate now", True, ACCENT_RATE)
            name = font_big.render(wav_entries[current_idx]["path"].name, True, TEXT)
            y = 20
            screen.blit(title, (20, y))
            y += title.get_height() + 8
            screen.blit(name, (20, y))
            y += name.get_height() + 6
            if selection is None:
                prompt = font.render(
                    "Click to rate, then press ENTER to continue", True, TEXT
                )
                screen.blit(prompt, (400, 20))
            else:
                arousal, valence = selection
                vals = f"Valence: {valence:+.2f} | Arousal: {arousal:+.2f}"
                screen.blit(font.render(vals, True, TEXT), (20, y))
                done = font.render("Press ENTER to continue", True, TEXT_DIM)
                screen.blit(done, (400, 20))
        elif state == STATE_DONE:
            lines = [
                font_title.render("CONGRATS U HAVE FINISHED THE TEST", True, ACCENT_DONE),
                font_big.render("Press any key to exit", True, TEXT_DIM),
            ]
            blit_centered_lines(screen, lines, center)
        elif state == STATE_SILENCE:
            remaining = max(0.0, SILENCE_SECONDS - (time.monotonic() - silence_start))
            lines = [
                font_title.render("Silence", True, ACCENT_READY),
                font_big.render(f"Next clip in {remaining:0.1f}s", True, TEXT_DIM),
            ]
            blit_centered_lines(screen, lines, center)

        pygame.display.flip()

    pygame.mixer.music.stop()
    marker_client.close()
    pygame.quit()


if __name__ == "__main__":
    main()
