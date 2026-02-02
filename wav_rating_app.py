import csv
import math
import random
import socket
import time
from pathlib import Path

import pygame

# --- config ---
WAV_DIRS = [
    Path(r"extraits\Min"),
    Path(r"extraits\Maj"),
    Path(r"extraits\Post-tonal"),
]
TRIAL_SECONDS = 10
COUNTDOWN_SECONDS = 3
TUTORIAL_SECONDS = 5
SILENCE_SECONDS = 5
OUTPUT_CSV = Path("ratings.csv")
WINDOW_SIZE = (1200, 800)
GRID_MARGIN = 90
FPS = 60
SEND_MARKERS = True
MARKER_HOST = "127.0.0.1"
MARKER_PORT = 15361
MARKER_LOG_PATH = Path("experiment_markers.txt")
FILE_MAP_PATH = Path(r"C:\Users\rayen\eeg\file_marker_map.csv")

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


def read_file_marker_codes(map_path, marker_name):
    if not map_path.exists():
        return None, None
    try:
        with map_path.open("r", encoding="ascii") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("file_name"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                name, start_code, end_code = parts[0], parts[1], parts[2]
                if name.upper() == marker_name.upper():
                    return start_code, end_code
    except OSError:
        return None, None
    return None, None


def append_marker_log(log_path, wav_path, marker_name, map_path):
    new_file = not log_path.exists()
    with log_path.open("a", encoding="ascii") as f:
        if new_file:
            f.write("song_name,start_id,end_id\n")
        start_code, end_code = read_file_marker_codes(map_path, marker_name)
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


def normalize_marker_name(name):
    safe = []
    last_underscore = False
    for ch in name:
        if ch.isalnum():
            safe.append(ch.upper())
            last_underscore = False
        else:
            if not last_underscore:
                safe.append("_")
                last_underscore = True
    text = "".join(safe).strip("_")
    return text or "UNTITLED"


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
    missing_dirs = [d for d in WAV_DIRS if not d.exists()]
    if missing_dirs:
        missing_text = ", ".join(str(d) for d in missing_dirs)
        raise SystemExit(f"WAV_DIRS not found: {missing_text}")

    wav_files = []
    empty_dirs = []
    for wav_dir in WAV_DIRS:
        files = sorted(wav_dir.glob("*.wav"))
        if not files:
            empty_dirs.append(wav_dir)
        wav_files.extend(files)

    if empty_dirs:
        empty_text = ", ".join(str(d) for d in empty_dirs)
        raise SystemExit(f"No .wav files found in: {empty_text}")

    if not wav_files:
        raise SystemExit("No .wav files found across WAV_DIRS")

    random.shuffle(wav_files)

    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Valence-Arousal Rating")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 24)
    font_big = pygame.font.SysFont(None, 32)
    font_title = pygame.font.SysFont(None, 48)
    font_count = pygame.font.SysFont(None, 160)

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
    current_marker_name = None

    def start_countdown():
        nonlocal state, countdown_start
        countdown_start = time.monotonic()
        state = STATE_COUNTDOWN

    def start_playback():
        nonlocal state, current_marker_name
        marker_name = normalize_marker_name(wav_files[current_idx].stem)
        current_marker_name = marker_name
        pygame.mixer.music.load(str(wav_files[current_idx]))
        pygame.mixer.music.play()
        marker_client.send(f"START_{marker_name}")
        state = STATE_PLAYING

    def start_rating():
        nonlocal state, rating_start, selection
        selection = None
        rating_start = time.monotonic()
        state = STATE_RATING

    def finish_rating():
        nonlocal current_idx, state, current_marker_name, silence_start
        elapsed = time.monotonic() - rating_start
        selected = selection is not None
        arousal = selection[0] if selection else None
        valence = selection[1] if selection else None
        append_rating(OUTPUT_CSV, wav_files[current_idx], valence, arousal, selected, elapsed)
        marker_name = current_marker_name or normalize_marker_name(wav_files[current_idx].stem)
        append_marker_log(
            MARKER_LOG_PATH,
            wav_files[current_idx],
            marker_name,
            FILE_MAP_PATH,
        )
        current_idx += 1
        current_marker_name = None
        if current_idx >= len(wav_files):
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
            if elapsed >= TUTORIAL_SECONDS:
                state = STATE_READY
        elif state == STATE_PLAYING:
            if not pygame.mixer.music.get_busy():
                marker_name = current_marker_name or normalize_marker_name(wav_files[current_idx].stem)
                marker_client.send(f"END_{marker_name}")
                start_rating()
        elif state == STATE_RATING:
            elapsed = time.monotonic() - rating_start
            if elapsed >= TRIAL_SECONDS:
                finish_rating()
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
        if state == STATE_RATING:
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

            # selection marker
            if selection is not None:
                pos = rating_to_pos(selection[0], selection[1], grid_rect)
                pygame.draw.circle(screen, HILITE, pos, 8, 0)
                pygame.draw.circle(screen, (0, 0, 0), pos, 8, 2)

        # UI text
        center = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
        if state == STATE_READY:
            lines = [
                font_title.render("Ready?", True, ACCENT_READY),
                font_big.render("Press SPACE to start", True, TEXT),
            ]
            # if current_idx < len(wav_files):
            #     lines.append(font.render(f"Next: {wav_files[current_idx].name}", True, TEXT_DIM))
            blit_centered_lines(screen, lines, center)
        elif state == STATE_TUTORIAL:
            lines = [
                font_title.render("How to Rate", True, ACCENT_READY),
                font_big.render("Listen to each sound clip", True, TEXT),
                font.render("Then rate valence (left/right) and arousal (down/up).", True, TEXT_DIM),
                font.render("Click or drag on the grid to choose a point.", True, TEXT_DIM),
                font.render(f"You have {TRIAL_SECONDS} seconds to rate each clip.", True, TEXT_DIM),
            ]
            blit_centered_lines(screen, lines, center)
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
                # font_big.render(wav_files[current_idx].name, True, TEXT),
                font.render("Listen carefully...", True, TEXT_DIM),
            ]
            blit_centered_lines(screen, lines, center)
        elif state == STATE_RATING:
            remaining = max(0.0, TRIAL_SECONDS - (time.monotonic() - rating_start))
            title = font_title.render("Rate now", True, ACCENT_RATE)
            name = font_big.render(wav_files[current_idx].name, True, TEXT)
            timer = font.render(f"Remaining: {remaining:0.1f}s", True, TEXT_DIM)
            y = 20
            screen.blit(title, (20, y))
            y += title.get_height() + 8
            screen.blit(name, (20, y))
            y += name.get_height() + 6
            screen.blit(timer, (400, 20))
            # y += timer.get_height() + 6
            if selection is None:
                prompt = font.render(
                    "Click to rate (you can adjust until time ends)", True, TEXT
                )
                screen.blit(prompt, (400, 40))
            else:
                arousal, valence = selection
                vals = f"Valence: {valence:+.2f} | Arousal: {arousal:+.2f}"
                screen.blit(font.render(vals, True, TEXT), (20, y))
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
