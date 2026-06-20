from __future__ import annotations

import math
import queue
import random
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import librosa
import numpy as np
import sounddevice as sd


@dataclass(slots=True)
class Branch:
    dest: int
    distance: float


@dataclass(slots=True)
class Beat:
    index: int
    start_sample: int
    end_sample: int
    feature: np.ndarray
    loudness_db: float
    bar_position: int
    branches: deque[Branch] = field(default_factory=deque)

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(slots=True)
class Analysis:
    samples: np.ndarray
    sample_rate: int
    beats: list[Beat]
    tempo: float


@dataclass(slots=True)
class Transition:
    next_index: int
    branch: Branch | None
    branch_source: int | None = None
    jump_from: int | None = None
    jump_chance: float = 0.0
    branch_options: int = 0
    end_playback: bool = False


@dataclass(slots=True)
class PlaybackEvent:
    beat_index: int
    beats_played: int
    played_seconds: float
    position_seconds: float
    jumps_taken: int
    jump_from: int | None = None
    jump_to: int | None = None
    jump_distance: float | None = None
    beat_loudness_db: float = 0.0
    beat_duration_seconds: float = 0.0
    bar_position: int = 0
    branch_options: int = 0
    jump_chance: float = 0.0
    next_branch_index: int | None = None
    next_branch_jump_chance: float = 0.0
    next_branch_options: int = 0
    planned_next: int | None = None
    planned_jump_from: int | None = None
    planned_jump_to: int | None = None
    planned_jump_distance: float | None = None
    ended: bool = False


MAX_LOUDNESS_DIFF_DB = 5.0
LOUDNESS_DISTANCE_WEIGHT = 0.35
DEFAULT_MAX_BRANCHES_PER_MINUTE = 10.0
NEARBY_BRANCH_WINDOW_BEATS = 8
BRANCH_OFFSET_TOLERANCE_BEATS = 1
UNUSED_BRANCH_CHANCE_MULTIPLIER = 1.35
BRANCH_USAGE_NORMALIZE_THRESHOLD = 5
END_BRANCH_BOOST_WINDOW_BEATS = 32
END_BRANCH_FORCE_WINDOW_BEATS = 8
END_BRANCH_MAX_CHANCE = 1.0
SEEK_STEP_SECONDS = 10.0
EXPORT_CHUNK_SECONDS = 5.0
EXPORT_BITRATE = "192k"


class EternalPlayer:
    def __init__(
        self,
        analysis: Analysis,
        min_branch_chance: float,
        max_branch_chance: float,
        branch_chance_delta: float,
        volume: float,
        on_event: Callable[[PlaybackEvent], None] | None = None,
        timed_end_seconds: float | None = None,
    ) -> None:
        self.analysis = analysis
        self.min_branch_chance = min_branch_chance
        self.max_branch_chance = max_branch_chance
        self.branch_chance_delta = branch_chance_delta
        self.volume = volume
        self.on_event = on_event
        self.timed_end_seconds = timed_end_seconds if timed_end_seconds is None or timed_end_seconds > 0 else None

        self._beat_index = 0
        self._sample_in_beat = 0
        self._played_samples = 0
        self._branch_chance = min_branch_chance
        self._branch_use_counts: dict[tuple[int, int], int] = {}
        self._branch_routes = [
            (beat.index, branch.dest)
            for beat in analysis.beats
            for branch in beat.branches
        ]
        self._beat_start_samples = np.array([beat.start_sample for beat in analysis.beats], dtype=np.int64)
        branch_source_indexes = [beat.index for beat in analysis.beats if beat.branches]
        self._last_branch_source_index = branch_source_indexes[-1] if branch_source_indexes else None
        self._beats_played = 0
        self._jumps_taken = 0
        self._stream: sd.OutputStream | None = None
        self._finished = False
        self._lock = threading.Lock()
        self._transition = self._plan_transition()

    def start(self) -> None:
        if self._stream is not None:
            return
        channels = self.analysis.samples.shape[1]
        self._stream = sd.OutputStream(
            samplerate=self.analysis.sample_rate,
            channels=channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._emit_event()

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def seek_seconds(self, delta_seconds: float) -> float:
        with self._lock:
            current_sample = self._current_source_sample()
            delta_samples = int(delta_seconds * self.analysis.sample_rate)
            target_sample = self._clamp_source_sample(current_sample + delta_samples)
            self._seek_to_sample(target_sample)
            position_seconds = target_sample / self.analysis.sample_rate

        self._emit_event()
        return position_seconds

    def render_frames(self, frames: int) -> tuple[np.ndarray, bool]:
        channels = self.analysis.samples.shape[1]
        outdata = np.zeros((frames, channels), dtype=np.float32)
        write_pos = 0
        ended = False

        while write_pos < frames:
            with self._lock:
                if self._finished:
                    ended = True
                    break

                beat_index = self._beat_index
                beat = self.analysis.beats[beat_index]
                take = min(frames - write_pos, beat.duration_samples - self._sample_in_beat)
                if take > 0:
                    start = beat.start_sample + self._sample_in_beat
                    self._sample_in_beat += take
                    reached_beat_end = self._sample_in_beat >= beat.duration_samples
                else:
                    start = beat.start_sample
                    reached_beat_end = True

            if take <= 0:
                ended = self._finish_beat(expected_beat_index=beat_index)
                if ended:
                    break
                continue

            outdata[write_pos : write_pos + take] = self.analysis.samples[start : start + take] * self.volume
            write_pos += take

            if reached_beat_end:
                ended = self._finish_beat(expected_beat_index=beat_index)
                if ended:
                    break

        return outdata[:write_pos], ended

    def _callback(self, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            print(status, file=sys.stderr)
        with self._lock:
            if self._finished:
                outdata.fill(0)
                raise sd.CallbackStop

        write_pos = 0
        while write_pos < frames:
            with self._lock:
                beat_index = self._beat_index
                beat = self.analysis.beats[beat_index]
                take = min(frames - write_pos, beat.duration_samples - self._sample_in_beat)
                if take > 0:
                    start = beat.start_sample + self._sample_in_beat
                    self._sample_in_beat += take
                    reached_beat_end = self._sample_in_beat >= beat.duration_samples
                else:
                    start = beat.start_sample
                    reached_beat_end = True

            if take <= 0:
                if self._finish_beat(expected_beat_index=beat_index):
                    outdata[write_pos:] = 0
                    raise sd.CallbackStop
                continue

            outdata[write_pos : write_pos + take] = self.analysis.samples[start : start + take] * self.volume

            write_pos += take

            if reached_beat_end:
                if self._finish_beat(expected_beat_index=beat_index):
                    outdata[write_pos:] = 0
                    raise sd.CallbackStop

    def _finish_beat(self, expected_beat_index: int | None = None) -> bool:
        with self._lock:
            current = self._beat_index
            if expected_beat_index is not None and current != expected_beat_index:
                return False

            transition = self._transition
            jump = transition.branch
            jump_from = transition.jump_from
            current_duration = self.analysis.beats[current].duration_samples

            if transition.end_playback:
                self._sample_in_beat = 0
                self._beats_played += 1
                self._played_samples += current_duration
                self._finished = True
                ended = True
            else:
                self._played_samples += current_duration
                if jump is not None and not self._branches_enabled():
                    jump = None
                    jump_from = None
                    next_index = current + 1
                else:
                    next_index = transition.next_index

                self._beat_index = next_index
                self._sample_in_beat = 0
                self._beats_played += 1
                if jump is not None:
                    self._jumps_taken += 1
                    if jump_from is not None:
                        route = (jump_from, jump.dest)
                        self._branch_use_counts[route] = self._branch_use_counts.get(route, 0) + 1
                        self._normalize_branch_usage_counts()
                self._transition = self._plan_transition()
                ended = False

        self._emit_event(jump_from, jump, ended=ended)
        return ended

    def _current_source_sample(self) -> int:
        beat = self.analysis.beats[self._beat_index]
        return self._clamp_source_sample(beat.start_sample + self._sample_in_beat)

    def _clamp_source_sample(self, sample: int) -> int:
        return max(0, min(len(self.analysis.samples) - 1, sample))

    def _seek_to_sample(self, target_sample: int) -> None:
        beat_index = int(np.searchsorted(self._beat_start_samples, target_sample, side="right") - 1)
        beat_index = max(0, min(len(self.analysis.beats) - 1, beat_index))
        beat = self.analysis.beats[beat_index]
        self._beat_index = beat_index
        self._sample_in_beat = max(0, min(beat.duration_samples - 1, target_sample - beat.start_sample))
        self._branch_chance = self.min_branch_chance
        self._finished = False
        self._transition = self._plan_transition()

    def _plan_transition(self) -> Transition:
        current = self._beat_index
        expected_next = current + 1
        if expected_next >= len(self.analysis.beats):
            return Transition(
                next_index=current,
                branch=None,
                end_playback=True,
            )

        branch: Branch | None = None
        expected_beat = self.analysis.beats[expected_next]
        branch_options = len(expected_beat.branches)
        branch_source = expected_next if branch_options > 0 else None
        jump_chance = 0.0
        if expected_beat.branches and self._branches_enabled():
            base_chance = self._advance_branch_chance()
            branch_candidate = self._choose_weighted_branch(expected_next, expected_beat.branches)
            jump_chance = self._adjusted_branch_chance(base_chance, expected_next, branch_candidate)
            should_branch = random.random() < jump_chance
            if should_branch:
                self._branch_chance = self.min_branch_chance
                branch = branch_candidate

        if branch is None:
            return Transition(
                next_index=expected_next,
                branch=None,
                branch_source=branch_source,
                jump_chance=jump_chance,
                branch_options=branch_options,
            )
        return Transition(
            next_index=branch.dest,
            branch=branch,
            branch_source=branch_source,
            jump_from=expected_next,
            jump_chance=jump_chance,
            branch_options=branch_options,
        )

    def _advance_branch_chance(self) -> float:
        self._branch_chance = min(
            self.max_branch_chance,
            self._branch_chance + self.branch_chance_delta,
        )
        return self._branch_chance

    def _choose_weighted_branch(self, source_index: int, branches: deque[Branch]) -> Branch | None:
        if not branches:
            return None

        weighted_branches = [
            (branch, 1.0 / (self._route_use_count(source_index, branch.dest) + 1.0))
            for branch in branches
        ]
        total_weight = sum(weight for _branch, weight in weighted_branches)
        target = random.random() * total_weight
        cumulative = 0.0
        for branch, weight in weighted_branches:
            cumulative += weight
            if cumulative >= target:
                return branch
        return weighted_branches[-1][0]

    def _least_used_branch(self, source_index: int) -> Branch | None:
        branches = self.analysis.beats[source_index].branches
        if not branches:
            return None
        return min(
            branches,
            key=lambda branch: (self._route_use_count(source_index, branch.dest), branch.distance),
        )

    def _adjusted_branch_chance(self, base_chance: float, source_index: int, branch: Branch | None) -> float:
        if branch is None:
            return 0.0
        if self._is_forced_end_branch(source_index):
            return 1.0

        usage_multiplier = UNUSED_BRANCH_CHANCE_MULTIPLIER / math.sqrt(
            self._route_use_count(source_index, branch.dest) + 1.0
        )
        chance = base_chance * usage_multiplier
        end_pressure = self._end_branch_pressure(source_index)
        if end_pressure > 0.0:
            chance += (END_BRANCH_MAX_CHANCE - chance) * end_pressure
        return min(END_BRANCH_MAX_CHANCE, max(0.0, chance))

    def _route_use_count(self, source_index: int, dest_index: int) -> int:
        return self._branch_use_counts.get((source_index, dest_index), 0)

    def _branches_enabled(self) -> bool:
        if self.timed_end_seconds is None:
            return True
        played_seconds = self._played_samples / self.analysis.sample_rate
        return played_seconds < self.timed_end_seconds

    def _normalize_branch_usage_counts(self) -> None:
        if not self._branch_routes:
            return

        minimum_usage = min(self._route_use_count(source_index, dest_index) for source_index, dest_index in self._branch_routes)
        if minimum_usage < BRANCH_USAGE_NORMALIZE_THRESHOLD:
            return

        for route in list(self._branch_use_counts):
            normalized_count = self._branch_use_counts[route] - minimum_usage
            if normalized_count > 0:
                self._branch_use_counts[route] = normalized_count
            else:
                del self._branch_use_counts[route]

    def _end_branch_pressure(self, source_index: int) -> float:
        if self._last_branch_source_index is None:
            return 0.0
        beats_before_last_branch = self._last_branch_source_index - source_index
        if beats_before_last_branch < 0 or beats_before_last_branch > END_BRANCH_BOOST_WINDOW_BEATS:
            return 0.0
        return 1.0 - (beats_before_last_branch / END_BRANCH_BOOST_WINDOW_BEATS)

    def _is_forced_end_branch(self, source_index: int) -> bool:
        if self._last_branch_source_index is None:
            return False
        beats_before_last_branch = self._last_branch_source_index - source_index
        return 0 <= beats_before_last_branch <= END_BRANCH_FORCE_WINDOW_BEATS

    def _next_branch_opportunity(self) -> tuple[int | None, float, int]:
        transition = self._transition
        if not self._branches_enabled():
            return None, 0.0, 0
        if transition.branch_source is not None:
            return transition.branch_source, transition.jump_chance, transition.branch_options
        if transition.end_playback:
            return None, 0.0, 0

        next_chance = min(self.max_branch_chance, self._branch_chance + self.branch_chance_delta)
        for beat in self.analysis.beats[transition.next_index + 1 :]:
            branch_options = len(beat.branches)
            if branch_options > 0:
                branch = self._least_used_branch(beat.index)
                jump_chance = self._adjusted_branch_chance(next_chance, beat.index, branch)
                return beat.index, jump_chance, branch_options
        return None, 0.0, 0

    def _emit_event(
        self,
        jump_from: int | None = None,
        jump: Branch | None = None,
        ended: bool = False,
    ) -> None:
        if self.on_event is None:
            return
        with self._lock:
            beat = self.analysis.beats[self._beat_index]
            planned_branch = self._transition.branch
            position_seconds = (
                len(self.analysis.samples) / self.analysis.sample_rate
                if ended
                else self._current_source_sample() / self.analysis.sample_rate
            )
            next_branch_index, next_branch_jump_chance, next_branch_options = self._next_branch_opportunity()
            event = PlaybackEvent(
                beat_index=self._beat_index,
                beats_played=self._beats_played,
                played_seconds=self._played_samples / self.analysis.sample_rate,
                position_seconds=position_seconds,
                jumps_taken=self._jumps_taken,
                jump_from=jump_from,
                jump_to=jump.dest if jump is not None else None,
                jump_distance=jump.distance if jump is not None else None,
                beat_loudness_db=beat.loudness_db,
                beat_duration_seconds=beat.duration_samples / self.analysis.sample_rate,
                bar_position=beat.bar_position,
                branch_options=self._transition.branch_options,
                jump_chance=self._transition.jump_chance,
                next_branch_index=next_branch_index,
                next_branch_jump_chance=next_branch_jump_chance,
                next_branch_options=next_branch_options,
                planned_next=self._transition.next_index,
                planned_jump_from=self._transition.jump_from,
                planned_jump_to=planned_branch.dest if planned_branch is not None else None,
                planned_jump_distance=planned_branch.distance if planned_branch is not None else None,
                ended=ended,
            )
        self.on_event(event)


def load_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    samples, sr = librosa.load(path, sr=sample_rate, mono=False)
    if samples.ndim == 1:
        samples = np.stack([samples, samples], axis=0)
    return np.ascontiguousarray(samples.T.astype(np.float32)), sr


def analyze_song(
    path: Path,
    sample_rate: int,
    max_branches: int,
    max_distance: float,
    max_branches_per_minute: float,
    beats_per_bar: int,
    long_branches_only: bool,
    backwards_only: bool,
    same_bar_only: bool,
) -> Analysis:
    print("Loading audio...")
    samples, sr = load_audio(path, sample_rate)
    mono = samples.mean(axis=1)
    duration = len(mono) / sr

    print("Detecting beats...")
    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times = np.array([beat_time for beat_time in beat_times if 0.0 <= beat_time < duration], dtype=np.float64)
    if len(beat_times) < 4:
        raise RuntimeError("Could not detect enough beats. Try a clearer song file.")

    beat_samples = np.unique(np.clip((beat_times * sr).astype(np.int64), 0, len(mono) - 1))
    if beat_samples[0] > int(0.15 * sr):
        beat_samples = np.insert(beat_samples, 0, 0)

    final_sample = len(mono)
    if final_sample - beat_samples[-1] < int(0.10 * sr):
        beat_samples = beat_samples[:-1]

    print("Extracting per-beat features...")
    features = extract_beat_features(mono, sr, beat_samples, final_sample)
    loudness_db = measure_beat_loudness_db(mono, beat_samples, final_sample)
    beat_count = len(features)
    if beat_count < 4:
        raise RuntimeError("The usable beat grid is too short for eternal playback.")

    norm_features = normalize_features(features)
    beats = [
        Beat(
            index=i,
            start_sample=int(beat_samples[i]),
            end_sample=int(beat_samples[i + 1] if i + 1 < len(beat_samples) else final_sample),
            feature=norm_features[i],
            loudness_db=float(loudness_db[i]),
            bar_position=i % beats_per_bar,
        )
        for i in range(beat_count)
    ]

    print("Finding similar beat branches...")
    connect_branches(
        beats,
        max_branches=max_branches,
        max_distance=max_distance,
        max_branches_per_minute=max_branches_per_minute,
        duration_seconds=duration,
        long_branches_only=long_branches_only,
        backwards_only=backwards_only,
        same_bar_only=same_bar_only,
    )

    branch_count = sum(len(beat.branches) for beat in beats)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    print(f"Analysis complete: {len(beats)} beats, tempo ~{tempo_value:.1f} BPM, {branch_count} branches.")
    if branch_count == 0:
        print("No strong branches found; playback will stop at the final beat.")

    return Analysis(samples=samples, sample_rate=sr, beats=beats, tempo=tempo_value)


def extract_beat_features(
    mono: np.ndarray,
    sample_rate: int,
    beat_samples: np.ndarray,
    final_sample: int,
) -> np.ndarray:
    hop_length = 512
    chroma = librosa.feature.chroma_stft(y=mono, sr=sample_rate, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=mono, sr=sample_rate, hop_length=hop_length, n_mfcc=13)
    contrast = librosa.feature.spectral_contrast(y=mono, sr=sample_rate, hop_length=hop_length)
    rms = librosa.feature.rms(y=mono, hop_length=hop_length)
    onset = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=hop_length)[None, :]
    all_features = np.vstack([chroma * 1.4, mfcc, contrast, rms * 3.0, onset])

    beat_count = len(beat_samples)
    result = []
    for i in range(beat_count):
        start = beat_samples[i]
        end = beat_samples[i + 1] if i + 1 < beat_count else final_sample
        frame_start = librosa.samples_to_frames(start, hop_length=hop_length)
        frame_end = max(frame_start + 1, librosa.samples_to_frames(end, hop_length=hop_length))
        frame_end = min(frame_end, all_features.shape[1])
        result.append(np.mean(all_features[:, frame_start:frame_end], axis=1))
    return np.vstack(result).astype(np.float32)


def normalize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std


def measure_beat_loudness_db(
    mono: np.ndarray,
    beat_samples: np.ndarray,
    final_sample: int,
) -> np.ndarray:
    beat_count = len(beat_samples)
    loudness = np.empty(beat_count, dtype=np.float32)
    for i in range(beat_count):
        start = int(beat_samples[i])
        end = int(beat_samples[i + 1] if i + 1 < beat_count else final_sample)
        segment = mono[start:end]
        rms = float(np.sqrt(np.mean(segment * segment))) if len(segment) else 0.0
        loudness[i] = 20.0 * math.log10(max(rms, 1e-9))
    return loudness


def connect_branches(
    beats: list[Beat],
    max_branches: int,
    max_distance: float,
    max_branches_per_minute: float,
    duration_seconds: float,
    long_branches_only: bool,
    backwards_only: bool,
    same_bar_only: bool,
) -> None:
    min_long_distance = max(12, len(beats) // 5)
    intro_target_skip = 4
    branch_limit = max(1, round(duration_seconds / 60.0 * max_branches_per_minute))
    branches_by_source: dict[int, list[Branch]] = {beat.index: [] for beat in beats}
    candidates: list[tuple[float, int, Branch]] = []

    for src in beats:
        for dest in beats:
            distance_in_beats = abs(src.index - dest.index)
            if src.index == dest.index or distance_in_beats <= 3:
                continue
            if dest.index < intro_target_skip:
                continue
            if backwards_only and dest.index > src.index:
                continue
            if long_branches_only and distance_in_beats < min_long_distance:
                continue
            if same_bar_only and src.bar_position != dest.bar_position:
                continue

            loudness_diff = abs(src.loudness_db - dest.loudness_db)
            if loudness_diff > MAX_LOUDNESS_DIFF_DB:
                continue

            feature_distance = float(np.linalg.norm(src.feature - dest.feature))
            loudness_penalty = loudness_diff * LOUDNESS_DISTANCE_WEIGHT
            position_penalty = 0.0 if src.bar_position == dest.bar_position else 4.0
            short_jump_penalty = max(0.0, (12.0 - distance_in_beats) / 12.0) * 2.0
            direction_penalty = 0.6 if dest.index > src.index else 0.0
            distance = (
                feature_distance
                + loudness_penalty
                + position_penalty
                + short_jump_penalty
                + direction_penalty
            )
            if distance <= max_distance:
                candidates.append((distance, src.index, Branch(dest=dest.index, distance=distance)))

    candidates.sort(key=lambda item: item[0])
    selected_count = 0
    selected_routes: list[tuple[int, int]] = []
    for _distance, src_index, branch in candidates:
        if is_nearby_equivalent_branch(src_index, branch.dest, selected_routes):
            continue
        source_branches = branches_by_source[src_index]
        if len(source_branches) >= max_branches:
            continue
        source_branches.append(branch)
        selected_routes.append((src_index, branch.dest))
        selected_count += 1
        if selected_count >= branch_limit:
            break

    for src in beats:
        src.branches = deque(branches_by_source[src.index])


def is_nearby_equivalent_branch(
    src_index: int,
    dest_index: int,
    selected_routes: list[tuple[int, int]],
) -> bool:
    route_offset = src_index - dest_index
    for selected_src, selected_dest in selected_routes:
        selected_offset = selected_src - selected_dest
        if abs(route_offset - selected_offset) > BRANCH_OFFSET_TOLERANCE_BEATS:
            continue
        source_is_near = abs(src_index - selected_src) <= NEARBY_BRANCH_WINDOW_BEATS
        destination_is_near = abs(dest_index - selected_dest) <= NEARBY_BRANCH_WINDOW_BEATS
        if source_is_near and destination_is_near:
            return True
    return False


def format_playback_time(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration_for_filename(seconds: float) -> str:
    total_seconds = max(1, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def render_mp3_export(
    analysis: Analysis,
    output_path: Path,
    duration_seconds: float,
    ffmpeg_path: Path,
    volume: float,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[int, int]:
    total_frames = max(1, int(round(duration_seconds * analysis.sample_rate)))
    chunk_frames = max(1, int(round(EXPORT_CHUNK_SECONDS * analysis.sample_rate)))
    player = EternalPlayer(
        analysis,
        min_branch_chance=0.18,
        max_branch_chance=0.50,
        branch_chance_delta=0.018,
        volume=volume,
    )
    command = [
        str(ffmpeg_path),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(analysis.sample_rate),
        "-ac",
        str(analysis.samples.shape[1]),
        "-i",
        "pipe:0",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        EXPORT_BITRATE,
        str(output_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    frames_written = 0

    try:
        assert process.stdin is not None
        while frames_written < total_frames:
            requested = min(chunk_frames, total_frames - frames_written)
            chunk, ended = player.render_frames(requested)
            if len(chunk) == 0:
                raise RuntimeError(
                    "Playback ended before the requested export length. "
                    "Try a lower match threshold, more branches, or a shorter export."
                )

            raw_chunk = np.ascontiguousarray(chunk, dtype=np.float32).astype("<f4", copy=False)
            process.stdin.write(raw_chunk.tobytes())
            frames_written += len(chunk)

            if progress_callback is not None:
                progress_callback(frames_written / analysis.sample_rate)

            if ended and frames_written < total_frames:
                raise RuntimeError(
                    "Playback ended before the requested export length. "
                    "Try a lower match threshold, more branches, or a shorter export."
                )

        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        returncode = process.wait()
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.kill()
        except OSError:
            pass
        process.wait()
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        raise

    if returncode != 0:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        message = stderr.decode(errors="replace").strip()
        raise RuntimeError(message or f"ffmpeg failed with exit code {returncode}.")

    return player._beats_played, player._jumps_taken


class EternalJukeboxApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Song Eternal")
        self.analysis: Analysis | None = None
        self.player: EternalPlayer | None = None
        self.events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.current_event: PlaybackEvent | None = None
        self.last_jump: tuple[int, int, float] | None = None
        self.pending_play = False
        self.exporting = False
        self.ffmpeg_path: Path | None = None
        self.stat_labels: dict[str, ttk.Label] = {}

        default_mp3 = find_default_mp3()
        self.file_var = tk.StringVar(value=str(default_mp3) if default_mp3 else "")
        initial_status = f"Auto-selected {default_mp3.name}. Ready to analyze." if default_mp3 else "Choose an audio file."
        self.status_var = tk.StringVar(value=initial_status)
        self.stats_var = tk.StringVar(value="No analysis loaded.")
        self.max_distance_var = tk.DoubleVar(value=1.0)
        self.max_branches_var = tk.IntVar(value=4)
        self.branch_rate_var = tk.DoubleVar(value=DEFAULT_MAX_BRANCHES_PER_MINUTE)
        self.volume_var = tk.DoubleVar(value=0.55)
        self.long_var = tk.BooleanVar(value=True)
        self.same_bar_var = tk.BooleanVar(value=True)
        self.timed_end_var = tk.BooleanVar(value=False)
        self.listen_hours_var = tk.IntVar(value=0)
        self.listen_minutes_var = tk.IntVar(value=30)
        self.timed_end_widgets: list[ttk.Widget] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(80, self._poll_events)

    def _build_ui(self) -> None:
        self.root.geometry("980x650")
        self.root.minsize(820, 540)

        style = ttk.Style()
        style.configure("TButton", padding=(10, 6))
        style.configure("TLabel", padding=(0, 2))

        main = ttk.Frame(self.root, padding=14)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main)
        top.pack(fill=tk.X)
        ttk.Entry(top, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(top, text="Open", command=self._choose_file).pack(side=tk.LEFT)
        ttk.Button(top, text="Analyze", command=lambda: self._analyze(False)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Show Branches", command=self._show_branches).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Play", command=self._play).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Stop", command=self._stop).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="-10s", command=lambda: self._seek_relative(-SEEK_STEP_SECONDS)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="+10s", command=lambda: self._seek_relative(SEEK_STEP_SECONDS)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Export MP3", command=self._export_mp3).pack(side=tk.LEFT, padx=(8, 0))

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=(12, 10))
        self._slider(
            controls,
            "Match threshold",
            self.max_distance_var,
            0.1,
            16.0,
            0,
            value_input=True,
            value_increment=0.05,
        )
        self._slider(controls, "Volume", self.volume_var, 0.0, 1.0, 1)

        checks = ttk.Frame(main)
        checks.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(checks, text="Backwards jumps only").pack(side=tk.LEFT)
        ttk.Checkbutton(checks, text="Long jumps", variable=self.long_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Checkbutton(checks, text="Same bar", variable=self.same_bar_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Label(checks, text="Branches/min").pack(side=tk.LEFT, padx=(24, 6))
        ttk.Spinbox(checks, from_=1, to=30, increment=1, textvariable=self.branch_rate_var, width=5).pack(side=tk.LEFT)
        ttk.Label(checks, text="Per beat").pack(side=tk.LEFT, padx=(14, 6))
        ttk.Spinbox(checks, from_=1, to=12, textvariable=self.max_branches_var, width=5).pack(side=tk.LEFT)

        timed = ttk.Frame(main)
        timed.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            timed,
            text="Timed ending",
            variable=self.timed_end_var,
            command=self._sync_timed_end_controls,
        ).pack(side=tk.LEFT)
        ttk.Label(timed, text="Hours").pack(side=tk.LEFT, padx=(14, 6))
        hours = ttk.Spinbox(timed, from_=0, to=24, increment=1, textvariable=self.listen_hours_var, width=5)
        hours.pack(side=tk.LEFT)
        ttk.Label(timed, text="Minutes").pack(side=tk.LEFT, padx=(14, 6))
        minutes = ttk.Spinbox(timed, from_=0, to=59, increment=1, textvariable=self.listen_minutes_var, width=5)
        minutes.pack(side=tk.LEFT)
        self.timed_end_widgets = [hours, minutes]
        self._sync_timed_end_controls()

        middle = ttk.Frame(main)
        middle.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(middle, height=330, bg="#111820", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw_visual())
        self._build_stats_panel(middle)

        bottom = ttk.Frame(main)
        bottom.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Label(bottom, textvariable=self.stats_var).pack(side=tk.RIGHT)

    def _build_stats_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Stats", padding=(10, 8))
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        panel.columnconfigure(1, weight=1)

        rows = [
            ("duration", "Duration"),
            ("tempo", "Tempo"),
            ("beat_count", "Beats"),
            ("branch_count", "Branches"),
            ("branch_rate", "Branches/min"),
            ("played", "Played"),
            ("playback_time", "Playback time"),
            ("position", "Song position"),
            ("jumps", "Jumps"),
            ("current_beat", "Current beat"),
            ("bar_position", "Bar pos"),
            ("beat_duration", "Beat length"),
            ("loudness", "Loudness"),
            ("jump_chance", "Next branch chance"),
            ("branch_options", "Next branch opts"),
            ("planned", "Planned"),
            ("planned_distance", "Planned dist"),
            ("last_jump", "Last jump"),
            ("last_distance", "Last distance"),
        ]
        for row, (key, label) in enumerate(rows):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky=tk.W, pady=1)
            value = ttk.Label(panel, text="-", anchor=tk.E)
            value.grid(row=row, column=1, sticky=tk.EW, padx=(12, 0), pady=1)
            self.stat_labels[key] = value

    def _slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        start: float,
        end: float,
        column: int,
        value_input: bool = False,
        value_increment: float = 0.1,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, sticky="ew", padx=(0, 16))
        parent.columnconfigure(column, weight=1)
        header = ttk.Frame(frame)
        header.pack(fill=tk.X)
        ttk.Label(header, text=label).pack(side=tk.LEFT)
        if value_input:
            ttk.Spinbox(
                header,
                from_=start,
                to=end,
                increment=value_increment,
                textvariable=variable,
                width=7,
                format="%.2f",
            ).pack(side=tk.RIGHT)
        ttk.Scale(frame, variable=variable, from_=start, to=end, orient=tk.HORIZONTAL).pack(fill=tk.X)

    def _sync_timed_end_controls(self) -> None:
        state = tk.NORMAL if self.timed_end_var.get() else tk.DISABLED
        for widget in self.timed_end_widgets:
            widget.configure(state=state)

    def _choose_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choose audio",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.file_var.set(filename)
            self.analysis = None
            self.current_event = None
            self.last_jump = None
            self.status_var.set("Ready to analyze.")
            self.stats_var.set("No analysis loaded.")
            self._clear_stats()
            self._draw_visual()

    def _set_stat(self, key: str, value: object) -> None:
        label = self.stat_labels.get(key)
        if label is not None:
            label.configure(text=str(value))

    def _clear_stats(self) -> None:
        for label in self.stat_labels.values():
            label.configure(text="-")

    def _show_branches(self) -> None:
        if self.analysis is None:
            messagebox.showinfo("Branches", "Analyze a song first to see its branches.")
            return

        rows = []
        for beat in self.analysis.beats:
            source_time = beat.start_sample / self.analysis.sample_rate
            for branch in beat.branches:
                dest = self.analysis.beats[branch.dest]
                dest_time = dest.start_sample / self.analysis.sample_rate
                loudness_diff = abs(beat.loudness_db - dest.loudness_db)
                rows.append(
                    (
                        len(rows) + 1,
                        beat.index + 1,
                        branch.dest + 1,
                        source_time,
                        dest_time,
                        branch.dest - beat.index,
                        branch.distance,
                        loudness_diff,
                    )
                )

        if not rows:
            messagebox.showinfo("Branches", "No branches were found for this analysis.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Branches ({len(rows)})")
        dialog.geometry("760x430")
        dialog.minsize(640, 320)
        dialog.transient(self.root)

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        columns = ("number", "source", "dest", "source_time", "dest_time", "offset", "distance", "loudness")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        tree.grid(row=0, column=0, sticky="nsew")

        headings = {
            "number": "#",
            "source": "From beat",
            "dest": "To beat",
            "source_time": "From sec",
            "dest_time": "To sec",
            "offset": "Offset",
            "distance": "Distance",
            "loudness": "Loudness dB",
        }
        widths = {
            "number": 54,
            "source": 82,
            "dest": 82,
            "source_time": 86,
            "dest_time": 86,
            "offset": 72,
            "distance": 86,
            "loudness": 92,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor=tk.E, stretch=column in {"distance", "loudness"})

        for row in rows:
            number, source, dest, source_time, dest_time, offset, distance, loudness_diff = row
            tree.insert(
                "",
                tk.END,
                values=(
                    number,
                    source,
                    dest,
                    f"{source_time:.2f}",
                    f"{dest_time:.2f}",
                    offset,
                    f"{distance:.3f}",
                    f"{loudness_diff:.1f}",
                ),
            )

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scrollbar.set)

        summary = ttk.Label(frame, text=f"{len(rows)} branches across {len(self.analysis.beats)} beats")
        summary.grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Button(frame, text="Close", command=dialog.destroy).grid(row=1, column=0, sticky=tk.E, pady=(8, 0))

    def _refresh_analysis_stats(self) -> tuple[int, float]:
        if self.analysis is None:
            self._clear_stats()
            return 0, 0.0

        duration_minutes = len(self.analysis.samples) / self.analysis.sample_rate / 60.0
        branch_count = sum(len(beat.branches) for beat in self.analysis.beats)
        branch_rate = branch_count / duration_minutes if duration_minutes > 0 else 0.0
        self._set_stat("duration", f"{duration_minutes:.2f} min")
        self._set_stat("tempo", f"{self.analysis.tempo:.1f} BPM")
        self._set_stat("beat_count", len(self.analysis.beats))
        self._set_stat("branch_count", branch_count)
        self._set_stat("branch_rate", f"{branch_rate:.1f}")
        self._set_stat("played", "0")
        self._set_stat("playback_time", format_playback_time(0))
        self._set_stat("position", format_playback_time(0))
        self._set_stat("jumps", "0")
        self._set_stat("current_beat", "-")
        self._set_stat("bar_position", "-")
        self._set_stat("beat_duration", "-")
        self._set_stat("loudness", "-")
        self._set_stat("jump_chance", "-")
        self._set_stat("branch_options", "-")
        self._set_stat("planned", "-")
        self._set_stat("planned_distance", "-")
        self._set_stat("last_jump", "-")
        self._set_stat("last_distance", "-")
        return branch_count, branch_rate

    def _refresh_playback_stats(self, event: PlaybackEvent) -> None:
        if self.analysis is None:
            return

        self._set_stat("played", event.beats_played)
        self._set_stat("playback_time", format_playback_time(event.played_seconds))
        self._set_stat("position", format_playback_time(event.position_seconds))
        self._set_stat("jumps", event.jumps_taken)
        self._set_stat("current_beat", f"{event.beat_index + 1}/{len(self.analysis.beats)}")
        self._set_stat("bar_position", event.bar_position + 1)
        self._set_stat("beat_duration", f"{event.beat_duration_seconds:.3f}s")
        self._set_stat("loudness", f"{event.beat_loudness_db:.1f} dB")
        if event.next_branch_index is not None:
            self._set_stat("jump_chance", f"{event.next_branch_jump_chance * 100:.1f}% @ {event.next_branch_index + 1}")
            self._set_stat("branch_options", event.next_branch_options)
        else:
            self._set_stat("jump_chance", "-")
            self._set_stat("branch_options", "-")

        if event.ended:
            planned = "End"
        elif event.planned_jump_from is not None and event.planned_jump_to is not None:
            planned = f"{event.planned_jump_from + 1} -> {event.planned_jump_to + 1}"
        elif event.planned_next is not None:
            planned = f"Next {event.planned_next + 1}"
        else:
            planned = "-"
        self._set_stat("planned", planned)
        if event.planned_jump_distance is not None:
            self._set_stat("planned_distance", f"{event.planned_jump_distance:.2f}")
        else:
            self._set_stat("planned_distance", "-")

        if event.jump_from is not None and event.jump_to is not None and event.jump_distance is not None:
            source = self.analysis.beats[event.jump_from]
            dest = self.analysis.beats[event.jump_to]
            loudness_diff = abs(source.loudness_db - dest.loudness_db)
            self._set_stat("last_jump", f"{event.jump_from + 1} -> {event.jump_to + 1}")
            self._set_stat("last_distance", f"{event.jump_distance:.2f} / {loudness_diff:.1f} dB")

    def _analyze(self, auto_play: bool) -> None:
        path = Path(self.file_var.get().strip('" '))
        if not path.exists():
            messagebox.showerror("Missing file", "Choose a valid audio file first.")
            return
        self._stop()
        self.pending_play = auto_play
        self.status_var.set("Analyzing audio...")
        self.stats_var.set("This can take a moment for long songs.")
        self._clear_stats()
        self._draw_visual()

        def worker() -> None:
            try:
                analysis = analyze_song(
                    path=path,
                    sample_rate=44100,
                    max_branches=int(self.max_branches_var.get()),
                    max_distance=float(self.max_distance_var.get()),
                    max_branches_per_minute=float(self.branch_rate_var.get()),
                    beats_per_bar=4,
                    long_branches_only=bool(self.long_var.get()),
                    backwards_only=True,
                    same_bar_only=bool(self.same_bar_var.get()),
                )
                self.events.put(("analysis", analysis))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _timed_end_seconds(self) -> float | None:
        if not self.timed_end_var.get():
            return None
        try:
            hours = max(0, int(self.listen_hours_var.get()))
            minutes = max(0, int(self.listen_minutes_var.get()))
        except (tk.TclError, ValueError):
            return None
        total_seconds = float((hours * 60 + minutes) * 60)
        return total_seconds if total_seconds > 0 else None

    def _play(self) -> None:
        if self.analysis is None:
            self._analyze(True)
            return
        timed_end_seconds = self._timed_end_seconds()
        if self.timed_end_var.get() and timed_end_seconds is None:
            messagebox.showerror("Timed ending", "Enter at least 1 minute or disable timed ending.")
            return
        self._stop()
        self.player = EternalPlayer(
            self.analysis,
            min_branch_chance=0.18,
            max_branch_chance=0.50,
            branch_chance_delta=0.018,
            volume=float(self.volume_var.get()),
            on_event=lambda event: self.events.put(("playback", event)),
            timed_end_seconds=timed_end_seconds,
        )
        try:
            self.player.start()
            self.status_var.set("Playing.")
        except Exception as exc:
            self.player = None
            messagebox.showerror("Playback failed", str(exc))

    def _export_mp3(self) -> None:
        if self.exporting:
            self.status_var.set("Export already in progress.")
            return
        if self.analysis is None:
            messagebox.showerror("Export MP3", "Analyze a song before exporting.")
            return
        analysis = self.analysis

        duration_seconds = self._ask_export_duration()
        if duration_seconds is None:
            return

        branch_count = sum(len(beat.branches) for beat in analysis.beats)
        source_seconds = len(analysis.samples) / analysis.sample_rate
        if branch_count == 0 and duration_seconds > source_seconds:
            messagebox.showerror(
                "Export MP3",
                "No branches were found, so this analysis cannot extend the song beyond its original length.",
            )
            return

        source_path = Path(self.file_var.get().strip('" '))
        duration_label = format_duration_for_filename(duration_seconds)
        initial_name = f"{source_path.stem or 'song'}_eternal_{duration_label}.mp3"
        output_filename = filedialog.asksaveasfilename(
            title="Save MP3 export",
            defaultextension=".mp3",
            initialfile=initial_name,
            filetypes=[("MP3 audio", "*.mp3"), ("All files", "*.*")],
        )
        if not output_filename:
            return

        ffmpeg_path = self._resolve_ffmpeg_path()
        if ffmpeg_path is None:
            self.status_var.set("Export canceled; ffmpeg.exe was not selected.")
            return

        output_path = Path(output_filename)
        self._stop()
        export_volume = float(self.volume_var.get())
        self.exporting = True
        self.status_var.set(f"Exporting MP3: 00:00:00 / {format_playback_time(duration_seconds)}")
        self.stats_var.set(str(output_path))

        def worker() -> None:
            try:
                result = render_mp3_export(
                    analysis=analysis,
                    output_path=output_path,
                    duration_seconds=duration_seconds,
                    ffmpeg_path=ffmpeg_path,
                    volume=export_volume,
                    progress_callback=lambda seconds: self.events.put(
                        ("export_progress", (seconds, duration_seconds, output_path))
                    ),
                )
                self.events.put(("export_done", (output_path, duration_seconds, result[0], result[1])))
            except Exception as exc:
                self.events.put(("export_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _ask_export_duration(self) -> float | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Length")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        hours_var = tk.IntVar(value=10)
        minutes_var = tk.IntVar(value=0)
        seconds_var = tk.IntVar(value=0)
        result: list[float | None] = [None]

        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Hours").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 8))
        ttk.Spinbox(frame, from_=0, to=240, increment=1, textvariable=hours_var, width=7).grid(
            row=0,
            column=1,
            sticky=tk.W,
            pady=(0, 8),
        )
        ttk.Label(frame, text="Minutes").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 8))
        ttk.Spinbox(frame, from_=0, to=59, increment=1, textvariable=minutes_var, width=7).grid(
            row=1,
            column=1,
            sticky=tk.W,
            pady=(0, 8),
        )
        ttk.Label(frame, text="Seconds").grid(row=2, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 12))
        ttk.Spinbox(frame, from_=0, to=59, increment=1, textvariable=seconds_var, width=7).grid(
            row=2,
            column=1,
            sticky=tk.W,
            pady=(0, 12),
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky=tk.E)

        def confirm() -> None:
            try:
                hours = max(0, int(hours_var.get()))
                minutes = max(0, int(minutes_var.get()))
                seconds = max(0, int(seconds_var.get()))
            except (tk.TclError, ValueError):
                messagebox.showerror("Export Length", "Enter whole numbers for the export length.", parent=dialog)
                return

            total_seconds = float(hours * 3600 + minutes * 60 + seconds)
            if total_seconds <= 0:
                messagebox.showerror("Export Length", "Enter an export length greater than zero.", parent=dialog)
                return

            result[0] = total_seconds
            dialog.destroy()

        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Export", command=confirm).pack(side=tk.RIGHT, padx=(0, 8))

        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set()
        dialog.wait_window()
        return result[0]

    def _resolve_ffmpeg_path(self) -> Path | None:
        if self.ffmpeg_path is not None and self.ffmpeg_path.exists():
            return self.ffmpeg_path

        local_ffmpeg = Path(__file__).resolve().with_name("ffmpeg.exe")
        if local_ffmpeg.exists():
            self.ffmpeg_path = local_ffmpeg
            return local_ffmpeg

        path_ffmpeg = shutil.which("ffmpeg")
        if path_ffmpeg is not None:
            self.ffmpeg_path = Path(path_ffmpeg)
            return self.ffmpeg_path

        filename = filedialog.askopenfilename(
            title="Choose ffmpeg.exe",
            filetypes=[("ffmpeg.exe", "ffmpeg.exe"), ("Executables", "*.exe"), ("All files", "*.*")],
        )
        if not filename:
            return None

        self.ffmpeg_path = Path(filename)
        return self.ffmpeg_path

    def _seek_relative(self, seconds: float) -> None:
        if self.player is None:
            if self.analysis is None:
                self.status_var.set("Analyze a song before seeking.")
            else:
                self.status_var.set("Press Play before seeking.")
            return

        position = self.player.seek_seconds(seconds)
        direction = "forward" if seconds > 0 else "back"
        self.status_var.set(f"Skipped {direction} {abs(seconds):.0f}s to {format_playback_time(position)}.")

    def _stop(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.player = None
        if self.analysis is not None:
            self.status_var.set("Stopped.")

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == "analysis":
                self.analysis = payload  # type: ignore[assignment]
                self.current_event = None
                self.last_jump = None
                branch_count, branch_rate = self._refresh_analysis_stats()
                self.status_var.set("Analysis ready.")
                self.stats_var.set(
                    f"{len(self.analysis.beats)} beats | {branch_count} branches "
                    f"({branch_rate:.1f}/min) | {self.analysis.tempo:.1f} BPM"
                )
                self._draw_visual()
                if self.pending_play:
                    self.pending_play = False
                    self._play()
            elif kind == "playback":
                event = payload
                assert isinstance(event, PlaybackEvent)
                self.current_event = event
                if event.jump_from is not None and event.jump_to is not None and event.jump_distance is not None:
                    self.last_jump = (event.jump_from, event.jump_to, event.jump_distance)
                self._refresh_playback_stats(event)
                if event.ended:
                    self._stop()
                    self.status_var.set("Finished.")
                    self.stats_var.set(
                        f"Finished | Time {format_playback_time(event.played_seconds)} | "
                        f"Played {event.beats_played} | Jumps {event.jumps_taken}"
                    )
                else:
                    self.stats_var.set(
                        f"Beat {event.beat_index + 1}/{len(self.analysis.beats) if self.analysis else 0} | "
                        f"Position {format_playback_time(event.position_seconds)} | "
                        f"Played {event.beats_played} | Jumps {event.jumps_taken}"
                    )
                self._draw_visual()
            elif kind == "error":
                self.pending_play = False
                self.status_var.set("Analysis failed.")
                messagebox.showerror("Analysis failed", str(payload))
            elif kind == "export_progress":
                seconds, duration_seconds, output_path = payload  # type: ignore[misc]
                self.status_var.set(
                    f"Exporting MP3: {format_playback_time(float(seconds))} / "
                    f"{format_playback_time(float(duration_seconds))}"
                )
                self.stats_var.set(str(output_path))
            elif kind == "export_done":
                output_path, duration_seconds, beats_played, jumps_taken = payload  # type: ignore[misc]
                self.exporting = False
                self.status_var.set("Export finished.")
                self.stats_var.set(
                    f"{Path(output_path).name} | {format_playback_time(float(duration_seconds))} | "
                    f"Played {beats_played} | Jumps {jumps_taken}"
                )
                messagebox.showinfo("Export MP3", f"Export finished:\n{output_path}")
            elif kind == "export_error":
                self.exporting = False
                self.status_var.set("Export failed.")
                messagebox.showerror("Export MP3 failed", str(payload))

        self.root.after(80, self._poll_events)

    def _draw_visual(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())

        canvas.create_rectangle(0, 0, width, height, fill="#111820", outline="")
        if self.analysis is None:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Open an audio file to see the beat map",
                fill="#aeb9c4",
                font=("Segoe UI", 16),
            )
            return

        mid = height * 0.58
        wave_top = height * 0.30
        wave_scale = height * 0.22
        mono = self.analysis.samples.mean(axis=1)
        total = len(mono)
        step = max(1, total // width)
        for x in range(width):
            start = x * step
            end = min(total, start + step)
            if start >= total:
                break
            peak = float(np.max(np.abs(mono[start:end]))) if end > start else 0.0
            y1 = mid - peak * wave_scale
            y2 = mid + peak * wave_scale
            color = "#5b8fb9" if x % 2 else "#46799f"
            canvas.create_line(x, y1, x, y2, fill=color)

        beat_step = max(1, len(self.analysis.beats) // 260)
        for beat in self.analysis.beats[::beat_step]:
            x = beat.start_sample / total * width
            canvas.create_line(x, wave_top, x, mid + wave_scale + 8, fill="#263545")

        arcs = []
        for beat in self.analysis.beats:
            if beat.branches:
                arcs.append((beat.index, beat.branches[0].dest, beat.branches[0].distance))
        arcs.sort(key=lambda item: item[2])
        for src, dest, _distance in arcs[:180]:
            x1 = self.analysis.beats[src].start_sample / total * width
            x2 = self.analysis.beats[dest].start_sample / total * width
            top = 28 + min(120, abs(x2 - x1) * 0.18)
            canvas.create_line(x1, wave_top, (x1 + x2) / 2, top, x2, wave_top, smooth=True, fill="#345b70")

        if self.last_jump is not None:
            src, dest, distance = self.last_jump
            x1 = self.analysis.beats[src].start_sample / total * width
            x2 = self.analysis.beats[dest].start_sample / total * width
            canvas.create_line(x1, wave_top, (x1 + x2) / 2, 18, x2, wave_top, smooth=True, fill="#43d17a", width=3)
            canvas.create_text(
                min(width - 90, max(90, (x1 + x2) / 2)),
                18,
                text=f"{src + 1} -> {dest + 1} ({distance:.2f})",
                fill="#c8f7d8",
                font=("Segoe UI", 10),
            )

        if self.current_event is not None:
            beat = self.analysis.beats[self.current_event.beat_index]
            x = beat.start_sample / total * width
            canvas.create_line(x, 0, x, height, fill="#f0d35b", width=2)

        canvas.create_text(12, height - 24, anchor=tk.W, text="Waveform, beat grid, branch map, and live jumps", fill="#aeb9c4")

    def _close(self) -> None:
        if self.exporting and not messagebox.askyesno(
            "Export in progress",
            "An MP3 export is still running. Close the app anyway?",
        ):
            return
        self._stop()
        self.root.destroy()


def run_gui() -> int:
    root = tk.Tk()
    EternalJukeboxApp(root)
    root.mainloop()
    return 0


def find_default_mp3() -> Path | None:
    app_dir = Path(__file__).resolve().parent
    mp3s = sorted(app_dir.glob("*.mp3"), key=lambda path: path.name.lower())
    return mp3s[0] if mp3s else None


def main() -> int:
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
