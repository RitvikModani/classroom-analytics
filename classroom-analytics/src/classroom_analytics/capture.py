"""Threaded video capture with a bounded queue and reconnect.

The grabber runs on its own thread so a slow inference loop never blocks frame
acquisition, and vice-versa. The queue is bounded and drops the oldest frame
under backpressure — we always process the freshest frame, never accumulate a
growing backlog. A read failure (camera unplugged, RTSP drop) triggers a
release + retry loop instead of crashing.
"""

from __future__ import annotations

import queue
import threading
import time

import cv2


class FrameGrabber:
    def __init__(self, source=0, queue_size: int = 4, reconnect_delay: float = 2.0):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self._q: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> "FrameGrabber":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self


    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            self._connected.set()
        return cap

    def _run(self):
        cap = self._open()
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                self._connected.clear()
                time.sleep(self.reconnect_delay)
                cap = self._open()
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                # source dropped — release and retry, don't die
                self._connected.clear()
                cap.release()
                time.sleep(self.reconnect_delay)
                cap = self._open()
                continue

            self._connected.set()
            self._push(frame)

        if cap is not None:
            cap.release()

    def _push(self, frame):
        # drop-oldest: keep the queue fresh under backpressure
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout: float = 1.0):
        """Return the freshest frame, or None if none arrived in `timeout`s."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
