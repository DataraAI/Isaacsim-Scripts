#!/usr/bin/env python3
"""Mirror stdout/stderr to a log file at the OS file-descriptor level."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path


class RunOutputTee:
    """
    Mirror process stdout/stderr to the terminal and one overwrite-on-run file.

    This operates at the OS file-descriptor level, so it captures ordinary
    Python prints plus native Isaac/RTX output written directly to stdout or
    stderr.
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._log_fd: int | None = None
        self._pipe_read_fd: int | None = None
        self._pipe_write_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)

        while view:
            written = os.write(fd, view)

            if written <= 0:
                raise RuntimeError("Console tee write returned no progress.")

            view = view[written:]

    def _copy_output(self) -> None:
        if (
            self._pipe_read_fd is None
            or self._saved_stdout_fd is None
            or self._log_fd is None
        ):
            return

        try:
            while True:
                chunk = os.read(self._pipe_read_fd, 65536)

                if not chunk:
                    break

                self._write_all(self._saved_stdout_fd, chunk)
                self._write_all(self._log_fd, chunk)
        except OSError:
            # Shutdown may close descriptors while the reader is exiting.
            pass

    def start(self) -> None:
        if self._started:
            return

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sys.stdout.flush()
        sys.stderr.flush()

        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        self._log_fd = os.open(
            self.output_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644,
        )

        (
            self._pipe_read_fd,
            self._pipe_write_fd,
        ) = os.pipe()

        os.dup2(self._pipe_write_fd, 1)
        os.dup2(self._pipe_write_fd, 2)
        os.close(self._pipe_write_fd)
        self._pipe_write_fd = None

        self._thread = threading.Thread(
            target=self._copy_output,
            name="run-output-tee",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        sys.stdout.flush()
        sys.stderr.flush()

        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)

        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        descriptors = (
            self._pipe_read_fd,
            self._log_fd,
            self._saved_stdout_fd,
            self._saved_stderr_fd,
        )

        for fd in descriptors:
            if fd is None:
                continue

            try:
                os.close(fd)
            except OSError:
                pass

        self._pipe_read_fd = None
        self._log_fd = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None
        self._thread = None
        self._started = False


