"""Background jobs, so loading and fitting never freeze the window.

    from empower_explorer.workers import run_async
    run_async(fn, arg1, arg2, on_done=callback, on_error=err_callback, on_progress=prog)

fn runs in a QThreadPool thread; the callbacks run in the GUI thread.  If fn accepts a keyword
argument `progress`, it receives a callable progress(done, total) that may be called often.
Outdated results (e.g. the user already selected another galaxy) are dropped by the caller:
remember what was requested and compare it in on_done, or set job.cancelled = True.
"""
import inspect
import traceback

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

_LIVE = set()      # keeps signal objects alive until the job reports back


class _Signals(QObject):
    done = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)


class Job(QRunnable):
    def __init__(self, fn, args, kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()
        self.cancelled = False
        self.setAutoDelete(True)

    def _emit(self, signal, *args):
        """Emit unless cancelled; the receiver may be gone when the app is closing."""
        if self.cancelled:
            return
        try:
            getattr(self.signals, signal).emit(*args)
        except RuntimeError:
            self.cancelled = True

    def run(self):
        try:
            kwargs = dict(self.kwargs)
            try:
                if 'progress' in inspect.signature(self.fn).parameters:
                    kwargs['progress'] = lambda done, total: self._emit('progress', int(done), int(total))
            except (TypeError, ValueError):
                pass
            result = self.fn(*self.args, **kwargs)
        except Exception as e:
            self._emit('error', f'{type(e).__name__}: {e}\n\n{traceback.format_exc()}')
            return
        self._emit('done', result)


def run_async(fn, *args, on_done=None, on_error=None, on_progress=None, **kwargs):
    """Run fn(*args, **kwargs) in the thread pool; returns the Job (job.cancelled = True drops it)."""
    job = Job(fn, args, kwargs)
    sig = job.signals
    _LIVE.add(sig)

    def _finish(*_):
        _LIVE.discard(sig)

    if on_done is not None:
        sig.done.connect(on_done)
    if on_error is not None:
        sig.error.connect(on_error)
    if on_progress is not None:
        sig.progress.connect(on_progress)
    sig.done.connect(_finish)
    sig.error.connect(_finish)
    QThreadPool.globalInstance().start(job)
    return job
