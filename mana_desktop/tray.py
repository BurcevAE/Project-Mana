"""
mana_desktop.tray — a notification-area icon, so closing the window does
not kill a running research cycle.

Why this exists
---------------
Closing the window called `session.close()`, which is correct for an app
whose window IS the app and wrong for this one: a cognitive cycle runs for
minutes, spends a real budget of brain calls, and the only way to keep it
alive was to leave a window open and not touch it. With a tray icon, the
window closes to the tray and the cycle keeps its budget.

Why ctypes and not pystray
--------------------------
pystray draws its icon with Pillow, and Pillow is on the list of packages
this build deliberately does not carry (packaging_deps). Adding a 10 MB
imaging library and a dependency whose absence breaks the tray, in order
to draw a 16x16 icon that Windows will happily take straight from the
executable, is the wrong trade. Shell_NotifyIcon is the API pystray would
have called anyway.

The window and the message loop live on this thread, not the main one:
pywebview owns the main thread from `webview.start()` until the last
window closes, and a second GetMessage loop there would never run. Windows
gives each thread its own message queue, so an icon owned by a window
created here is serviced here.

Everything in this module is best-effort. A tray icon that fails to
appear must never stop MANA from starting, so `start()` returns None on
any failure and the caller carries on with an ordinary window.
"""
from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes
from typing import Callable, Optional

APP_TITLE = "MANA"

WM_DESTROY = 0x0002
WM_COMMAND = 0x0111
WM_LBUTTONUP = 0x0202
WM_LBUTTONDBLCLK = 0x0203
WM_RBUTTONUP = 0x0205
WM_TRAY = 0x0400 + 20            # WM_USER + 20, our callback message

NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP, NIF_INFO = 0x01, 0x02, 0x04, 0x10
NIIF_INFO = 0x01

MF_STRING, MF_SEPARATOR = 0x0000, 0x0800
TPM_RIGHTBUTTON, TPM_RETURNCMD = 0x0002, 0x0100

IDM_OPEN, IDM_QUIT = 1001, 1002
IDI_APPLICATION = 32512

WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_longlong, wintypes.HWND, wintypes.UINT,
                             wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND),
                ("uID", wintypes.UINT), ("uFlags", wintypes.UINT),
                ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                ("szTip", wintypes.WCHAR * 128),
                ("dwState", wintypes.DWORD), ("dwStateMask", wintypes.DWORD),
                ("szInfo", wintypes.WCHAR * 256), ("uVersion", wintypes.UINT),
                ("szInfoTitle", wintypes.WCHAR * 64),
                ("dwInfoFlags", wintypes.DWORD)]


def _declare() -> None:
    """Give ctypes the real signatures for the calls that return handles.

    Without this every handle below is silently truncated to 32 bits (see
    the module note): ctypes assumes c_int for any function whose restype
    was never set, and on 64-bit Windows an HWND does not fit in one.
    """
    user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
    kernel32 = ctypes.windll.kernel32

    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]

    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]

    user32.DefWindowProcW.restype = ctypes.c_longlong
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                      wintypes.WPARAM, wintypes.LPARAM]

    user32.LoadIconW.restype = wintypes.HICON
    user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]

    user32.CreatePopupMenu.restype = wintypes.HMENU
    user32.CreatePopupMenu.argtypes = []

    user32.TrackPopupMenu.restype = ctypes.c_int
    user32.TrackPopupMenu.argtypes = [
        wintypes.HMENU, wintypes.UINT, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, wintypes.HWND, wintypes.LPVOID]

    user32.DestroyMenu.argtypes = [wintypes.HMENU]
    user32.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT,
                                   ctypes.c_void_p, wintypes.LPCWSTR]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                    wintypes.WPARAM, wintypes.LPARAM]

    shell32.ExtractIconW.restype = wintypes.HICON
    shell32.ExtractIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR,
                                     wintypes.UINT]
    shell32.Shell_NotifyIconW.restype = wintypes.BOOL
    shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.c_void_p]


class TrayIcon:
    """A notification-area icon driving two callbacks: open, and quit."""

    def __init__(self, on_open: Callable[[], None],
                 on_quit: Callable[[], None], tip: str = APP_TITLE) -> None:
        self.on_open = on_open
        self.on_quit = on_quit
        self.tip = tip
        self._hwnd: Optional[int] = None
        self._data: Optional[NOTIFYICONDATAW] = None
        # Held as an attribute, not a local: ctypes callbacks are freed
        # with the Python object, and a collected WNDPROC leaves Windows
        # calling into released memory the next time the icon is clicked.
        self._wndproc = WNDPROC(self._on_message)
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="MANA-Tray",
                                        daemon=True)

    # ---------- lifecycle ----------

    def start(self, timeout: float = 3.0) -> bool:
        self._thread.start()
        return self._ready.wait(timeout) and self._hwnd is not None

    def stop(self) -> None:
        if self._hwnd:
            try:
                ctypes.windll.shell32.Shell_NotifyIconW(NIM_DELETE,
                                                        ctypes.byref(self._data))
                ctypes.windll.user32.PostMessageW(self._hwnd, WM_DESTROY, 0, 0)
            except Exception:
                pass

    def notify(self, title: str, text: str) -> None:
        """A balloon over the icon. Best-effort, like everything here.

        Windows may suppress it -- notifications turned off, focus
        assist, a machine policy -- and there is no reliable way to find
        out that it did. So nothing may depend on the user having seen
        it: this says where the window went, and the icon itself is the
        durable version of that message.
        """
        if not self._hwnd or self._data is None:
            return
        try:
            self._data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_INFO
            self._data.szInfoTitle = title[:63]
            self._data.szInfo = text[:255]
            self._data.dwInfoFlags = NIIF_INFO
            ctypes.windll.shell32.Shell_NotifyIconW(NIM_MODIFY,
                                                    ctypes.byref(self._data))
            # Cleared straight after: NIF_INFO left set means the balloon
            # reappears on every later NIM_MODIFY, including the ones that
            # only update the tooltip.
            self._data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
            self._data.szInfo = ""
        except Exception:
            pass

    # ---------- the thread ----------

    def _icon(self) -> int:
        """The executable's own icon, or the generic application one.

        A frozen build has an icon resource; a source checkout runs under
        python.exe and gets Python's. Both are better than no icon at
        all, which Windows draws as an empty gap the user cannot click.
        """
        try:
            handle = ctypes.windll.shell32.ExtractIconW(
                None, ctypes.c_wchar_p(sys.executable), 0)
            if handle and handle > 1:
                return handle
        except Exception:
            pass
        return ctypes.windll.user32.LoadIconW(
            None, ctypes.cast(IDI_APPLICATION, wintypes.LPCWSTR))

    def _run(self) -> None:
        try:
            self._create()
        except Exception:
            self._hwnd = None
        finally:
            self._ready.set()
        if not self._hwnd:
            return
        message = wintypes.MSG()
        user32 = ctypes.windll.user32
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))

    def _create(self) -> None:
        _declare()
        user32 = ctypes.windll.user32
        instance = ctypes.windll.kernel32.GetModuleHandleW(None)

        window_class = WNDCLASSW()
        window_class.lpfnWndProc = self._wndproc
        window_class.hInstance = instance
        window_class.lpszClassName = "MANA_TrayWindow"
        if not user32.RegisterClassW(ctypes.byref(window_class)):
            # Already registered by an earlier instance in this process.
            # Harmless: CreateWindowExW below uses the existing class.
            pass

        # A real window, created and never shown, rather than a
        # message-only one: TrackPopupMenu needs a foreground-capable
        # window, and HWND_MESSAGE windows cannot become foreground, so
        # the context menu would refuse to close when clicked away from.
        self._hwnd = user32.CreateWindowExW(
            0, "MANA_TrayWindow", APP_TITLE, 0, 0, 0, 0, 0,
            None, None, instance, None)
        if not self._hwnd:
            raise OSError("CreateWindowExW failed")

        data = NOTIFYICONDATAW()
        data.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        data.hWnd = self._hwnd
        data.uID = 1
        data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        data.uCallbackMessage = WM_TRAY
        data.hIcon = self._icon()
        data.szTip = self.tip[:127]
        self._data = data
        if not ctypes.windll.shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(data)):
            raise OSError("Shell_NotifyIconW(NIM_ADD) failed")

    # ---------- messages ----------

    def _on_message(self, hwnd, message, wparam, lparam):
        user32 = ctypes.windll.user32
        if message == WM_TRAY:
            event = lparam & 0xFFFF
            if event in (WM_LBUTTONUP, WM_LBUTTONDBLCLK):
                self._safely(self.on_open)
            elif event == WM_RBUTTONUP:
                self._menu(hwnd)
            return 0
        if message == WM_COMMAND:
            command = wparam & 0xFFFF
            if command == IDM_OPEN:
                self._safely(self.on_open)
            elif command == IDM_QUIT:
                self._safely(self.on_quit)
            return 0
        if message == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def _menu(self, hwnd) -> None:
        user32 = ctypes.windll.user32
        menu = user32.CreatePopupMenu()
        user32.AppendMenuW(menu, MF_STRING, IDM_OPEN, "Открыть MANA")
        user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(menu, MF_STRING, IDM_QUIT, "Выход")
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        # Required, and the reason tray menus so often stick open: without
        # it the menu keeps the mouse capture after the user clicks
        # elsewhere and never dismisses.
        user32.SetForegroundWindow(hwnd)
        chosen = user32.TrackPopupMenu(
            menu, TPM_RIGHTBUTTON | TPM_RETURNCMD, point.x, point.y, 0, hwnd, None)
        user32.PostMessageW(hwnd, 0, 0, 0)
        user32.DestroyMenu(menu)
        if chosen == IDM_OPEN:
            self._safely(self.on_open)
        elif chosen == IDM_QUIT:
            self._safely(self.on_quit)

    @staticmethod
    def _safely(action: Callable[[], None]) -> None:
        """A failing callback must not take the message loop down with it.

        The loop is what makes the icon respond at all; if an exception
        escaped here, the icon would stay on screen and stop reacting,
        which looks exactly like a frozen application.
        """
        try:
            action()
        except Exception:
            pass


def start(on_open: Callable[[], None],
          on_quit: Callable[[], None]) -> Optional[TrayIcon]:
    """Best-effort tray icon. None means the window runs without one."""
    if sys.platform != "win32":
        return None
    icon = TrayIcon(on_open, on_quit)
    try:
        return icon if icon.start() else None
    except Exception:
        return None
