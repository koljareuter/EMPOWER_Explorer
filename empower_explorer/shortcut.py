"""Create 'EMPOWER Explorer' shortcuts (Start menu, optionally Desktop) with the VLT 'M' icon.

    python -m empower_explorer.shortcut            # Start menu + Desktop
    python -m empower_explorer.shortcut --no-desktop

Windows does not let programs pin themselves to the taskbar.  Afterwards: Start -> type 'EMPOWER'
-> right-click -> 'Pin to taskbar' (or drag the desktop icon onto the taskbar).  The shortcut carries
the same AppUserModelID as the running app (branding.APP_ID), so the pinned button and the open
window share one taskbar button.  It starts this copy of the app: the .exe itself, or pythonw.exe
with EMPOWER_Explorer.py (no console window).

Because of the shared ID, a shortcut left pointing to a moved or deleted copy takes the icon away from
the taskbar button and leaves a pinned button that starts nothing; the app rewrites such a stale
shortcut at start (refresh_stale).
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from empower_explorer import APP_NAME
from empower_explorer.branding import APP_ID, launch_command

_PS = r'''
$ErrorActionPreference = 'Stop'
$lnk, $target, $arguments, $workdir, $icon, $appid, $desc = $args
if ($arguments -eq 'NOARGS') { $arguments = '' }   # the .exe: no arguments (an empty one would get lost)
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut($lnk)
$s.TargetPath = $target
$s.Arguments = $arguments
$s.WorkingDirectory = $workdir
$s.IconLocation = "$icon,0"
$s.Description = $desc
$s.Save()
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct PROPERTYKEY { public Guid fmtid; public uint pid; }
[StructLayout(LayoutKind.Explicit)]
public struct PROPVARIANT { [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public IntPtr p; [FieldOffset(16)] public IntPtr p2; }
[ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IPropertyStore {
    [PreserveSig] int GetCount(out uint count);
    [PreserveSig] int GetAt(uint index, out PROPERTYKEY key);
    [PreserveSig] int GetValue(ref PROPERTYKEY key, out PROPVARIANT value);
    [PreserveSig] int SetValue(ref PROPERTYKEY key, ref PROPVARIANT value);
    [PreserveSig] int Commit();
}
public static class EmpowerLnk {
    [DllImport("shell32.dll", PreserveSig = false)]
    static extern void SHGetPropertyStoreFromParsingName([MarshalAs(UnmanagedType.LPWStr)] string path,
        IntPtr bind, int flags, ref Guid riid, [MarshalAs(UnmanagedType.Interface)] out IPropertyStore store);
    public static void SetAppId(string path, string id) {
        Guid iid = new Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99");
        IPropertyStore store;
        SHGetPropertyStoreFromParsingName(path, IntPtr.Zero, 2, ref iid, out store);   // GPS_READWRITE
        PROPERTYKEY key = new PROPERTYKEY();
        key.fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"); key.pid = 5;   // System.AppUserModel.ID
        PROPVARIANT v = new PROPVARIANT();
        v.vt = 31;                                                                   // VT_LPWSTR
        v.p = Marshal.StringToCoTaskMemUni(id);
        try {
            int hr = store.SetValue(ref key, ref v); if (hr != 0) Marshal.ThrowExceptionForHR(hr);
            hr = store.Commit(); if (hr != 0) Marshal.ThrowExceptionForHR(hr);
        } finally {
            Marshal.FreeCoTaskMem(v.p);
            Marshal.ReleaseComObject(store);
        }
    }
}
"@
[EmpowerLnk]::SetAppId($lnk, $appid)
$sh = New-Object -ComObject Shell.Application
$item = $sh.Namespace((Split-Path $lnk)).ParseName((Split-Path $lnk -Leaf))
Write-Output ("AppUserModelID=" + $item.ExtendedProperty('System.AppUserModel.ID'))
'''


def _desktop():
    """The user's Desktop folder (also when OneDrive moved it)."""
    import ctypes
    buf = ctypes.create_unicode_buffer(260)
    if ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf) == 0 and buf.value:   # CSIDL_DESKTOPDIRECTORY
        return Path(buf.value)
    return None


def _folders(desktop):
    out = [Path(os.environ['APPDATA']) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs']
    if desktop and _desktop() is not None:
        out.append(_desktop())
    return out


def read_lnk(path):
    """{'target', 'arguments', 'workdir', 'icon'} of a .lnk file (MS-SHLLINK format); '' where absent."""
    b = Path(path).read_bytes()
    flags = struct.unpack_from('<I', b, 20)[0]
    out = dict(target='', arguments='', workdir='', icon='')
    pos = 76                                                     # after the header
    if flags & 0x1:                                              # HasLinkTargetIDList
        pos += 2 + struct.unpack_from('<H', b, pos)[0]
    if flags & 0x2:                                              # HasLinkInfo: the target's local path
        size, _hsize, info_flags, _volume, base = struct.unpack_from('<5I', b, pos)
        if info_flags & 0x1:
            start = pos + base
            out['target'] = b[start:b.index(b'\0', start)].decode('mbcs', 'replace')
        pos += size
    wide = bool(flags & 0x80)                                    # IsUnicode
    for flag, key in ((0x4, ''), (0x8, ''), (0x10, 'workdir'), (0x20, 'arguments'), (0x40, 'icon')):
        if flags & flag:                                         # counted strings in this order
            n = struct.unpack_from('<H', b, pos)[0] * (2 if wide else 1)
            text = b[pos + 2:pos + 2 + n].decode('utf-16-le' if wide else 'mbcs', 'replace')
            pos += 2 + n
            if key:
                out[key] = text
    return out


def is_stale(info):
    """True if a shortcut can no longer start the app: its program, the script it runs or its icon is gone."""
    paths = [info['target'], re.sub(r',-?\d+$', '', os.path.expandvars(info['icon']))]
    paths += re.findall(r'"([^"]+\.pyw?)"', info['arguments'])
    return not info['target'] or any(p and not os.path.exists(p) for p in paths)


def refresh_stale():
    """Rewrite the app's Start-menu and Desktop shortcuts that point to a moved or deleted copy, so that
    they start this one.  Returns the rewritten paths; never raises (runs in the background at start)."""
    if sys.platform != 'win32':
        return []
    done = []
    try:
        for folder in _folders(desktop=True):
            lnk = folder / f'{APP_NAME}.lnk'
            if lnk.is_file() and is_stale(read_lnk(lnk)):
                done += [path for path, _ in create(folders=[folder])]
    except Exception:
        pass
    return done


def create(desktop=True, folders=None):
    """Write the shortcuts (to `folders`, default Start menu and optionally Desktop) for this copy of the
    app; returns [(path, AppUserModelID read back)]."""
    if sys.platform != 'win32':
        raise SystemExit('Shortcuts are only made on Windows.')
    target, arguments, workdir, icon = launch_command()
    with tempfile.NamedTemporaryFile('w', suffix='.ps1', delete=False, encoding='utf-8-sig') as f:
        f.write(_PS)
        ps1 = f.name
    done = []
    try:
        for folder in folders or _folders(desktop):
            folder.mkdir(parents=True, exist_ok=True)
            lnk = folder / f'{APP_NAME}.lnk'
            r = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ps1,
                                str(lnk), target, arguments or 'NOARGS', workdir, icon, APP_ID,
                                'EMPOWER KMOS combined cubes explorer'],
                               capture_output=True, text=True, encoding='oem', errors='replace',
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))   # no console flash
            if r.returncode != 0:
                raise RuntimeError(f'Could not create {lnk}:\n{r.stderr.strip() or r.stdout.strip()}')
            appid = next((ln.split('=', 1)[1] for ln in r.stdout.splitlines() if ln.startswith('AppUserModelID=')), '')
            done.append((str(lnk), appid))
    finally:
        os.unlink(ps1)
    return done


if __name__ == '__main__':
    for path, appid in create(desktop='--no-desktop' not in sys.argv):
        print(f'created {path}  (AppUserModelID {appid or "not set"})')
    print("Pin it: Start -> type 'EMPOWER' -> right-click -> 'Pin to taskbar'.")
