"""Create 'EMPOWER Explorer' shortcuts (Start menu, optionally Desktop) with the VLT 'M' icon.

    python -m empower_explorer.shortcut            # Start menu + Desktop
    python -m empower_explorer.shortcut --no-desktop

Windows does not let programs pin themselves to the taskbar.  Afterwards: Start -> type 'EMPOWER'
-> right-click -> 'Pin to taskbar' (or drag the desktop icon onto the taskbar).  The shortcut carries
the same AppUserModelID as the running app (branding.APP_ID), so the pinned button and the open
window share one taskbar button.  No console window: it starts pythonw.exe.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from empower_explorer import APP_NAME
from empower_explorer.branding import APP_ID, ICON_ICO

_PS = r'''
$ErrorActionPreference = 'Stop'
$lnk, $target, $arguments, $workdir, $icon, $appid, $desc = $args
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


def _folders(desktop):
    out = [Path(os.environ['APPDATA']) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs']
    if desktop:
        r = subprocess.run(['powershell', '-NoProfile', '-Command', "[Environment]::GetFolderPath('Desktop')"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            out.append(Path(r.stdout.strip()))
    return out


def create(desktop=True):
    """Write the shortcuts; returns [(path, AppUserModelID read back)]."""
    if sys.platform != 'win32':
        raise SystemExit('Shortcuts are only made on Windows.')
    repo = Path(__file__).resolve().parents[1]
    exe = Path(sys.executable).resolve()
    pythonw = exe.with_name('pythonw.exe')
    target = pythonw if pythonw.exists() else exe
    script = repo / 'EMPOWER_Explorer.py'
    with tempfile.NamedTemporaryFile('w', suffix='.ps1', delete=False, encoding='utf-8-sig') as f:
        f.write(_PS)
        ps1 = f.name
    done = []
    try:
        for folder in _folders(desktop):
            folder.mkdir(parents=True, exist_ok=True)
            lnk = folder / f'{APP_NAME}.lnk'
            r = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ps1,
                                str(lnk), str(target), f'"{script}"', str(repo), ICON_ICO, APP_ID,
                                'EMPOWER KMOS combined cubes explorer'],
                               capture_output=True, text=True)
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
