# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec γιὰ ARK_DIAGNOSTIC.exe (Windows, αὐτόνομο).

Build:
    uv run pyinstaller ARK_DIAGNOSTIC.spec --clean --noconfirm

Output: dist/ARK_DIAGNOSTIC.exe (~50-80 MB μὲ numpy+scipy bundled)

Run:
    set PYTHONIOENCODING=utf-8
    dist\\ARK_DIAGNOSTIC.exe --equivariant --testbed=beta4

Στάδιο 3 — αὐτόνομο .exe χωρὶς Python installation στὸν target.
"""

block_cipher = None


a = Analysis(
    ['ark_main.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'ark_geometry', 'ark_irreps', 'ark_diagnostics', 'ark_state',
        'ark_wu', 'ark_wu_b10_probe', 'ark_wu_psi',
        'ark_hashimoto', 'ark_local_b10',
        'scipy.spatial.distance', 'scipy._lib.messagestream',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PIL', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ARK_DIAGNOSTIC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
