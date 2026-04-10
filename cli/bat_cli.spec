# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


template_root = Path("src") / "create" / "templates"
template_datas = [
    (
        str(file_path),
        str((Path("create") / "templates" / file_path.relative_to(template_root).parent).as_posix()),
    )
    for file_path in template_root.rglob("*")
    if file_path.is_file()
]

datas = collect_data_files("create") + template_datas
hiddenimports = collect_submodules("create") + collect_submodules("add") + collect_submodules("build") + collect_submodules("push") + collect_submodules("set")


a = Analysis(
    ["src/cli.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="bat",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)