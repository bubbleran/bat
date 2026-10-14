# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


spec_root = Path(SPECPATH).resolve()
cli_src = spec_root / "src"
adk_src = spec_root.parent / "adk" / "src"

for source_path in (adk_src, cli_src):
    sys.path.insert(0, str(source_path))

from create.rendering import template_files

template_root = cli_src / "create" / "templates"
datas = [
    (str(template_root / name), (Path("create/templates") / name).parent.as_posix())
    for name in template_files(template_root)
]
datas += collect_data_files("matplotlib")

hiddenimports = []
hiddenimports += collect_submodules("langchain_core")
hiddenimports += collect_submodules("langchain.chat_models")
hiddenimports += collect_submodules("langchain_openai")
hiddenimports += collect_submodules("matplotlib")


a = Analysis(
    [str(cli_src / "cli.py")],
    pathex=[str(cli_src), str(adk_src)],
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
