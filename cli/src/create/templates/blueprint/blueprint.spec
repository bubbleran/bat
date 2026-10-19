# -*- mode: python ; coding: utf-8 -*-

from importlib.metadata import PackageNotFoundError
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata
def _metadata(*distributions):
    datas = []
    for distribution in distributions:
        try:
            datas += copy_metadata(distribution)
        except PackageNotFoundError:
            pass
    return datas


telemetry_metadata = _metadata(
    'langchain-core',
    'langchain',
    'openinference-instrumentation',
    'openinference-instrumentation-langchain',
    'opentelemetry-api',
    'opentelemetry-sdk',
    'langchain-anthropic',
    'langchain-deepseek',
    'langchain-nvidia-ai-endpoints',
    'langchain-ollama',
    'langchain-openai',
)

# The dispatcher imports agents by name, which PyInstaller cannot follow:
# every agent folder (an app.py next to its agent.json) is bundled.
agents = sorted(
    card.parent.name
    for card in Path(SPECPATH).glob('*/agent.json')
    if (card.parent / 'app.py').is_file()
)

a = Analysis(
	['__main__.py'],
	pathex=['.'],
	binaries=[],
	datas=telemetry_metadata,
	hiddenimports=agents,
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
	name='__BLUEPRINT_NAME__',
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
