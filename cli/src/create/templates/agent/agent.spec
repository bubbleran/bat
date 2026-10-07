# -*- mode: python ; coding: utf-8 -*-

from importlib.metadata import PackageNotFoundError

from PyInstaller.utils.hooks import copy_metadata

# PyInstaller bundles modules but not their .dist-info metadata, which
# OpenInference checks before instrumenting LangChain: without it the binary
# exports no LLM/tool spans and no token counts. Missing distributions are skipped.
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

datas = [
	('src', 'src'),
	*telemetry_metadata,
]

a = Analysis(
	['__main__.py'],
	pathex=[],
	binaries=[],
	datas=datas,
	hiddenimports=[],
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
	name='__PROJECT_NAME__',
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
