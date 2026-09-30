# -*- mode: python ; coding: utf-8 -*-

from importlib.metadata import PackageNotFoundError

from PyInstaller.utils.hooks import copy_metadata

# OpenInference's LangChain instrumentor gates itself on a dependency check that
# reads the *installed distribution metadata* of langchain-core. PyInstaller
# bundles modules but not their .dist-info directories, so inside the binary
# that lookup fails and BaseInstrumentor.instrument() logs
#     DependencyConflict: requested: "langchain_core >= 0.1.0" but found: "None"
# and returns without instrumenting anything. Nothing raises, so the agent
# starts, reports telemetry enabled, and exports only its own invoke_agent
# spans: no LLM/TOOL spans, no token counts. Shipping the metadata is what
# makes the auto-instrumentation engage in the frozen build. Distributions
# that are not installed (another provider's integration, or telemetry when
# its extra was dropped) are skipped.
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
