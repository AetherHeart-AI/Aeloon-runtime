# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata


datas = collect_data_files("aeloon_core")
datas += collect_data_files(
    "aeloon_core",
    include_py_files=True,
    includes=["resources/skills/**/*.py"],
)
binaries = []
hiddenimports = []

for package in (
    "docx",
    "markitdown",
    "nodejs_wheel",
    "paddle",
    "paddleocr",
    "paddlex",
    "pdfplumber",
    "pypdf",
    "pypdfium2",
    "reportlab",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for distribution in (
    "markitdown",
    "nodejs-wheel",
    "nodejs-wheel-binaries",
    "paddleocr",
    "paddlepaddle",
    "pdfplumber",
    "pypdf",
    "pypdfium2",
    "python-docx",
    "reportlab",
):
    datas += copy_metadata(distribution, recursive=True)

a = Analysis(
    ["aeloon_core/__main__.py"],
    pathex=[],
    binaries=binaries,
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
    name="aeloon",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
