from setuptools import setup
import sys
sys.setrecursionlimit(3000)  # или даже выше, если нужно

APP = ['webcam_detect.py']
OPTIONS = {
    'argv_emulation': True,
    'plist': {
        'NSCameraUseContinuityCameraDeviceType': True
    }
}
"""com"""
setup(
    app=APP,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

