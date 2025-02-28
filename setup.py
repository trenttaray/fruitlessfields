from setuptools import setup

setup(
    name="fruitlessfields",
    options = {
        'build_apps': {
            'include_modules': ['direct.particles', 'numpy'],
            'exclude_patterns': [
                '**/*.tmp',
                '**/*.bak',
            ],
            'include_patterns': [
                '**/*.png',
                '**/*.bam',
                '**/*.egg',
                '**/*.json',
                '**/*.wav',
                '**/licenses/*',
                '**/README.md',
                '**/data/icon.ico',
            ],
            'gui_apps': {
                'fruitlessfields': 'main.py',
            },
            'log_filename': '$USER_APPDATA/fruitlessfields/output.log',
            'log_append': False,
            'plugins': [
                'pandagl',
                'p3openal_audio',
            ],
            'use_optimized_wheels': True,

        }
    }
)
