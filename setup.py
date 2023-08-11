import sys
import os
import re
import logging
import argparse
import subprocess
from setuptools import setup, find_packages
from contextlib import contextmanager

ROOT = os.path.dirname(__file__)


def get_long_description():
    with open(os.path.join(ROOT, 'README.md'), encoding='utf-8') as f:
        markdown_txt = f.read()
        return markdown_txt


def get_version():
    for line in open(os.path.join(ROOT, "version")).readlines():
        words = line.strip().split()
        if len(words) == 0:
            continue
        if words[0] == "version":
            return words[-1]
    return "0.0.0"


def get_requirements(filename):
    with open(os.path.join(ROOT, filename)) as f:
        return [line.rstrip() for line in f]


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-r', '--requirement', help='Optionally specify a different requirements file.',
    required=False
)
args, unparsed_args = parser.parse_known_args()
sys.argv[1:] = unparsed_args

if args.requirement is None:
    install_requires = get_requirements("requirements")
else:
    install_requires = get_requirements(args.requirement)


# entry_points = {
#     'console_scripts': [
#         'sockeye-average = sockeye.average:main',
#     ],
# }

args = dict(
    name='llmplus_tmp',

    version=get_version(),

    description='Useful modules for LLM',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",

    url='https://****',

    author='Xiongwen Wang',
    author_email='wangxiongwen1@huawei.com',
    maintainer_email='wangxiongwen1@huawei.com',

    license='****',

    python_requires='>=3.7',

    packages=find_packages(exclude=("test", "test.*")),

    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pillow'],

    extras_require={
        'optional': ['tensorboard', 'matplotlib'],
    },

    install_requires=install_requires,

    # entry_points=entry_points,

    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',

    ],

)


setup(**args)
